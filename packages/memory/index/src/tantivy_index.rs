use std::fmt::Write as _;
use std::io::{self, BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use tantivy::collector::TopDocs;
use tantivy::directory::error::{DeleteError, OpenReadError, OpenWriteError};
use tantivy::directory::{
    AntiCallToken, Directory, FileHandle, FileSlice, TerminatingWrite, WatchCallback,
    WatchCallbackList, WatchHandle, WritePtr,
};
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy};
use uuid::Uuid;

use hx_core::*;
use hx_storage::sealed_runtime::{runtime_root_key, sealed_mode_enabled};
use hx_storage::vault_crypto::VaultCrypto;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

const TANTIVY_SEALED_CONTEXT: &str = "sealed:tantivy:index";
const TANTIVY_SEALED_MAGIC: &[u8] = b"MVTIDX1";
const TANTIVY_ENCRYPTED_SUFFIX: &str = ".mvx";

fn io_invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn io_other(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

fn derive_tantivy_kek() -> MvResult<[u8; 32]> {
    let root = runtime_root_key().ok_or(HxError::VaultSealed)?;
    let mut crypto = VaultCrypto::new();
    crypto.set_master_key(Zeroizing::new(root));
    let key = crypto
        .derive_namespace_kek(TANTIVY_SEALED_CONTEXT)
        .map_err(|err| HxError::Storage(format!("derive tantivy key failed: {err}")))?;
    Ok(*key)
}

fn hashed_logical_name(path: &Path) -> String {
    let logical = path.to_string_lossy();
    let digest = Sha256::digest(logical.as_bytes());
    let mut out = String::with_capacity(digest.len() * 2 + TANTIVY_ENCRYPTED_SUFFIX.len());
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out.push_str(TANTIVY_ENCRYPTED_SUFFIX);
    out
}

fn seal_tantivy_bytes(kek: &[u8; 32], plaintext: &[u8]) -> io::Result<Vec<u8>> {
    let dek = VaultCrypto::generate_node_dek();
    let wrapped = VaultCrypto::wrap_node_dek(kek, &dek)
        .map_err(|err| io_other(format!("wrap tantivy DEK failed: {err}")))?;
    let ciphertext = VaultCrypto::aes_gcm_encrypt_pub(&dek, plaintext)
        .map_err(|err| io_other(format!("encrypt tantivy bytes failed: {err}")))?;

    let wrapped_bytes = wrapped.as_bytes();
    if wrapped_bytes.len() > u16::MAX as usize {
        return Err(io_invalid_data("wrapped DEK too large"));
    }

    let mut out =
        Vec::with_capacity(TANTIVY_SEALED_MAGIC.len() + 2 + wrapped_bytes.len() + ciphertext.len());
    out.extend_from_slice(TANTIVY_SEALED_MAGIC);
    out.extend_from_slice(&(wrapped_bytes.len() as u16).to_le_bytes());
    out.extend_from_slice(wrapped_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn open_tantivy_envelope(kek: &[u8; 32], sealed: &[u8]) -> io::Result<Vec<u8>> {
    let min_header_len = TANTIVY_SEALED_MAGIC.len() + 2;
    if sealed.len() < min_header_len {
        return Err(io_invalid_data("sealed tantivy file is too short"));
    }
    if !sealed.starts_with(TANTIVY_SEALED_MAGIC) {
        return Err(io_invalid_data("invalid tantivy sealed file magic"));
    }

    let mut len_bytes = [0u8; 2];
    len_bytes.copy_from_slice(&sealed[TANTIVY_SEALED_MAGIC.len()..min_header_len]);
    let wrapped_len = u16::from_le_bytes(len_bytes) as usize;
    let wrapped_start = min_header_len;
    let wrapped_end = wrapped_start + wrapped_len;
    if sealed.len() < wrapped_end {
        return Err(io_invalid_data(
            "sealed tantivy file is missing wrapped DEK",
        ));
    }

    let wrapped = std::str::from_utf8(&sealed[wrapped_start..wrapped_end])
        .map_err(|err| io_invalid_data(format!("wrapped DEK utf8 decode failed: {err}")))?;
    let ciphertext = &sealed[wrapped_end..];

    let dek = VaultCrypto::unwrap_node_dek(kek, wrapped)
        .map_err(|err| io_other(format!("unwrap tantivy DEK failed: {err}")))?;
    VaultCrypto::aes_gcm_decrypt_pub(&dek, ciphertext)
        .map_err(|err| io_other(format!("decrypt tantivy bytes failed: {err}")))
}

#[derive(Clone)]
struct EncryptedTantivyDirectory {
    root: PathBuf,
    kek: [u8; 32],
    watch_callbacks: Arc<WatchCallbackList>,
}

impl std::fmt::Debug for EncryptedTantivyDirectory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedTantivyDirectory")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

impl EncryptedTantivyDirectory {
    fn open(root: &Path) -> MvResult<Self> {
        std::fs::create_dir_all(root)
            .map_err(|err| HxError::Index(format!("create sealed tantivy dir: {err}")))?;

        Ok(Self {
            root: root.to_path_buf(),
            kek: derive_tantivy_kek()?,
            watch_callbacks: Arc::new(WatchCallbackList::default()),
        })
    }

    fn storage_path_for(&self, logical_path: &Path) -> PathBuf {
        self.root.join(hashed_logical_name(logical_path))
    }

    fn persist_plaintext(&self, logical_path: &Path, plaintext: &[u8]) -> io::Result<()> {
        let storage_path = self.storage_path_for(logical_path);
        let sealed = seal_tantivy_bytes(&self.kek, plaintext)?;

        if let Some(parent) = storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp_path = storage_path.with_extension("tmp");
        std::fs::write(&tmp_path, sealed)?;
        std::fs::rename(tmp_path, storage_path)?;
        Ok(())
    }

    fn read_plaintext(&self, logical_path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let storage_path = self.storage_path_for(logical_path);
        let bytes = std::fs::read(&storage_path).map_err(|err| {
            if err.kind() == io::ErrorKind::NotFound {
                OpenReadError::FileDoesNotExist(logical_path.to_path_buf())
            } else {
                OpenReadError::wrap_io_error(err, logical_path.to_path_buf())
            }
        })?;

        open_tantivy_envelope(&self.kek, &bytes)
            .map_err(|err| OpenReadError::wrap_io_error(err, logical_path.to_path_buf()))
    }

    fn broadcast_watchers(&self) {
        drop(self.watch_callbacks.broadcast());
    }
}

struct EncryptedTantivyWriter {
    path: PathBuf,
    directory: EncryptedTantivyDirectory,
    data: Cursor<Vec<u8>>,
    is_flushed: bool,
}

impl EncryptedTantivyWriter {
    fn new(path: PathBuf, directory: EncryptedTantivyDirectory) -> Self {
        Self {
            path,
            directory,
            data: Cursor::new(Vec::new()),
            is_flushed: true,
        }
    }
}

impl Drop for EncryptedTantivyWriter {
    fn drop(&mut self) {
        if !self.is_flushed {
            tracing::warn!(
                path = %self.path.display(),
                "encrypted tantivy writer dropped without flush"
            );
        }
    }
}

impl Write for EncryptedTantivyWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.is_flushed = false;
        self.data.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.directory
            .persist_plaintext(&self.path, self.data.get_ref())?;
        self.is_flushed = true;
        self.directory.broadcast_watchers();
        Ok(())
    }
}

impl TerminatingWrite for EncryptedTantivyWriter {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        self.flush()
    }
}

impl Directory for EncryptedTantivyDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_slice = self.open_read(path)?;
        Ok(Arc::new(file_slice))
    }

    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        let plaintext = self.read_plaintext(path)?;
        Ok(FileSlice::from(plaintext))
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        let storage_path = self.storage_path_for(path);
        match std::fs::remove_file(storage_path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                Err(DeleteError::FileDoesNotExist(path.to_path_buf()))
            }
            Err(err) => Err(DeleteError::IoError {
                io_error: Arc::new(err),
                filepath: path.to_path_buf(),
            }),
        }
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        let storage_path = self.storage_path_for(path);
        storage_path
            .try_exists()
            .map_err(|err| OpenReadError::wrap_io_error(err, path.to_path_buf()))
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        if self.exists(path).unwrap_or(false) {
            return Err(OpenWriteError::FileAlreadyExists(path.to_path_buf()));
        }

        self.persist_plaintext(path, &[])
            .map_err(|err| OpenWriteError::wrap_io_error(err, path.to_path_buf()))?;

        Ok(BufWriter::new(Box::new(EncryptedTantivyWriter::new(
            path.to_path_buf(),
            self.clone(),
        ))))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.read_plaintext(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.persist_plaintext(path, data)?;
        self.broadcast_watchers();
        Ok(())
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(self.watch_callbacks.subscribe(watch_callback))
    }
}

pub struct TantivyFullTextIndex {
    index: Index,
    writer: RwLock<IndexWriter>,
    reader: IndexReader,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    f_id: Field,
    f_title: Field,
    f_content: Field,
    f_tags: Field,
    f_kind: Field,
    f_namespace: Field,
}

impl TantivyFullTextIndex {
    pub fn open(path: &Path) -> MvResult<Self> {
        std::fs::create_dir_all(path)
            .map_err(|e| HxError::Index(format!("create index dir: {e}")))?;

        if sealed_mode_enabled() {
            if runtime_root_key().is_none() {
                // Startup in sealed mode occurs before an explicit unseal step, so
                // an in-memory index is required until runtime key material exists.
                return Self::open_in_memory();
            }
            let dir = EncryptedTantivyDirectory::open(path)?;
            return Self::open_with_dir(dir);
        }

        let dir = tantivy::directory::MmapDirectory::open(path)
            .map_err(|e| HxError::Index(format!("mmap dir: {e}")))?;

        Self::open_with_dir(dir)
    }

    pub fn open_in_memory() -> MvResult<Self> {
        Self::open_with_dir(tantivy::directory::RamDirectory::create())
    }

    fn open_with_dir<D: Directory + 'static>(dir: D) -> MvResult<Self> {
        let mut schema_builder = Schema::builder();
        let f_id = schema_builder.add_text_field("id", STRING | STORED);
        let f_title = schema_builder.add_text_field("title", TEXT);
        let f_content = schema_builder.add_text_field("content", TEXT);
        let f_tags = schema_builder.add_text_field("tags", TEXT);
        let f_kind = schema_builder.add_text_field("kind", STRING);
        let f_namespace = schema_builder.add_text_field("namespace", STRING);
        let schema = schema_builder.build();

        let index = Index::open_or_create(dir, schema.clone())
            .map_err(|e| HxError::Index(format!("open index: {e}")))?;

        let writer = index
            .writer(50_000_000) // 50MB heap
            .map_err(|e| HxError::Index(format!("create writer: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| HxError::Index(format!("create reader: {e}")))?;

        Ok(Self {
            index,
            writer: RwLock::new(writer),
            reader,
            schema,
            f_id,
            f_title,
            f_content,
            f_tags,
            f_kind,
            f_namespace,
        })
    }
}

impl FullTextIndex for TantivyFullTextIndex {
    fn index_node(&self, node: &KnowledgeNode) -> MvResult<()> {
        let writer = self
            .writer
            .write()
            .map_err(|e| HxError::Index(e.to_string()))?;

        // Remove existing document with same ID
        let id_term = tantivy::Term::from_field_text(self.f_id, &node.id.to_string());
        writer.delete_term(id_term);

        let mut doc = tantivy::TantivyDocument::new();
        doc.add_text(self.f_id, node.id.to_string());
        if let Some(ref title) = node.title {
            doc.add_text(self.f_title, title);
        }
        let attachment_search_blob = node
            .metadata
            .get("attachment_search_text")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if attachment_search_blob.trim().is_empty() {
            doc.add_text(self.f_content, &node.content);
        } else {
            let mut combined_content =
                String::with_capacity(node.content.len() + attachment_search_blob.len() + 1);
            combined_content.push_str(&node.content);
            combined_content.push('\n');
            combined_content.push_str(attachment_search_blob);
            doc.add_text(self.f_content, &combined_content);
        }
        doc.add_text(self.f_tags, node.tags.join(" "));
        doc.add_text(self.f_kind, node.kind.as_str());
        doc.add_text(self.f_namespace, &node.namespace);

        writer
            .add_document(doc)
            .map_err(|e| HxError::Index(e.to_string()))?;
        Ok(())
    }

    fn remove_node(&self, id: Uuid) -> MvResult<()> {
        let writer = self
            .writer
            .write()
            .map_err(|e| HxError::Index(e.to_string()))?;
        let id_term = tantivy::Term::from_field_text(self.f_id, &id.to_string());
        writer.delete_term(id_term);
        Ok(())
    }

    fn search(&self, query: &str, limit: usize) -> MvResult<Vec<(Uuid, f64)>> {
        let searcher = self.reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.f_title, self.f_content, self.f_tags]);

        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| HxError::Index(format!("parse query: {e}")))?;

        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(limit))
            .map_err(|e| HxError::Index(format!("search: {e}")))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| HxError::Index(format!("fetch doc: {e}")))?;

            if let Some(id_value) = doc.get_first(self.f_id) {
                if let Some(id_str) = id_value.as_str() {
                    if let Ok(uuid) = Uuid::parse_str(id_str) {
                        results.push((uuid, score as f64));
                    }
                }
            }
        }

        Ok(results)
    }

    fn commit(&self) -> MvResult<()> {
        let mut writer = self
            .writer
            .write()
            .map_err(|e| HxError::Index(e.to_string()))?;
        writer
            .commit()
            .map_err(|e| HxError::Index(format!("commit: {e}")))?;
        self.reader
            .reload()
            .map_err(|e| HxError::Index(format!("reload: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_storage::sealed_runtime::{
        clear_runtime_root_key, set_runtime_root_key, set_sealed_mode_enabled,
    };
    use tempfile::tempdir;
    use uuid::Uuid;

    struct SealedRuntimeReset;

    impl Drop for SealedRuntimeReset {
        fn drop(&mut self) {
            clear_runtime_root_key();
            set_sealed_mode_enabled(false);
        }
    }

    fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() || haystack.len() < needle.len() {
            return false;
        }
        haystack.windows(needle.len()).any(|window| window == needle)
    }

    #[test]
    fn test_index_and_search() {
        let idx = TantivyFullTextIndex::open_in_memory().unwrap();

        let node = KnowledgeNode::new(NodeKind::Fact, "Rust is a systems programming language")
            .with_title("About Rust")
            .with_tags(vec!["rust".into(), "programming".into()]);

        idx.index_node(&node).unwrap();
        idx.commit().unwrap();

        let results = idx.search("rust programming", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, node.id);
    }

    #[test]
    fn test_remove_node() {
        let idx = TantivyFullTextIndex::open_in_memory().unwrap();

        let node = KnowledgeNode::new(NodeKind::Fact, "temporary fact");
        let id = node.id;

        idx.index_node(&node).unwrap();
        idx.commit().unwrap();

        let results = idx.search("temporary", 5).unwrap();
        assert!(!results.is_empty());

        idx.remove_node(id).unwrap();
        idx.commit().unwrap();

        let results = idx.search("temporary", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_attachment_search_blob_is_indexed() {
        let idx = TantivyFullTextIndex::open_in_memory().unwrap();
        let mut node = KnowledgeNode::new(NodeKind::Fact, "Sprint plan")
            .with_title("Planning")
            .with_tags(vec!["ops".into()]);
        node.metadata.insert(
            "attachment_search_text".into(),
            "invoice reference 2026".into(),
        );

        idx.index_node(&node).unwrap();
        idx.commit().unwrap();

        let results = idx.search("invoice 2026", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, node.id);
    }

    #[test]
    fn tantivy_sealed_envelope_roundtrip() {
        let kek = [3u8; 32];
        let plaintext = b"tantivy payload";
        let sealed = seal_tantivy_bytes(&kek, plaintext).unwrap();
        assert_ne!(sealed, plaintext);

        let opened = open_tantivy_envelope(&kek, &sealed).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn sealed_tantivy_files_do_not_store_plaintext_payload() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        set_runtime_root_key([19u8; 32], false);

        let dir = tempdir().expect("tempdir");
        let idx = TantivyFullTextIndex::open(dir.path()).expect("sealed index open");
        let marker = format!("sealed-tantivy-marker-{}", Uuid::now_v7());
        let node = KnowledgeNode::new(NodeKind::Fact, marker.clone())
            .with_title(marker.clone())
            .with_tags(vec!["sealed".into()]);

        idx.index_node(&node).expect("index node");
        idx.commit().expect("commit");

        let mut scanned_files = 0usize;
        let mut stack = vec![dir.path().to_path_buf()];
        while let Some(path) = stack.pop() {
            for entry in std::fs::read_dir(&path).expect("read dir") {
                let entry = entry.expect("entry");
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    stack.push(entry_path);
                    continue;
                }
                if !entry_path.is_file() {
                    continue;
                }
                scanned_files += 1;
                let bytes = std::fs::read(&entry_path).expect("read file");
                assert!(
                    !bytes_contains(&bytes, marker.as_bytes()),
                    "found plaintext marker in {}",
                    entry_path.display()
                );
            }
        }

        assert!(scanned_files > 0, "expected sealed index files to be created");
    }
}
