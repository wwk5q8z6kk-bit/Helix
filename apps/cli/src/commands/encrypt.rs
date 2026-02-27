//! Encryption management commands for Helix CLI.

use anyhow::{bail, Context, Result};
use hx_storage::crypto::{EncryptionConfig, KeyManager};
use std::io::{self, Write};

use super::load_config;

const SQLITE_DB_FILE: &str = "helix.sqlite";

const SQL_COUNT_NODES: &str = "SELECT COUNT(*) FROM knowledge_nodes";
const SQL_COUNT_ENCRYPTED: &str =
    "SELECT COUNT(*) FROM knowledge_nodes WHERE content LIKE 'enc:v1:%'";
const SQL_SELECT_UNENCRYPTED: &str =
    "SELECT id, content, metadata_json FROM knowledge_nodes WHERE content NOT LIKE 'enc:v1:%' LIMIT ?";
const SQL_SELECT_ENCRYPTED: &str =
    "SELECT id, content, metadata_json FROM knowledge_nodes WHERE content LIKE 'enc:v1:%' LIMIT ?";
const SQL_UPDATE_NODE: &str =
    "UPDATE knowledge_nodes SET content = ?, metadata_json = ? WHERE id = ?";

/// Validate data_dir: canonicalize and ensure it stays within the user's home directory.
fn validate_data_dir(data_dir: &str) -> Result<std::path::PathBuf> {
    let path = std::path::Path::new(data_dir);
    // Create directory first if needed so canonicalize can resolve
    std::fs::create_dir_all(path)
        .with_context(|| format!("Failed to create data directory: {data_dir}"))?;
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Failed to resolve data directory: {data_dir}"))?;
    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    let home_path = std::path::Path::new(&home);
    if !canonical.starts_with(home_path) {
        bail!(
            "data_dir '{}' resolves outside home directory ({})",
            canonical.display(),
            home_path.display()
        );
    }
    Ok(canonical)
}

/// Initialize encryption for the vault.
pub async fn init(from_env: bool, config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir_path = validate_data_dir(&config.data_dir)?;
    let data_dir = data_dir_path.to_str().context("data_dir is not valid UTF-8")?;

    // Check if encryption is already initialized
    let key_marker_path = format!("{data_dir}/.encryption_initialized");
    if std::path::Path::new(&key_marker_path).exists() {
        bail!("Encryption is already initialized. Use `mv encrypt status` to check status.");
    }

    let password = if from_env {
        std::env::var("HELIX_ENCRYPTION_KEY")
            .context("HELIX_ENCRYPTION_KEY environment variable not set")?
    } else {
        prompt_password("Enter encryption password: ")?
    };

    if password.len() < 8 {
        bail!("Password must be at least 8 characters");
    }

    if !from_env {
        let confirm = prompt_password("Confirm encryption password: ")?;
        if password != confirm {
            bail!("Passwords do not match");
        }
    }

    // Generate a random salt and store it
    let salt = generate_salt();
    let salt_path = format!("{data_dir}/.encryption_salt");

    // Test key derivation
    let crypto_config = EncryptionConfig {
        enabled: true,
        argon2_memory_kib: config.encryption.argon2_memory_kib,
        argon2_iterations: config.encryption.argon2_iterations,
        argon2_parallelism: config.encryption.argon2_parallelism,
    };

    let mut key_manager = KeyManager::new(crypto_config);
    key_manager
        .derive_master_key(&password, salt.as_bytes())
        .context("Failed to derive encryption key")?;

    // Test encryption/decryption
    let test_data = "helix-encryption-test";
    let encrypted = key_manager
        .encrypt_string(test_data)
        .context("Failed to encrypt test data")?;
    let decrypted = key_manager
        .decrypt_string(&encrypted)
        .context("Failed to decrypt test data")?;

    if decrypted != test_data {
        bail!("Encryption verification failed");
    }

    // Save salt
    std::fs::write(&salt_path, &salt).context("Failed to save encryption salt")?;

    // Create marker file
    std::fs::write(&key_marker_path, "v1").context("Failed to create encryption marker")?;

    println!("Encryption initialized successfully.");
    println!();
    println!("IMPORTANT: Store your password securely. If lost, your data cannot be recovered.");
    println!();
    println!("To enable encryption, set these environment variables:");
    println!("  export HELIX_ENCRYPTION_ENABLED=true");
    println!("  export HELIX_ENCRYPTION_KEY=\"your-password\"");
    println!();
    println!("Or add to config.toml:");
    println!("  [encryption]");
    println!("  enabled = true");

    Ok(())
}

/// Migrate an existing unencrypted vault to encrypted storage.
pub async fn migrate(dry_run: bool, config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir_path = validate_data_dir(&config.data_dir)?;
    let data_dir = data_dir_path.to_str().context("data_dir is not valid UTF-8")?;

    // Check if encryption is initialized
    let key_marker_path = format!("{data_dir}/.encryption_initialized");
    if !std::path::Path::new(&key_marker_path).exists() {
        bail!("Encryption not initialized. Run `mv encrypt init` first.");
    }

    // Get the encryption key
    let password = std::env::var("HELIX_ENCRYPTION_KEY")
        .context("HELIX_ENCRYPTION_KEY environment variable required for migration")?;

    let salt_path = format!("{data_dir}/.encryption_salt");
    let salt = std::fs::read_to_string(&salt_path).context("Failed to read encryption salt")?;

    let crypto_config = EncryptionConfig {
        enabled: true,
        argon2_memory_kib: config.encryption.argon2_memory_kib,
        argon2_iterations: config.encryption.argon2_iterations,
        argon2_parallelism: config.encryption.argon2_parallelism,
    };

    let mut key_manager = KeyManager::new(crypto_config);
    key_manager
        .derive_master_key(&password, salt.as_bytes())
        .context("Failed to derive encryption key")?;

    // Find SQLite database
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");
    if !std::path::Path::new(&db_path).exists() {
        println!("No database found at {db_path}. Nothing to migrate.");
        return Ok(());
    }

    if dry_run {
        println!("Dry run mode - no changes will be made");
        println!();
    }

    // Connect to database and count records
    let conn = rusqlite::Connection::open(&db_path).context("Failed to open database")?;

    let node_count: i64 = conn
        .query_row(SQL_COUNT_NODES, [], |row| row.get(0))
        .unwrap_or(0);

    println!("Found {node_count} nodes to migrate");

    if dry_run {
        println!();
        println!("Would encrypt:");
        println!("  - {node_count} node content fields");
        println!("  - Associated metadata fields");
        println!();
        println!("Run without --dry-run to perform migration.");
        return Ok(());
    }

    if node_count == 0 {
        println!("No nodes to migrate.");
        return Ok(());
    }

    // Confirm migration
    print!("Proceed with encryption migration? [y/N] ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Migration cancelled.");
        return Ok(());
    }

    println!("Migrating nodes...");

    // Migrate nodes in batches
    let batch_size = 100;
    let mut migrated = 0;

    loop {
        let mut stmt = conn.prepare(SQL_SELECT_UNENCRYPTED)?;

        let rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([batch_size], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        if rows.is_empty() {
            break;
        }

        for (id, content, metadata) in rows {
            let encrypted_content = key_manager
                .encrypt_string(&content)
                .context("Failed to encrypt content")?;
            let encrypted_content = format!("enc:v1:{encrypted_content}");

            let encrypted_metadata = if let Some(meta) = metadata {
                let encrypted = key_manager
                    .encrypt_string(&meta)
                    .context("Failed to encrypt metadata")?;
                Some(format!("enc:v1:{encrypted}"))
            } else {
                None
            };

            conn.execute(
                SQL_UPDATE_NODE,
                rusqlite::params![encrypted_content, encrypted_metadata, id],
            )?;

            migrated += 1;
        }

        print!("\rMigrated {migrated}/{node_count} nodes...");
        io::stdout().flush()?;
    }

    println!("\rMigrated {migrated} nodes successfully.     ");

    // Create migration marker
    let migration_marker = format!("{data_dir}/.encryption_migrated");
    std::fs::write(&migration_marker, chrono::Utc::now().to_rfc3339())?;

    println!();
    println!("Migration complete. Your vault is now encrypted.");

    Ok(())
}

/// Decrypt the vault (disable encryption).
pub async fn decrypt(confirm: bool, config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir_path = validate_data_dir(&config.data_dir)?;
    let data_dir = data_dir_path.to_str().context("data_dir is not valid UTF-8")?;

    // Check if encryption is initialized
    let key_marker_path = format!("{data_dir}/.encryption_initialized");
    if !std::path::Path::new(&key_marker_path).exists() {
        println!("Encryption is not initialized.");
        return Ok(());
    }

    if !confirm {
        println!("This will decrypt all data in your vault.");
        println!("Run with --confirm to proceed.");
        return Ok(());
    }

    // Get the encryption key
    let password = std::env::var("HELIX_ENCRYPTION_KEY")
        .context("HELIX_ENCRYPTION_KEY environment variable required for decryption")?;

    let salt_path = format!("{data_dir}/.encryption_salt");
    let salt = std::fs::read_to_string(&salt_path).context("Failed to read encryption salt")?;

    let crypto_config = EncryptionConfig {
        enabled: true,
        argon2_memory_kib: config.encryption.argon2_memory_kib,
        argon2_iterations: config.encryption.argon2_iterations,
        argon2_parallelism: config.encryption.argon2_parallelism,
    };

    let mut key_manager = KeyManager::new(crypto_config);
    key_manager
        .derive_master_key(&password, salt.as_bytes())
        .context("Failed to derive encryption key")?;

    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");
    if !std::path::Path::new(&db_path).exists() {
        println!("No database found. Removing encryption markers.");
        std::fs::remove_file(&key_marker_path).ok();
        std::fs::remove_file(&salt_path).ok();
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path).context("Failed to open database")?;

    let encrypted_count: i64 = conn
        .query_row(SQL_COUNT_ENCRYPTED, [], |row| row.get(0))
        .unwrap_or(0);

    if encrypted_count == 0 {
        println!("No encrypted nodes found. Removing encryption markers.");
        std::fs::remove_file(&key_marker_path).ok();
        std::fs::remove_file(&salt_path).ok();
        return Ok(());
    }

    println!("Decrypting {encrypted_count} nodes...");

    let batch_size = 100;
    let mut decrypted = 0;

    loop {
        let mut stmt = conn.prepare(SQL_SELECT_ENCRYPTED)?;

        let rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([batch_size], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        if rows.is_empty() {
            break;
        }

        for (id, content, metadata) in rows {
            let plain_content = if let Some(encrypted) = content.strip_prefix("enc:v1:") {
                key_manager
                    .decrypt_string(encrypted)
                    .context("Failed to decrypt content")?
            } else {
                content
            };

            let plain_metadata = if let Some(meta) = metadata {
                if let Some(encrypted) = meta.strip_prefix("enc:v1:") {
                    Some(
                        key_manager
                            .decrypt_string(encrypted)
                            .context("Failed to decrypt metadata")?,
                    )
                } else {
                    Some(meta)
                }
            } else {
                None
            };

            conn.execute(
                SQL_UPDATE_NODE,
                rusqlite::params![plain_content, plain_metadata, id],
            )?;

            decrypted += 1;
        }

        print!("\rDecrypted {decrypted}/{encrypted_count} nodes...");
        io::stdout().flush()?;
    }

    println!("\rDecrypted {decrypted} nodes successfully.     ");

    // Remove encryption markers
    std::fs::remove_file(&key_marker_path).ok();
    std::fs::remove_file(&salt_path).ok();
    let migration_marker = format!("{data_dir}/.encryption_migrated");
    std::fs::remove_file(&migration_marker).ok();

    println!();
    println!("Vault decrypted. Encryption has been disabled.");

    Ok(())
}

/// Check encryption status.
pub async fn status(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir_path = validate_data_dir(&config.data_dir)?;
    let data_dir = data_dir_path.to_str().context("data_dir is not valid UTF-8")?;

    let key_marker_path = format!("{data_dir}/.encryption_initialized");
    let salt_path = format!("{data_dir}/.encryption_salt");
    let migration_marker = format!("{data_dir}/.encryption_migrated");

    let initialized = std::path::Path::new(&key_marker_path).exists();
    let salt_exists = std::path::Path::new(&salt_path).exists();
    let migrated = std::path::Path::new(&migration_marker).exists();

    println!("Encryption Status");
    println!("=================");
    println!();
    println!("Data directory: {data_dir}");
    println!();

    if !initialized {
        println!("Status: NOT INITIALIZED");
        println!();
        println!("Run `mv encrypt init` to set up encryption.");
        return Ok(());
    }

    println!("Initialized: yes");
    println!(
        "Salt file:   {}",
        if salt_exists { "present" } else { "MISSING" }
    );
    println!("Migrated:    {}", if migrated { "yes" } else { "no" });

    // Check environment
    let env_enabled = std::env::var("HELIX_ENCRYPTION_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);
    let env_key_set = std::env::var("HELIX_ENCRYPTION_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false);

    println!();
    println!("Environment:");
    println!(
        "  HELIX_ENCRYPTION_ENABLED: {}",
        if env_enabled { "true" } else { "false" }
    );
    println!(
        "  HELIX_ENCRYPTION_KEY:     {}",
        if env_key_set { "set" } else { "not set" }
    );

    // Check database
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");
    if std::path::Path::new(&db_path).exists() {
        let conn = rusqlite::Connection::open(&db_path)?;

        let total_nodes: i64 = conn
            .query_row(SQL_COUNT_NODES, [], |row| row.get(0))
            .unwrap_or(0);

        let encrypted_nodes: i64 = conn
            .query_row(SQL_COUNT_ENCRYPTED, [], |row| row.get(0))
            .unwrap_or(0);

        println!();
        println!("Database:");
        println!("  Total nodes:     {total_nodes}");
        println!("  Encrypted nodes: {encrypted_nodes}");

        if total_nodes > 0 {
            let pct = (encrypted_nodes as f64 / total_nodes as f64) * 100.0;
            println!("  Encryption:      {pct:.1}%");
        }
    } else {
        println!();
        println!("Database: not found");
    }

    Ok(())
}

fn prompt_password(prompt: &str) -> Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;

    // Try to disable echo for password input
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let stdin_fd = io::stdin().as_raw_fd();
        let mut termios = termios::Termios::from_fd(stdin_fd).ok();
        if let Some(ref mut t) = termios {
            let original = *t;
            t.c_lflag &= !termios::ECHO;
            let _ = termios::tcsetattr(stdin_fd, termios::TCSANOW, t);

            let mut password = String::new();
            io::stdin().read_line(&mut password)?;
            println!();

            let _ = termios::tcsetattr(stdin_fd, termios::TCSANOW, &original);
            return Ok(password.trim().to_string());
        }
    }

    // Fallback: read with echo
    let mut password = String::new();
    io::stdin().read_line(&mut password)?;
    Ok(password.trim().to_string())
}

fn generate_salt() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: [u8; 32] = rng.gen();
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use hx_storage::crypto::{EncryptionConfig, KeyManager};

    fn test_crypto_config() -> EncryptionConfig {
        EncryptionConfig {
            enabled: true,
            argon2_memory_kib: 256,
            argon2_iterations: 1,
            argon2_parallelism: 1,
        }
    }

    // ── generate_salt ────────────────────────────────────────────────

    #[test]
    fn generate_salt_returns_valid_base64() {
        let salt = generate_salt();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&salt)
            .expect("salt should be valid base64");
        assert_eq!(decoded.len(), 32, "salt should decode to 32 bytes");
    }

    #[test]
    fn generate_salt_produces_unique_values() {
        let a = generate_salt();
        let b = generate_salt();
        assert_ne!(a, b, "two salts should not be identical");
    }

    // ── validate_data_dir ────────────────────────────────────────────

    #[test]
    fn validate_data_dir_accepts_path_under_home() {
        let home = std::env::var("HOME").expect("HOME must be set");
        let tmp = format!("{home}/.helix-test-encrypt-{}", std::process::id());
        let result = validate_data_dir(&tmp);
        assert!(result.is_ok(), "path under HOME should be accepted");
        // Clean up
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validate_data_dir_rejects_path_outside_home() {
        let result = validate_data_dir("/tmp/helix-escape-test");
        assert!(result.is_err(), "path outside HOME should be rejected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("outside home directory"),
            "error should mention outside home: {err}"
        );
        // Clean up in case it was created before the check
        let _ = std::fs::remove_dir_all("/tmp/helix-escape-test");
    }

    // ── encrypt / decrypt round-trip ─────────────────────────────────

    #[test]
    fn encrypt_decrypt_round_trip() {
        let config = test_crypto_config();
        let mut km = KeyManager::new(config);
        km.derive_master_key("test-password-123", b"test-salt-value")
            .expect("key derivation should succeed");

        let plaintext = "hello helix encryption";
        let encrypted = km
            .encrypt_string(plaintext)
            .expect("encryption should succeed");
        assert_ne!(encrypted, plaintext, "encrypted output should differ from plaintext");

        let decrypted = km
            .decrypt_string(&encrypted)
            .expect("decryption should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_produces_different_ciphertexts() {
        let config = test_crypto_config();
        let mut km = KeyManager::new(config);
        km.derive_master_key("test-password-123", b"salt-long-enough")
            .expect("key derivation");

        let a = km.encrypt_string("same input").expect("encrypt a");
        let b = km.encrypt_string("same input").expect("encrypt b");
        assert_ne!(a, b, "encrypting the same plaintext twice should produce different ciphertexts (random nonce)");
    }

    #[test]
    fn decrypt_with_wrong_key_fails() {
        let config = test_crypto_config();

        let mut km1 = KeyManager::new(config.clone());
        km1.derive_master_key("password-one", b"salt-long-enough")
            .expect("derive key 1");
        let encrypted = km1.encrypt_string("secret data").expect("encrypt");

        let mut km2 = KeyManager::new(config);
        km2.derive_master_key("password-two", b"salt-long-enough")
            .expect("derive key 2");
        let result = km2.decrypt_string(&encrypted);
        assert!(result.is_err(), "decryption with wrong key should fail");
    }

    // ── migrate_encrypted content format ─────────────────────────────

    #[test]
    fn encrypted_content_prefix_format() {
        let config = test_crypto_config();
        let mut km = KeyManager::new(config);
        km.derive_master_key("pw", b"salt-long-enough").expect("derive");

        let encrypted = km.encrypt_string("data").expect("encrypt");
        let wrapped = format!("enc:v1:{encrypted}");
        assert!(wrapped.starts_with("enc:v1:"), "wrapped content should have enc:v1: prefix");

        // Verify the strip_prefix + decrypt pattern used in decrypt()
        let inner = wrapped.strip_prefix("enc:v1:").expect("strip prefix");
        let recovered = km.decrypt_string(inner).expect("decrypt inner");
        assert_eq!(recovered, "data");
    }

    // ── SQL constants ────────────────────────────────────────────────

    #[test]
    fn sql_constants_are_valid_sqlite() {
        let conn = rusqlite::Connection::open_in_memory().expect("open in-memory db");
        conn.execute_batch(
            "CREATE TABLE knowledge_nodes (id TEXT PRIMARY KEY, content TEXT, metadata_json TEXT)",
        )
        .expect("create table");

        // Each const SQL should prepare without error
        conn.prepare(SQL_COUNT_NODES).expect("SQL_COUNT_NODES should parse");
        conn.prepare(SQL_COUNT_ENCRYPTED).expect("SQL_COUNT_ENCRYPTED should parse");
        conn.prepare(SQL_SELECT_UNENCRYPTED).expect("SQL_SELECT_UNENCRYPTED should parse");
        conn.prepare(SQL_SELECT_ENCRYPTED).expect("SQL_SELECT_ENCRYPTED should parse");
        conn.prepare(SQL_UPDATE_NODE).expect("SQL_UPDATE_NODE should parse");
    }

    // ── database migrate + decrypt integration ───────────────────────

    #[test]
    fn migrate_and_decrypt_database_round_trip() {
        let config = test_crypto_config();
        let mut km = KeyManager::new(config);
        km.derive_master_key("integration-pw", b"integration-salt")
            .expect("derive");

        let conn = rusqlite::Connection::open_in_memory().expect("open db");
        conn.execute_batch(
            "CREATE TABLE knowledge_nodes (id TEXT PRIMARY KEY, content TEXT, metadata_json TEXT)",
        )
        .expect("create table");

        // Insert test rows
        conn.execute(
            "INSERT INTO knowledge_nodes VALUES (?1, ?2, ?3)",
            rusqlite::params!["id-1", "plaintext content", r#"{"key":"value"}"#],
        )
        .expect("insert row 1");
        conn.execute(
            "INSERT INTO knowledge_nodes VALUES (?1, ?2, ?3)",
            rusqlite::params!["id-2", "another node", rusqlite::types::Null],
        )
        .expect("insert row 2");

        // ── Simulate migrate: encrypt all unencrypted nodes ──
        let batch_size = 100i64;
        let mut stmt = conn.prepare(SQL_SELECT_UNENCRYPTED).expect("prepare select");
        let rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([batch_size], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .expect("query")
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(rows.len(), 2);

        for (id, content, metadata) in &rows {
            let enc_content = format!("enc:v1:{}", km.encrypt_string(content).expect("encrypt content"));
            let enc_metadata = metadata.as_ref().map(|m| {
                format!("enc:v1:{}", km.encrypt_string(m).expect("encrypt meta"))
            });
            conn.execute(SQL_UPDATE_NODE, rusqlite::params![enc_content, enc_metadata, id])
                .expect("update");
        }

        // Verify all nodes are now encrypted
        let encrypted_count: i64 = conn
            .query_row(SQL_COUNT_ENCRYPTED, [], |row| row.get(0))
            .expect("count encrypted");
        assert_eq!(encrypted_count, 2);

        // ── Simulate decrypt: decrypt all encrypted nodes ──
        let mut stmt = conn.prepare(SQL_SELECT_ENCRYPTED).expect("prepare select enc");
        let enc_rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([batch_size], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .expect("query enc")
            .filter_map(|r| r.ok())
            .collect();

        for (id, content, metadata) in &enc_rows {
            let plain = content
                .strip_prefix("enc:v1:")
                .map(|inner| km.decrypt_string(inner).expect("decrypt content"))
                .unwrap_or_else(|| content.clone());

            let plain_meta = metadata.as_ref().map(|m| {
                m.strip_prefix("enc:v1:")
                    .map(|inner| km.decrypt_string(inner).expect("decrypt meta"))
                    .unwrap_or_else(|| m.clone())
            });

            conn.execute(SQL_UPDATE_NODE, rusqlite::params![plain, plain_meta, id])
                .expect("update decrypt");
        }

        // Verify round-trip: content should match originals
        let recovered: String = conn
            .query_row("SELECT content FROM knowledge_nodes WHERE id = 'id-1'", [], |r| r.get(0))
            .expect("query id-1");
        assert_eq!(recovered, "plaintext content");

        let recovered_meta: String = conn
            .query_row("SELECT metadata_json FROM knowledge_nodes WHERE id = 'id-1'", [], |r| r.get(0))
            .expect("query id-1 meta");
        assert_eq!(recovered_meta, r#"{"key":"value"}"#);

        let recovered_null: Option<String> = conn
            .query_row("SELECT metadata_json FROM knowledge_nodes WHERE id = 'id-2'", [], |r| r.get(0))
            .expect("query id-2 meta");
        assert!(recovered_null.is_none(), "null metadata should remain null");
    }
}
