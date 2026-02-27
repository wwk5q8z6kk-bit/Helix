//! Encrypted vault backup and restore.
//!
//! Format: `MVBK(4) || version(1) || salt(16) || nonce(12) || AES-256-GCM(data) || HMAC-SHA256(32)`

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::Argon2;
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::Sha256;
use std::path::Path;
use zeroize::Zeroizing;

const MAGIC: &[u8; 4] = b"MVBK";
const VERSION: u8 = 0x01;
const SALT_SIZE: usize = 16;
const NONCE_SIZE: usize = 12;
const KEY_SIZE: usize = 32;
const HMAC_SIZE: usize = 32;
const HEADER_SIZE: usize = 4 + 1 + SALT_SIZE + NONCE_SIZE; // 33

/// Export the vault database to an encrypted backup.
pub fn export_vault(db_path: &Path, password: &str) -> Result<Vec<u8>, String> {
    let db_data = std::fs::read(db_path).map_err(|e| format!("read db: {e}"))?;

    // Derive backup key
    let mut salt = [0u8; SALT_SIZE];
    OsRng.fill_bytes(&mut salt);
    let key = derive_key(password, &salt)?;

    // Encrypt
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let cipher = Aes256Gcm::new_from_slice(&*key).map_err(|e| format!("cipher init: {e}"))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, db_data.as_ref())
        .map_err(|e| format!("encrypt: {e}"))?;

    // Build output: MAGIC || VERSION || salt || nonce || ciphertext
    let mut out = Vec::with_capacity(HEADER_SIZE + ciphertext.len() + HMAC_SIZE);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);

    // HMAC over everything so far
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(&*key).map_err(|e| format!("hmac init: {e}"))?;
    mac.update(&out);
    let hmac_result = mac.finalize().into_bytes();
    out.extend_from_slice(&hmac_result);

    Ok(out)
}

/// Import (restore) an encrypted vault backup.
pub fn import_vault(data: &[u8], password: &str, db_path: &Path) -> Result<(), String> {
    if data.len() < HEADER_SIZE + 16 + HMAC_SIZE {
        return Err("backup data too short".to_string());
    }

    // Verify magic
    if &data[0..4] != MAGIC {
        return Err("invalid backup magic".to_string());
    }

    // Verify version
    if data[4] != VERSION {
        return Err(format!("unsupported backup version: {}", data[4]));
    }

    let salt = &data[5..5 + SALT_SIZE];
    let nonce_bytes = &data[5 + SALT_SIZE..5 + SALT_SIZE + NONCE_SIZE];
    let ciphertext = &data[HEADER_SIZE..data.len() - HMAC_SIZE];
    let stored_hmac = &data[data.len() - HMAC_SIZE..];

    // Derive key
    let key = derive_key(password, salt)?;

    // Verify HMAC
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(&*key).map_err(|e| format!("hmac init: {e}"))?;
    mac.update(&data[..data.len() - HMAC_SIZE]);
    let computed_hmac = mac.finalize().into_bytes();
    if !constant_time_eq(&computed_hmac, stored_hmac) {
        return Err("HMAC verification failed — wrong password or corrupted backup".to_string());
    }

    // Decrypt
    let cipher = Aes256Gcm::new_from_slice(&*key).map_err(|e| format!("cipher init: {e}"))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| format!("decrypt: {e}"))?;

    // Write to disk
    std::fs::write(db_path, plaintext).map_err(|e| format!("write db: {e}"))?;

    Ok(())
}

fn derive_key(password: &str, salt: &[u8]) -> Result<Zeroizing<[u8; KEY_SIZE]>, String> {
    let argon2 = Argon2::new(
        argon2::Algorithm::Argon2id,
        argon2::Version::V0x13,
        argon2::Params::new(65536, 3, 4, Some(KEY_SIZE))
            .map_err(|e| format!("argon2 params: {e}"))?,
    );
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    argon2
        .hash_password_into(password.as_bytes(), salt, key.as_mut())
        .map_err(|e| format!("key derivation: {e}"))?;
    Ok(key)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn backup_restore_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.sqlite");

        // Create a test "database" file
        let mut f = std::fs::File::create(&db_path).unwrap();
        f.write_all(b"test database content 12345").unwrap();
        drop(f);

        // Export
        let backup = export_vault(&db_path, "test-backup-password").unwrap();
        assert!(backup.len() > HEADER_SIZE + HMAC_SIZE);
        assert_eq!(&backup[0..4], b"MVBK");
        assert_eq!(backup[4], 0x01);

        // Tamper check
        let result = import_vault(&backup, "wrong-password", &db_path);
        assert!(result.is_err());

        // Restore
        let restore_path = dir.path().join("restored.sqlite");
        import_vault(&backup, "test-backup-password", &restore_path).unwrap();
        let restored = std::fs::read(&restore_path).unwrap();
        assert_eq!(restored, b"test database content 12345");
    }
}
