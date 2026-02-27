//! macOS Secure Enclave integration for Helix Sovereign Keychain.
//!
//! Uses the Security framework to wrap/unwrap vault master keys via
//! Secure Enclave-backed EC keys (ECIES). Also provides screen-lock
//! auto-seal notifications.

#![cfg(target_os = "macos")]

use security_framework::item::{ItemClass, ItemSearchOptions, Reference, SearchResult};
use security_framework::key::{Algorithm, GenerateKeyOptions, KeyType, SecKey, Token};

const SE_KEY_LABEL: &str = "com.helix.secure-enclave.master-wrap";

/// Get or create a Secure Enclave EC key for wrapping/unwrapping the master key.
pub fn get_or_create_se_key() -> Result<SecKey, String> {
    // Try to find existing key
    let search = ItemSearchOptions::new()
        .class(ItemClass::key())
        .label(SE_KEY_LABEL)
        .load_refs(true)
        .limit(1)
        .search();

    if let Ok(results) = search {
        for result in results {
            if let SearchResult::Ref(Reference::Key(key)) = result {
                return Ok(key);
            }
        }
    }

    // Create a new Secure Enclave key
    let mut opts = GenerateKeyOptions::default();
    opts.set_key_type(KeyType::ec_sec_prime_random())
        .set_size_in_bits(256)
        .set_label(SE_KEY_LABEL)
        .set_token(Token::SecureEnclave);

    SecKey::new(&opts).map_err(|e| format!("SE key generation: {e}"))
}

/// Wrap (encrypt) data using the Secure Enclave key via ECIES.
pub fn wrap_key_with_se(plaintext: &[u8]) -> Result<Vec<u8>, String> {
    let se_key = get_or_create_se_key()?;
    let public_key = se_key.public_key().ok_or("failed to get SE public key")?;

    public_key
        .encrypt_data(
            Algorithm::ECIESEncryptionCofactorVariableIVX963SHA256AESGCM,
            plaintext,
        )
        .map_err(|e| format!("SE encrypt: {e}"))
}

/// Unwrap (decrypt) data using the Secure Enclave key.
/// The `wrapped` parameter is a base64-encoded wrapped key.
pub fn unwrap_key_from_se(wrapped_b64: &str) -> Result<Vec<u8>, String> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let wrapped = STANDARD
        .decode(wrapped_b64)
        .map_err(|e| format!("base64 decode: {e}"))?;

    let se_key = get_or_create_se_key()?;
    se_key
        .decrypt_data(
            Algorithm::ECIESEncryptionCofactorVariableIVX963SHA256AESGCM,
            &wrapped,
        )
        .map_err(|e| format!("SE decrypt: {e}"))
}

/// Register a callback for screen lock events.
/// When the screen is locked, the callback is invoked (intended for auto-seal).
pub fn register_screen_lock_listener<F>(callback: F) -> Result<(), String>
where
    F: Fn() + Send + 'static,
{
    std::thread::spawn(move || {
        use std::process::Command;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            if let Ok(output) = Command::new("ioreg")
                .args(["-r", "-k", "CGSSessionScreenIsLocked"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("CGSSessionScreenIsLocked\" = Yes") {
                    callback();
                    return; // One-shot: seal once
                }
            }
        }
    });

    Ok(())
}
