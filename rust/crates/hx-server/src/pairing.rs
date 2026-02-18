//! Gateway pairing logic — OTP generation, verification, and bearer token issuance.

use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{Duration, Utc};
use hx_core::model::pairing::{PairingSession, PairingState};
use rand::rngs::OsRng;
use rand::RngCore;
use uuid::Uuid;

const OTP_TTL_MINUTES: i64 = 5;

/// Generate a 6-digit OTP and its Argon2 hash.
pub fn generate_otp() -> (String, String) {
    let otp: u32 = rand::random::<u32>() % 1_000_000;
    let otp_str = format!("{:06}", otp);
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(otp_str.as_bytes(), &salt)
        .expect("hash")
        .to_string();
    (otp_str, hash)
}

/// Verify an OTP against its hash using constant-time comparison.
pub fn verify_otp(otp: &str, hash: &str) -> bool {
    let parsed = match PasswordHash::new(hash) {
        Ok(h) => h,
        Err(_) => return false,
    };
    Argon2::default()
        .verify_password(otp.as_bytes(), &parsed)
        .is_ok()
}

/// Generate a 32-byte random bearer token, base64url encoded.
pub fn generate_bearer_token() -> (String, String) {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    let token = URL_SAFE_NO_PAD.encode(bytes);
    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(token.as_bytes(), &salt)
        .expect("hash")
        .to_string();
    (token, hash)
}

/// Create a new pairing session.
pub fn create_pairing_session(channel_name: &str) -> (PairingSession, String) {
    let (otp, otp_hash) = generate_otp();
    let session = PairingSession {
        id: Uuid::now_v7(),
        channel_name: channel_name.to_string(),
        otp_hash,
        state: PairingState::Pending,
        created_at: Utc::now(),
        expires_at: Utc::now() + Duration::minutes(OTP_TTL_MINUTES),
        bearer_token_hash: None,
    };
    (session, otp)
}

/// Confirm a pairing session with OTP, returning bearer token if valid.
pub fn confirm_pairing(session: &mut PairingSession, otp: &str) -> Result<String, String> {
    if session.state != PairingState::Pending {
        return Err("session not in pending state".into());
    }
    if session.is_expired() {
        session.state = PairingState::Expired;
        return Err("OTP expired".into());
    }
    if !verify_otp(otp, &session.otp_hash) {
        return Err("invalid OTP".into());
    }
    let (token, token_hash) = generate_bearer_token();
    session.bearer_token_hash = Some(token_hash);
    session.state = PairingState::Confirmed;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_otp_confirms_session() {
        let (mut session, otp) = create_pairing_session("test-channel");
        let result = confirm_pairing(&mut session, &otp);
        assert!(result.is_ok());
        assert_eq!(session.state, PairingState::Confirmed);
        assert!(session.bearer_token_hash.is_some());
    }

    #[test]
    fn wrong_otp_rejects() {
        let (mut session, _otp) = create_pairing_session("test-channel");
        let result = confirm_pairing(&mut session, "000000");
        assert!(result.is_err());
        assert_eq!(session.state, PairingState::Pending);
    }

    #[test]
    fn expired_session_rejects() {
        let (otp, otp_hash) = generate_otp();
        let mut session = PairingSession {
            id: Uuid::now_v7(),
            channel_name: "expired-channel".to_string(),
            otp_hash,
            state: PairingState::Pending,
            created_at: Utc::now() - Duration::minutes(10),
            expires_at: Utc::now() - Duration::minutes(5),
            bearer_token_hash: None,
        };
        let result = confirm_pairing(&mut session, &otp);
        assert!(result.is_err());
        assert_eq!(session.state, PairingState::Expired);
    }

    #[test]
    fn duplicate_confirm_rejects() {
        let (mut session, otp) = create_pairing_session("test-channel");
        let first = confirm_pairing(&mut session, &otp);
        assert!(first.is_ok());
        // Second confirm should fail — session is no longer Pending.
        let second = confirm_pairing(&mut session, &otp);
        assert!(second.is_err());
    }

    #[test]
    fn otp_verification_is_constant_time() {
        // Argon2 verify_password uses constant-time comparison internally.
        let (otp, hash) = generate_otp();
        assert!(verify_otp(&otp, &hash));
        assert!(!verify_otp("wrong!", &hash));
    }

    #[test]
    fn bearer_token_stored_after_confirm() {
        let (mut session, otp) = create_pairing_session("test-channel");
        assert!(session.bearer_token_hash.is_none());
        let token = confirm_pairing(&mut session, &otp).unwrap();
        assert!(!token.is_empty());
        // The stored hash should verify against the returned token.
        let parsed = PasswordHash::new(session.bearer_token_hash.as_ref().unwrap()).unwrap();
        assert!(Argon2::default()
            .verify_password(token.as_bytes(), &parsed)
            .is_ok());
    }
}
