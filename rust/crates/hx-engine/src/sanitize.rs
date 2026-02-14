use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use zeroize::Zeroize;

/// Redacts secret values from output text.
///
/// Sorts secrets longest-first so that longer values are replaced before
/// shorter substrings of the same value. Also replaces base64-encoded
/// variants of each secret value.
pub struct OutputSanitizer;

impl OutputSanitizer {
    /// Replace all occurrences of each secret value (and its base64 forms)
    /// with `[REDACTED:key_name]`.
    ///
    /// Returns `(sanitized_output, was_any_redacted)`.
    pub fn sanitize(output: &str, secrets: &[(String, String)]) -> (String, bool) {
        if secrets.is_empty() || output.is_empty() {
            return (output.to_string(), false);
        }

        // Build replacement pairs: (needle, replacement).
        // Include both raw value and base64-encoded variants.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (key, value) in secrets {
            if value.is_empty() {
                continue;
            }
            let redacted = format!("[REDACTED:{key}]");

            // Raw value
            pairs.push((value.clone(), redacted.clone()));

            // Standard base64
            let b64_standard = STANDARD.encode(value.as_bytes());
            if b64_standard != *value {
                pairs.push((b64_standard, redacted.clone()));
            }

            // URL-safe base64
            let b64_url_safe = URL_SAFE_NO_PAD.encode(value.as_bytes());
            if b64_url_safe != *value && b64_url_safe != STANDARD.encode(value.as_bytes()) {
                pairs.push((b64_url_safe, redacted));
            }
        }

        // Sort longest-first so longer needles are replaced before shorter ones
        pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut result = output.to_string();
        let mut any_redacted = false;

        for (needle, replacement) in &pairs {
            if result.contains(needle.as_str()) {
                result = result.replace(needle.as_str(), replacement);
                any_redacted = true;
            }
        }

        // Zeroize plaintext secret needles before dropping
        for pair in &mut pairs {
            pair.0.zeroize();
        }

        (result, any_redacted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_replaces_raw_secret() {
        let secrets = vec![("API_KEY".to_string(), "sk-abc123".to_string())];
        let input = "Authorization: Bearer sk-abc123";
        let (output, redacted) = OutputSanitizer::sanitize(input, &secrets);
        assert!(redacted);
        assert_eq!(output, "Authorization: Bearer [REDACTED:API_KEY]");
        assert!(!output.contains("sk-abc123"));
    }

    #[test]
    fn sanitize_replaces_base64_encoded_secret() {
        let secrets = vec![("TOKEN".to_string(), "my-secret-token".to_string())];
        let b64 = STANDARD.encode("my-secret-token");
        let input = format!("encoded: {b64}");
        let (output, redacted) = OutputSanitizer::sanitize(&input, &secrets);
        assert!(redacted);
        assert!(output.contains("[REDACTED:TOKEN]"));
        assert!(!output.contains(&b64));
    }

    #[test]
    fn sanitize_no_secrets_returns_unchanged() {
        let (output, redacted) = OutputSanitizer::sanitize("hello world", &[]);
        assert!(!redacted);
        assert_eq!(output, "hello world");
    }

    #[test]
    fn sanitize_longer_secrets_replaced_first() {
        let secrets = vec![
            ("SHORT".to_string(), "abc".to_string()),
            ("LONG".to_string(), "abcdef".to_string()),
        ];
        let input = "value: abcdef";
        let (output, redacted) = OutputSanitizer::sanitize(input, &secrets);
        assert!(redacted);
        // The longer secret should be replaced, not the shorter one
        assert!(output.contains("[REDACTED:LONG]"));
    }

    #[test]
    fn sanitize_empty_output() {
        let secrets = vec![("KEY".to_string(), "value".to_string())];
        let (output, redacted) = OutputSanitizer::sanitize("", &secrets);
        assert!(!redacted);
        assert_eq!(output, "");
    }
}
