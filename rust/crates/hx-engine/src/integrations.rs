//! Integration catalog — discovers active/available integrations from config.
//!
//! Returns a static catalog of known integrations with their status,
//! determined by checking for relevant environment variables or config keys.

use serde::{Deserialize, Serialize};

/// An entry in the integration catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEntry {
    /// Name of the integration (e.g., "openai", "slack").
    pub name: String,
    /// Category grouping (e.g., "ai_provider", "messaging", "storage").
    pub category: String,
    /// Current status: "active", "available", or "disabled".
    pub status: String,
    /// The environment variable or config key that activates this integration.
    pub config_key: Option<String>,
}

/// Collect the integration catalog with current status.
///
/// Checks environment variables to determine which integrations are active.
/// Integrations with a set env var are "active", those without are "available".
pub fn collect_integrations() -> Vec<IntegrationEntry> {
    let catalog = [
        ("anthropic", "ai_provider", Some("ANTHROPIC_API_KEY")),
        ("openai", "ai_provider", Some("OPENAI_API_KEY")),
        ("google", "ai_provider", Some("GOOGLE_API_KEY")),
        ("xai", "ai_provider", Some("XAI_API_KEY")),
        ("deepseek", "ai_provider", Some("DEEPSEEK_API_KEY")),
        ("slack", "messaging", Some("SLACK_BOT_TOKEN")),
        ("discord", "messaging", Some("DISCORD_BOT_TOKEN")),
        ("email", "messaging", Some("SMTP_HOST")),
        ("sqlite", "storage", None::<&str>),
        ("lancedb", "vector_storage", None),
        ("tantivy", "search", None),
    ];

    catalog
        .iter()
        .map(|(name, category, config_key)| {
            let status = match config_key {
                Some(key) => {
                    if std::env::var(key).is_ok() {
                        "active".to_string()
                    } else {
                        "available".to_string()
                    }
                }
                // Built-in integrations are always active
                None => "active".to_string(),
            };

            IntegrationEntry {
                name: name.to_string(),
                category: category.to_string(),
                status,
                config_key: config_key.map(|k| k.to_string()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_expected_entries() {
        let entries = collect_integrations();

        // Should have all catalog entries
        assert!(entries.len() >= 11);

        // Check AI providers exist
        let ai_providers: Vec<&IntegrationEntry> = entries
            .iter()
            .filter(|e| e.category == "ai_provider")
            .collect();
        assert_eq!(ai_providers.len(), 5);

        // Built-in integrations (no config key) should always be "active"
        let sqlite = entries.iter().find(|e| e.name == "sqlite").unwrap();
        assert_eq!(sqlite.status, "active");
        assert!(sqlite.config_key.is_none());

        let tantivy = entries.iter().find(|e| e.name == "tantivy").unwrap();
        assert_eq!(tantivy.status, "active");

        // Entries with config keys should exist
        let anthropic = entries.iter().find(|e| e.name == "anthropic").unwrap();
        assert_eq!(anthropic.config_key.as_deref(), Some("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn serialization_roundtrip() {
        let entry = IntegrationEntry {
            name: "test".into(),
            category: "testing".into(),
            status: "active".into(),
            config_key: Some("TEST_KEY".into()),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: IntegrationEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
        assert_eq!(deserialized.status, "active");
        assert_eq!(deserialized.config_key.as_deref(), Some("TEST_KEY"));
    }
}
