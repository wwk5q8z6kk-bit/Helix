use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub data_dir: String,
    pub profile: OwnerProfileConfig,
    pub embedding: EmbeddingConfig,
    pub search: SearchConfig,
    pub graph: GraphConfig,
    pub ai: AiConfig,
    pub llm: LlmConfig,
    #[serde(default)]
    pub query_rewrite: QueryRewriteConfig,
    #[serde(default)]
    pub rerank: RerankConfig,
    #[serde(default)]
    pub session: SessionConfig,
    #[serde(default)]
    pub conversation: ConversationConfig,
    #[serde(default)]
    pub multihop: MultiHopConfig,
    #[serde(default)]
    pub local_llm: LocalLlmConfig,
    #[serde(default)]
    pub planning: PlanningConfig,
    pub email: EmailAdapterConfig,
    #[serde(default)]
    pub google_calendar: GoogleCalendarConfig,
    pub linking: LinkingConfig,
    pub daily_notes: DailyNotesConfig,
    pub recurrence: RecurrenceConfig,
    #[serde(default)]
    pub sealed_mode: bool,
    #[serde(default)]
    pub encryption_config: EncryptionConfig,
    #[serde(default)]
    pub encryption: EncryptionConfig,
    pub watcher: WatcherConfig,
    pub keychain: KeychainConfig,
    #[serde(default)]
    pub ai_sidecar: AiSidecarConfig,
}

/// Configuration for keychain lifecycle automation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeychainConfig {
    /// Interval between lifecycle checks in seconds (default: 3600 = 1 hour).
    pub lifecycle_check_interval_secs: u64,
    /// Whether automatic key rotation is enabled.
    pub auto_rotate_enabled: bool,
    /// How often to rotate the master key, in days (default: 90).
    pub auto_rotate_interval_days: u32,
    /// Grace period for old keys after rotation, in hours (default: 24).
    pub auto_rotate_grace_hours: u32,
}

impl Default for KeychainConfig {
    fn default() -> Self {
        Self {
            lifecycle_check_interval_secs: 3600,
            auto_rotate_enabled: false,
            auto_rotate_interval_days: 90,
            auto_rotate_grace_hours: 24,
        }
    }
}

/// Configuration for the Python AI sidecar proxy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSidecarConfig {
    /// Whether the AI sidecar proxy is enabled.
    pub enabled: bool,
    /// Base URL of the Python AI sidecar.
    pub base_url: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for AiSidecarConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://127.0.0.1:8100".into(),
            timeout_secs: 30,
        }
    }
}

/// Configuration for the Watcher Agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherConfig {
    /// Whether the watcher agent is enabled.
    pub enabled: bool,
    /// Interval between watcher cycles in seconds.
    pub interval_secs: u64,
    /// How far back to look for recently modified nodes (hours).
    pub lookback_hours: u64,
    /// Maximum nodes to scan per cycle.
    pub max_nodes_per_cycle: usize,
    /// Number of days after which pending proposals expire.
    pub expiry_days: u64,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 300,
            lookback_hours: 24,
            max_nodes_per_cycle: 50,
            expiry_days: 7,
        }
    }
}

/// Configuration for encryption at rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Whether encryption is enabled.
    pub enabled: bool,
    /// Argon2 memory parameter in KiB.
    pub argon2_memory_kib: u32,
    /// Argon2 iteration count.
    pub argon2_iterations: u32,
    /// Argon2 parallelism.
    pub argon2_parallelism: u32,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            argon2_memory_kib: 65536,
            argon2_iterations: 3,
            argon2_parallelism: 4,
        }
    }
}

impl EncryptionConfig {
    /// Create from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("HELIX_ENCRYPTION_ENABLED")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);

        let argon2_memory_kib = std::env::var("HELIX_ENCRYPTION_ARGON2_MEMORY_KIB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);

        let argon2_iterations = std::env::var("HELIX_ENCRYPTION_ARGON2_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let argon2_parallelism = std::env::var("HELIX_ENCRYPTION_ARGON2_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        Self {
            enabled,
            argon2_memory_kib,
            argon2_iterations,
            argon2_parallelism,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        let encryption = EncryptionConfig::default();
        Self {
            data_dir: shellexpand("~/.helix/data"),
            profile: OwnerProfileConfig::default(),
            embedding: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            graph: GraphConfig::default(),
            ai: AiConfig::default(),
            llm: LlmConfig::default(),
            query_rewrite: QueryRewriteConfig::default(),
            rerank: RerankConfig::default(),
            session: SessionConfig::default(),
            conversation: ConversationConfig::default(),
            multihop: MultiHopConfig::default(),
            local_llm: LocalLlmConfig::default(),
            planning: PlanningConfig::default(),
            email: EmailAdapterConfig::default(),
            google_calendar: GoogleCalendarConfig::default(),
            linking: LinkingConfig::default(),
            daily_notes: DailyNotesConfig::default(),
            recurrence: RecurrenceConfig::default(),
            sealed_mode: false,
            encryption_config: encryption.clone(),
            encryption,
            watcher: WatcherConfig::default(),
            keychain: KeychainConfig::default(),
            ai_sidecar: AiSidecarConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnerProfileConfig {
    pub display_name: String,
    pub primary_email: Option<String>,
    pub timezone: String,
    pub signature: Option<String>,
}

impl Default for OwnerProfileConfig {
    fn default() -> Self {
        Self {
            display_name: "Helix Owner".into(),
            primary_email: None,
            timezone: "UTC".into(),
            signature: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailAdapterConfig {
    pub enabled: bool,
    pub namespace: String,
    pub poll_interval_secs: u64,
    pub max_fetch: usize,
    pub max_attachment_bytes: usize,
    pub mark_seen: bool,
    pub imap_host: Option<String>,
    pub imap_port: u16,
    pub imap_username: Option<String>,
    pub imap_folder: String,
    pub imap_starttls: bool,
    pub smtp_host: Option<String>,
    pub smtp_port: u16,
    pub smtp_username: Option<String>,
    pub smtp_from: Option<String>,
    pub smtp_starttls: bool,
}

impl Default for EmailAdapterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: "default".into(),
            poll_interval_secs: 120,
            max_fetch: 20,
            max_attachment_bytes: 5 * 1024 * 1024,
            mark_seen: false,
            imap_host: None,
            imap_port: 993,
            imap_username: None,
            imap_folder: "INBOX".into(),
            imap_starttls: true,
            smtp_host: None,
            smtp_port: 587,
            smtp_username: None,
            smtp_from: None,
            smtp_starttls: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleCalendarConfig {
    pub enabled: bool,
    pub namespace: String,
    pub calendar_id: String,
    pub sync_interval_secs: u64,
    pub lookback_days: i64,
    pub lookahead_days: i64,
    pub max_results: usize,
    pub import_events: bool,
    pub export_events: bool,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub refresh_token: Option<String>,
}

impl Default for GoogleCalendarConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: "default".into(),
            calendar_id: "primary".into(),
            sync_interval_secs: 900,
            lookback_days: 30,
            lookahead_days: 90,
            max_results: 250,
            import_events: true,
            export_events: false,
            client_id: None,
            client_secret: None,
            refresh_token: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    /// Base URL for OpenAI-compatible embedding APIs.
    pub base_url: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "local_fastembed".into(),
            model: "bge-small-en-v1.5".into(),
            dimensions: 384,
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub default_limit: usize,
    pub default_strategy: String,
    pub min_score: f64,
    pub vector_weight: f64,
    pub fulltext_weight: f64,
    pub rrf_k: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            default_strategy: "hybrid".into(),
            min_score: 0.1,
            vector_weight: 0.6,
            fulltext_weight: 0.4,
            rrf_k: 60.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub default_traversal_depth: usize,
    pub graph_boost_factor: f64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            default_traversal_depth: 2,
            graph_boost_factor: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    pub auto_tagging_enabled: bool,
    pub auto_tagging_max_generated_tags: usize,
    pub auto_tagging_max_total_tags: usize,
    pub auto_tagging_similarity_seed_limit: usize,
    pub auto_tagging_min_token_length: usize,
    pub enrichment_enabled: bool,
    pub enrichment_model: String,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            auto_tagging_enabled: false,
            auto_tagging_max_generated_tags: 6,
            auto_tagging_max_total_tags: 12,
            auto_tagging_similarity_seed_limit: 8,
            auto_tagging_min_token_length: 4,
            enrichment_enabled: false,
            enrichment_model: "gpt-4o-mini".into(),
        }
    }
}

/// Configuration for the LLM provider used by assist/briefing endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub enabled: bool,
    pub auto_detect: bool,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_detect: true,
            base_url: "http://localhost:11434/v1".into(),
            model: "llama3.2".into(),
            max_tokens: 512,
            temperature: 0.3,
            timeout_secs: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkingConfig {
    pub auto_backlinks_enabled: bool,
    pub auto_backlinks_scan_limit: usize,
    pub auto_backlinks_max_targets: usize,
}

impl Default for LinkingConfig {
    fn default() -> Self {
        Self {
            auto_backlinks_enabled: true,
            auto_backlinks_scan_limit: 5_000,
            auto_backlinks_max_targets: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyNotesConfig {
    pub enabled: bool,
    pub midnight_scheduler_enabled: bool,
    pub namespace: String,
    pub title_template: String,
    pub content_template: String,
    pub default_importance: f64,
}

impl Default for DailyNotesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            midnight_scheduler_enabled: true,
            namespace: "journal".into(),
            title_template: "Daily Note {{date}}".into(),
            content_template: "\
## Top Priorities
- [ ] 

## Schedule
- Morning:
- Afternoon:
- Evening:

## Notes

## Wins
- 

## Blockers
- 

## Follow-up
- "
            .into(),
            default_importance: 0.6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrenceConfig {
    pub enabled: bool,
    pub scheduler_interval_secs: u64,
    pub max_instances_per_template: usize,
}

impl Default for RecurrenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheduler_interval_secs: 300,
            max_instances_per_template: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 Config Sections
// ---------------------------------------------------------------------------

/// Configuration for query rewriting before search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRewriteConfig {
    pub enabled: bool,
    pub default_strategy: String,
    pub max_sub_queries: usize,
    pub hyde_max_tokens: u32,
}

impl Default for QueryRewriteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_strategy: "none".into(),
            max_sub_queries: 3,
            hyde_max_tokens: 256,
        }
    }
}

/// Configuration for cross-encoder reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    pub enabled: bool,
    pub model_path: Option<String>,
    pub model_id: String,
    pub top_n: usize,
    pub min_score: f64,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".into(),
            top_n: 50,
            min_score: 0.0,
        }
    }
}

/// Configuration for session memory (conversation context for follow-up queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub enabled: bool,
    pub max_turns: usize,
    pub ttl_secs: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_turns: 20,
            ttl_secs: 3600,
        }
    }
}

/// Configuration for conversation-aware assist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    pub enabled: bool,
    pub expire_after_secs: u64,
    pub max_context_tokens: usize,
    pub summarize_after_turns: usize,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            expire_after_secs: 86400,
            max_context_tokens: 4096,
            summarize_after_turns: 10,
        }
    }
}

/// Configuration for multi-hop retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHopConfig {
    pub enabled: bool,
    pub max_hops: usize,
    pub token_budget: usize,
    pub results_per_hop: usize,
}

impl Default for MultiHopConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_hops: 2,
            token_budget: 2048,
            results_per_hop: 5,
        }
    }
}

/// Configuration for local LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLlmConfig {
    pub enabled: bool,
    pub model_path: Option<String>,
    pub model_id: Option<String>,
    pub max_ram_bytes: u64,
    pub gpu_layers: u32,
    pub context_size: u32,
    pub threads: u32,
    pub models_dir: String,
}

impl Default for LocalLlmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            model_id: None,
            max_ram_bytes: 3 * 1024 * 1024 * 1024,
            gpu_layers: 0,
            context_size: 2048,
            threads: 4,
            models_dir: shellexpand("~/.helix/models"),
        }
    }
}

/// Configuration for agent planning framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    pub enabled: bool,
    pub max_steps: usize,
    pub require_approval: bool,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_steps: 10,
            require_approval: true,
        }
    }
}

fn shellexpand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}
