use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(
    name = "hx",
    about = "Helix — unified AI platform",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Config file path
    #[arg(long, global = true, default_value = "~/.helix/config.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a knowledge node
    Store {
        /// The content to store
        content: String,

        /// Node kind
        #[arg(long, short, default_value = "fact")]
        kind: String,

        /// Title
        #[arg(long, short)]
        title: Option<String>,

        /// Source
        #[arg(long, short)]
        source: Option<String>,

        /// Tags (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,

        /// Namespace
        #[arg(long, short, default_value = "default")]
        ns: String,

        /// Importance (0.0 - 1.0)
        #[arg(long, short, default_value = "0.5")]
        importance: f64,
    },

    /// Recall knowledge
    Recall {
        /// The query text
        query: String,

        /// Max results
        #[arg(long, short, default_value = "5")]
        limit: usize,

        /// Search strategy (hybrid, vector, fulltext, graph)
        #[arg(long, default_value = "hybrid")]
        strategy: String,

        /// Min score threshold
        #[arg(long, default_value = "0.0")]
        min_score: f64,

        /// Namespace filter
        #[arg(long, short)]
        ns: Option<String>,
    },

    /// Full-text search
    Search {
        /// The search query
        query: String,

        /// Max results
        #[arg(long, short, default_value = "10")]
        limit: usize,

        /// Search type (fulltext, vector, hybrid)
        #[arg(long, short = 't', default_value = "fulltext")]
        r#type: String,
    },

    /// Knowledge graph operations
    Graph {
        #[command(subcommand)]
        action: GraphAction,
    },

    /// Import data from external sources
    Import {
        /// Source format
        #[arg(long)]
        from: String,

        /// Source path
        path: String,

        /// Target namespace (default varies by format)
        #[arg(long, short)]
        namespace: Option<String>,

        /// Show what would be imported without actually importing
        #[arg(long)]
        dry_run: bool,
    },

    /// Server management
    Server {
        #[command(subcommand)]
        action: ServerAction,
    },

    /// Configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Show stats
    Stats,

    /// Encryption management
    Encrypt {
        #[command(subcommand)]
        action: EncryptAction,
    },

    /// Backup and restore
    Backup {
        #[command(subcommand)]
        action: BackupAction,
    },

    /// Export data
    Export {
        #[command(subcommand)]
        action: ExportAction,
    },

    /// Database maintenance
    Db {
        #[command(subcommand)]
        action: DbAction,
    },

    /// Manage secrets (API keys, tokens)
    Secret {
        #[command(subcommand)]
        action: SecretAction,
    },

    /// Consumer profile management (AI consumer identities)
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },

    /// Git credential helper (store/retrieve git credentials via Helix)
    #[command(name = "git-credential")]
    GitCredential {
        #[command(subcommand)]
        action: GitCredentialAction,
    },

    /// Sovereign keychain (sealed vault, credentials, delegations)
    Keychain {
        #[command(subcommand)]
        action: KeychainAction,
    },

    /// Start the MCP (Model Context Protocol) server on stdio
    Mcp {
        /// Access key for scoped MCP access (or set HELIX_MCP_ACCESS_KEY)
        #[arg(long)]
        access_key: Option<String>,
        /// Allow unscoped read-only access (or set HELIX_MCP_ALLOW_UNSCOPED=true)
        #[arg(long)]
        allow_unscoped: bool,
    },

    /// Interactive chat session via CLI
    Chat {
        /// Server URL to connect to
        #[arg(long, default_value = "http://127.0.0.1:9470")]
        server: String,
        /// Display name for this chat session
        #[arg(long, default_value = "cli-user")]
        name: String,
    },

    /// Interactive setup wizard for first-time configuration
    Setup {
        /// Skip interactive prompts, use defaults
        #[arg(long)]
        quick: bool,
    },
}

#[derive(Subcommand)]
enum DbAction {
    /// Run vacuum to reclaim space
    Vacuum,
    /// Check database integrity
    Check,
    /// Analyze for query optimization
    Analyze,
    /// Show database info and statistics
    Info,
    /// Rebuild indexes
    Reindex,
    /// Rebuild vector index (LanceDB) from SQLite nodes
    #[command(name = "rebuild-vectors")]
    RebuildVectors {
        /// Batch size for embedding
        #[arg(long, default_value = "64")]
        batch_size: usize,
        /// Dry run (counts nodes and shows target paths)
        #[arg(long)]
        dry_run: bool,
        /// Apply: swap rebuilt index into place
        #[arg(long)]
        apply: bool,
        /// Confirm destructive swap (required with --apply)
        #[arg(long)]
        confirm: bool,
    },
}

#[derive(Subcommand)]
enum EncryptAction {
    /// Initialize encryption for the vault
    Init {
        /// Use password from environment variable HELIX_ENCRYPTION_KEY
        #[arg(long)]
        from_env: bool,
    },
    /// Encrypt an existing unencrypted vault
    Migrate {
        /// Dry run (show what would be encrypted)
        #[arg(long)]
        dry_run: bool,
    },
    /// Decrypt vault (disable encryption)
    Decrypt {
        /// Confirm decryption
        #[arg(long)]
        confirm: bool,
    },
    /// Check encryption status
    Status,
}

#[derive(Subcommand)]
enum BackupAction {
    /// Create a backup
    Create {
        /// Output file path
        #[arg(long, short)]
        output: Option<String>,
    },
    /// Restore from a backup
    Restore {
        /// Backup file path
        input: String,

        /// Force overwrite existing data
        #[arg(long)]
        force: bool,
    },
    /// List available backups
    List,
}

#[derive(Subcommand)]
enum ExportAction {
    /// Export to JSON
    Json {
        /// Output file path
        #[arg(long, short)]
        output: Option<String>,

        /// Namespace filter
        #[arg(long, short)]
        namespace: Option<String>,
    },
    /// Export to Markdown files
    Markdown {
        /// Output directory
        #[arg(long, short)]
        output: Option<String>,

        /// Namespace filter
        #[arg(long, short)]
        namespace: Option<String>,
    },
    /// Export to CSV
    Csv {
        /// Output file path
        #[arg(long, short)]
        output: Option<String>,

        /// Namespace filter
        #[arg(long, short)]
        namespace: Option<String>,
    },
}

#[derive(Subcommand)]
enum GraphAction {
    /// Show graph neighbors for a node
    Show {
        /// Node ID
        id: String,

        /// Traversal depth
        #[arg(long, short, default_value = "2")]
        depth: usize,
    },
    /// Add a relationship
    Link {
        /// Source node ID
        from: String,
        /// Target node ID
        to: String,
        /// Relationship kind
        #[arg(long, short, default_value = "relates_to")]
        kind: String,
    },
}

#[derive(Subcommand)]
enum ServerAction {
    /// Start the server
    Start {
        /// REST port
        #[arg(long)]
        port: Option<u16>,

        /// gRPC port
        #[arg(long)]
        grpc_port: Option<u16>,

        /// Run in foreground
        #[arg(long)]
        foreground: bool,
    },
    /// Stop the server
    Stop,
    /// Check server status
    Status,
    /// Validate bind/auth safety before startup
    Preflight,
    /// Migrate legacy plaintext artifacts for sealed mode
    MigrateSealed {
        /// Read vault password from HELIX_VAULT_PASSWORD env var
        #[arg(long)]
        from_env: bool,
        /// Vault passphrase (or use --from-env)
        #[arg(long)]
        passphrase: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current config
    Show,
    /// Set a config value
    Set { key: String, value: String },
    /// Validate config
    Validate,
}

#[derive(Subcommand)]
enum SecretAction {
    /// Store a secret (prompts for value if not provided)
    Set {
        /// Secret key name (e.g., OPENAI_API_KEY)
        key: String,
        /// Secret value (omit to be prompted securely)
        value: Option<String>,
    },
    /// Retrieve a secret value
    Get {
        /// Secret key name
        key: String,
    },
    /// List stored secret names
    List,
    /// Delete a secret from all backends
    Delete {
        /// Secret key name
        key: String,
    },
    /// Show credential backend status
    Status,
    /// Initialize encrypted file backend (~/.helix/secrets.enc)
    #[command(name = "file-init")]
    FileInit,
    /// Unlock encrypted file backend (prompts for master password, then runs a secret subcommand)
    #[command(name = "file-unlock")]
    FileUnlock {
        /// Action to perform after unlocking: get, set, list, delete, status
        #[command(subcommand)]
        action: Box<FileUnlockAction>,
    },
    /// Set an access policy for a consumer on a secret
    #[command(name = "policy-set")]
    PolicySet {
        /// Secret key name
        #[arg(long)]
        key: String,
        /// Consumer name
        #[arg(long)]
        consumer: String,
        /// Allow access
        #[arg(long)]
        allow: bool,
        /// Max TTL in seconds (zero-standing privilege)
        #[arg(long)]
        ttl: Option<i64>,
    },
    /// List access policies
    #[command(name = "policy-list")]
    PolicyList {
        /// Filter by secret key
        #[arg(long)]
        secret: Option<String>,
        /// Filter by consumer name
        #[arg(long)]
        consumer: Option<String>,
    },
    /// Delete an access policy
    #[command(name = "policy-delete")]
    PolicyDelete {
        /// Policy ID
        id: String,
    },
}

#[derive(Subcommand)]
enum FileUnlockAction {
    /// Store a secret in the encrypted file
    Set {
        /// Secret key name (e.g., OPENAI_API_KEY)
        key: String,
        /// Secret value (omit to be prompted securely)
        value: Option<String>,
    },
    /// Retrieve a secret
    Get {
        /// Secret key name
        key: String,
    },
    /// List stored secret names
    List,
    /// Delete a secret
    Delete {
        /// Secret key name
        key: String,
    },
    /// Show credential backend status
    Status,
}

#[derive(Subcommand)]
enum ProfileAction {
    /// Create a consumer profile
    Create {
        /// Consumer name
        name: String,
        /// Description
        #[arg(long)]
        description: Option<String>,
    },
    /// List consumer profiles
    List,
    /// Revoke a consumer profile
    Revoke {
        /// Consumer ID
        id: String,
    },
    /// Check which consumer a token belongs to
    Whoami {
        /// Consumer bearer token
        token: String,
    },
}

#[derive(Subcommand)]
enum GitCredentialAction {
    /// Get credentials for a git host (reads protocol/host from stdin)
    Get,
    /// Install Helix as a global git credential helper
    Install,
    /// Remove Helix from git credential helpers
    Uninstall,
}

#[derive(Subcommand)]
enum KeychainAction {
    /// Initialize the sealed vault
    Init {
        /// Read password from HELIX_VAULT_PASSWORD env var
        #[arg(long)]
        from_env: bool,
        /// Store master password in macOS Keychain
        #[arg(long)]
        macos_bridge: bool,
    },
    /// Unseal (unlock) the vault
    Unseal {
        /// Read password from HELIX_VAULT_PASSWORD env var
        #[arg(long)]
        from_env: bool,
        /// Passphrase for vault unseal
        #[arg(long)]
        passphrase: Option<String>,
        /// Unseal using password stored in macOS Keychain
        #[arg(long)]
        from_macos_keychain: bool,
        /// Unseal using macOS Secure Enclave
        #[arg(long)]
        from_secure_enclave: bool,
        /// Auto-seal timeout in seconds (default: 900)
        #[arg(long, default_value = "900")]
        timeout: u64,
    },
    /// Migrate legacy plaintext artifacts into sealed storage
    #[command(name = "migrate-sealed")]
    MigrateSealed {
        /// Read password from HELIX_VAULT_PASSWORD env var
        #[arg(long)]
        from_env: bool,
        /// Passphrase for vault unseal
        #[arg(long)]
        passphrase: Option<String>,
        /// Unseal using password stored in macOS Keychain
        #[arg(long)]
        from_macos_keychain: bool,
        /// Unseal using macOS Secure Enclave
        #[arg(long)]
        from_secure_enclave: bool,
        /// Keep vault unsealed after migration completes
        #[arg(long)]
        keep_unsealed: bool,
    },
    /// Seal (lock) the vault
    Seal,
    /// Show vault status
    Status,
    /// Scan sealed storage for legacy plaintext artifacts
    Doctor,
    /// Rotate the master key
    Rotate {
        /// Grace period in hours for old key
        #[arg(long, default_value = "24")]
        grace_hours: u32,
    },
    /// Create a credential domain
    #[command(name = "domain-create")]
    DomainCreate {
        /// Domain name
        name: String,
        /// Description
        #[arg(long)]
        description: Option<String>,
    },
    /// List credential domains
    #[command(name = "domain-list")]
    DomainList,
    /// Revoke a credential domain
    #[command(name = "domain-revoke")]
    DomainRevoke {
        /// Domain ID
        id: String,
    },
    /// Store a credential (reads value from stdin)
    Store {
        /// Domain name
        #[arg(long)]
        domain: String,
        /// Credential name
        #[arg(long)]
        name: String,
        /// Credential kind (e.g., api_key, token, password)
        #[arg(long)]
        kind: String,
        /// Tags (comma-separated)
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        /// Expiration date (ISO 8601)
        #[arg(long)]
        expires_at: Option<String>,
    },
    /// Decrypt and display a credential
    Get {
        /// Credential ID
        id: String,
    },
    /// List credentials (metadata only, no decryption)
    List {
        /// Filter by domain name
        #[arg(long)]
        domain: Option<String>,
        /// Filter by state (active, expiring, expired, archived, destroyed)
        #[arg(long)]
        state: Option<String>,
        /// Max results
        #[arg(long, default_value = "50")]
        limit: u32,
    },
    /// Archive a credential
    Archive {
        /// Credential ID
        id: String,
    },
    /// Permanently destroy a credential (cryptographic shred)
    Destroy {
        /// Credential ID
        id: String,
        /// Confirm destruction
        #[arg(long)]
        confirm: bool,
    },
    /// Create a delegation
    #[command(name = "delegate-create")]
    DelegateCreate {
        /// Credential ID
        #[arg(long)]
        credential_id: String,
        /// Delegatee name
        #[arg(long)]
        delegatee: String,
        /// Expiration (ISO 8601)
        #[arg(long)]
        expires_at: String,
        /// Maximum delegation depth
        #[arg(long, default_value = "1")]
        max_depth: u32,
    },
    /// List delegations for a credential
    #[command(name = "delegate-list")]
    DelegateList {
        /// Credential ID
        #[arg(long)]
        credential_id: String,
    },
    /// Revoke a delegation
    #[command(name = "delegate-revoke")]
    DelegateRevoke {
        /// Delegation ID
        id: String,
    },
    /// Generate a zero-knowledge proof
    Prove {
        /// Credential ID
        #[arg(long)]
        credential_id: String,
        /// Challenge nonce
        #[arg(long)]
        nonce: String,
    },
    /// Verify audit chain integrity
    #[command(name = "audit-verify")]
    AuditVerify,
    /// List breach alerts
    Alerts {
        /// Max results
        #[arg(long, default_value = "20")]
        limit: u32,
    },
    /// Export an encrypted vault backup
    #[command(name = "vault-backup")]
    VaultBackup {
        /// Output file path
        #[arg(long, short)]
        output: String,
    },
    /// Restore vault from an encrypted backup
    #[command(name = "vault-restore")]
    VaultRestore {
        /// Backup file path
        input: String,
    },
    /// Enable Shamir secret sharing on the vault (splits master key into shares)
    #[command(name = "shamir-enable")]
    ShamirEnable {
        /// Minimum shares needed to reconstruct (M)
        #[arg(long)]
        threshold: u8,
        /// Total shares to create (N)
        #[arg(long)]
        total: u8,
    },
    /// Submit a Shamir share for vault reconstruction
    #[command(name = "shamir-submit")]
    ShamirSubmit {
        /// Base64-encoded share
        share: String,
    },
    /// Unseal the vault using collected Shamir shares
    #[command(name = "shamir-unseal")]
    ShamirUnseal,
    /// Re-split the master key into new Shamir shares (invalidates old shares)
    #[command(name = "shamir-rotate")]
    ShamirRotate,
    /// Show Shamir share collection status
    #[command(name = "shamir-status")]
    ShamirStatus,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Store {
            content,
            kind,
            title,
            source,
            tags,
            ns,
            importance,
        } => {
            commands::store::run(
                content,
                kind,
                title,
                source,
                tags,
                ns,
                importance,
                &cli.config,
            )
            .await
        }

        Commands::Recall {
            query,
            limit,
            strategy,
            min_score,
            ns,
        } => commands::recall::run(query, limit, strategy, min_score, ns, &cli.config).await,

        Commands::Search {
            query,
            limit,
            r#type,
        } => commands::search::run(query, limit, r#type, &cli.config).await,

        Commands::Graph { action } => match action {
            GraphAction::Show { id, depth } => commands::graph::show(id, depth, &cli.config).await,
            GraphAction::Link { from, to, kind } => {
                commands::graph::link(from, to, kind, &cli.config).await
            }
        },

        Commands::Import {
            from,
            path,
            namespace,
            dry_run,
        } => commands::import::run(from, path, namespace, dry_run, &cli.config).await,

        Commands::Server { action } => match action {
            ServerAction::Start {
                port,
                grpc_port,
                foreground,
            } => commands::server::start(port, grpc_port, foreground, &cli.config).await,
            ServerAction::Stop => commands::server::stop(&cli.config).await,
            ServerAction::Status => commands::server::status(&cli.config).await,
            ServerAction::Preflight => commands::server::preflight(&cli.config).await,
            ServerAction::MigrateSealed {
                from_env,
                passphrase,
            } => {
                commands::server::migrate_sealed(from_env, passphrase.as_deref(), &cli.config).await
            }
        },

        Commands::Config { action } => match action {
            ConfigAction::Show => commands::config::show(&cli.config).await,
            ConfigAction::Set { key, value } => {
                commands::config::set(&cli.config, &key, &value).await
            }
            ConfigAction::Validate => commands::config::validate(&cli.config).await,
        },

        Commands::Stats => commands::stats::run(&cli.config).await,

        Commands::Encrypt { action } => match action {
            EncryptAction::Init { from_env } => {
                commands::encrypt::init(from_env, &cli.config).await
            }
            EncryptAction::Migrate { dry_run } => {
                commands::encrypt::migrate(dry_run, &cli.config).await
            }
            EncryptAction::Decrypt { confirm } => {
                commands::encrypt::decrypt(confirm, &cli.config).await
            }
            EncryptAction::Status => commands::encrypt::status(&cli.config).await,
        },

        Commands::Backup { action } => match action {
            BackupAction::Create { output } => commands::backup::create(output, &cli.config).await,
            BackupAction::Restore { input, force } => {
                commands::backup::restore(input, force, &cli.config).await
            }
            BackupAction::List => commands::backup::list(&cli.config).await,
        },

        Commands::Export { action } => match action {
            ExportAction::Json { output, namespace } => {
                commands::export::json(output, namespace, &cli.config).await
            }
            ExportAction::Markdown { output, namespace } => {
                commands::export::markdown(output, namespace, &cli.config).await
            }
            ExportAction::Csv { output, namespace } => {
                commands::export::csv(output, namespace, &cli.config).await
            }
        },

        Commands::Db { action } => match action {
            DbAction::Vacuum => commands::db::vacuum(&cli.config).await,
            DbAction::Check => commands::db::check(&cli.config).await,
            DbAction::Analyze => commands::db::analyze(&cli.config).await,
            DbAction::Info => commands::db::info(&cli.config).await,
            DbAction::Reindex => commands::db::reindex(&cli.config).await,
            DbAction::RebuildVectors {
                batch_size,
                dry_run,
                apply,
                confirm,
            } => {
                commands::db::rebuild_vectors(&cli.config, batch_size, dry_run, apply, confirm)
                    .await
            }
        },

        Commands::Secret { action } => match action {
            SecretAction::Set { key, value } => commands::secret::set(&key, value.as_deref()).await,
            SecretAction::Get { key } => commands::secret::get(&key).await,
            SecretAction::List => commands::secret::list().await,
            SecretAction::Delete { key } => commands::secret::delete(&key).await,
            SecretAction::Status => commands::secret::status().await,
            SecretAction::FileInit => commands::secret::file_init().await,
            SecretAction::FileUnlock { action } => commands::secret::file_unlock(*action).await,
            SecretAction::PolicySet {
                key,
                consumer,
                allow,
                ttl,
            } => commands::secret::policy_set(&key, &consumer, allow, ttl).await,
            SecretAction::PolicyList { secret, consumer } => {
                commands::secret::policy_list(secret.as_deref(), consumer.as_deref()).await
            }
            SecretAction::PolicyDelete { id } => commands::secret::policy_delete(&id).await,
        },

        Commands::Profile { action } => match action {
            ProfileAction::Create { name, description } => {
                commands::profile::create(&name, description.as_deref()).await
            }
            ProfileAction::List => commands::profile::list().await,
            ProfileAction::Revoke { id } => commands::profile::revoke(&id).await,
            ProfileAction::Whoami { token } => commands::profile::whoami(&token).await,
        },

        Commands::GitCredential { action } => match action {
            GitCredentialAction::Get => commands::git_credential::get().await,
            GitCredentialAction::Install => commands::git_credential::install().await,
            GitCredentialAction::Uninstall => commands::git_credential::uninstall().await,
        },

        Commands::Keychain { action } => match action {
            KeychainAction::Init {
                from_env,
                macos_bridge,
            } => commands::keychain::init_vault(from_env, macos_bridge, &cli.config).await,
            KeychainAction::Unseal {
                from_env,
                passphrase,
                from_macos_keychain,
                from_secure_enclave,
                timeout,
            } => {
                commands::keychain::unseal(
                    from_env,
                    passphrase.as_deref(),
                    from_macos_keychain,
                    from_secure_enclave,
                    timeout,
                    &cli.config,
                )
                .await
            }
            KeychainAction::MigrateSealed {
                from_env,
                passphrase,
                from_macos_keychain,
                from_secure_enclave,
                keep_unsealed,
            } => {
                commands::keychain::migrate_sealed(
                    from_env,
                    passphrase.as_deref(),
                    from_macos_keychain,
                    from_secure_enclave,
                    keep_unsealed,
                    &cli.config,
                )
                .await
            }
            KeychainAction::Seal => commands::keychain::seal(&cli.config).await,
            KeychainAction::Status => commands::keychain::status(&cli.config).await,
            KeychainAction::Doctor => commands::keychain::doctor(&cli.config).await,
            KeychainAction::Rotate { grace_hours } => {
                commands::keychain::rotate_key(grace_hours, &cli.config).await
            }
            KeychainAction::DomainCreate { name, description } => {
                commands::keychain::domain_create(&name, description.as_deref(), &cli.config).await
            }
            KeychainAction::DomainList => commands::keychain::domain_list(&cli.config).await,
            KeychainAction::DomainRevoke { id } => {
                commands::keychain::domain_revoke(&id, &cli.config).await
            }
            KeychainAction::Store {
                domain,
                name,
                kind,
                tags,
                expires_at,
            } => {
                commands::keychain::store_credential(
                    &domain,
                    &name,
                    &kind,
                    &tags,
                    expires_at.as_deref(),
                    &cli.config,
                )
                .await
            }
            KeychainAction::Get { id } => {
                commands::keychain::get_credential(&id, &cli.config).await
            }
            KeychainAction::List {
                domain,
                state,
                limit,
            } => {
                commands::keychain::list_credentials(
                    domain.as_deref(),
                    state.as_deref(),
                    limit,
                    &cli.config,
                )
                .await
            }
            KeychainAction::Archive { id } => {
                commands::keychain::archive_credential(&id, &cli.config).await
            }
            KeychainAction::Destroy { id, confirm } => {
                commands::keychain::destroy_credential(&id, confirm, &cli.config).await
            }
            KeychainAction::DelegateCreate {
                credential_id,
                delegatee,
                expires_at,
                max_depth,
            } => {
                commands::keychain::delegate_create(
                    &credential_id,
                    &delegatee,
                    &expires_at,
                    max_depth,
                    &cli.config,
                )
                .await
            }
            KeychainAction::DelegateList { credential_id } => {
                commands::keychain::delegate_list(&credential_id, &cli.config).await
            }
            KeychainAction::DelegateRevoke { id } => {
                commands::keychain::delegate_revoke(&id, &cli.config).await
            }
            KeychainAction::Prove {
                credential_id,
                nonce,
            } => commands::keychain::prove(&credential_id, &nonce, &cli.config).await,
            KeychainAction::AuditVerify => commands::keychain::audit_verify(&cli.config).await,
            KeychainAction::Alerts { limit } => {
                commands::keychain::alerts(limit, &cli.config).await
            }
            KeychainAction::VaultBackup { output } => {
                commands::keychain::vault_backup(&output, &cli.config).await
            }
            KeychainAction::VaultRestore { input } => {
                commands::keychain::vault_restore(&input, &cli.config).await
            }
            KeychainAction::ShamirEnable { threshold, total } => {
                commands::keychain::shamir_enable(threshold, total, &cli.config).await
            }
            KeychainAction::ShamirSubmit { share } => {
                commands::keychain::shamir_submit(&share, &cli.config).await
            }
            KeychainAction::ShamirUnseal => commands::keychain::shamir_unseal(&cli.config).await,
            KeychainAction::ShamirRotate => commands::keychain::shamir_rotate(&cli.config).await,
            KeychainAction::ShamirStatus => commands::keychain::shamir_status(&cli.config).await,
        },

        Commands::Mcp {
            access_key,
            allow_unscoped,
        } => commands::mcp::run(&cli.config, access_key, allow_unscoped).await,

        Commands::Chat { server, name } => commands::chat::run_chat(&server, &name).await,

        Commands::Setup { quick } => commands::setup::run_setup(quick, &cli.config).await,
    }
}
