use anyhow::Result;

pub async fn show(config_path: &str) -> Result<()> {
    let path = super::shellexpand(config_path);

    if std::path::Path::new(&path).exists() {
        let content = std::fs::read_to_string(&path)?;
        println!("config file: {path}\n");
        println!("{content}");
    } else {
        println!("no config file at {path}");
    }

    let runtime = super::load_runtime_config(config_path)?;
    println!("resolved runtime config:\n");
    println!("[server]");
    println!("  bind_host = {}", runtime.server.bind_host);
    println!("  rest_port = {}", runtime.server.rest_port);
    println!("  grpc_port = {}", runtime.server.grpc_port);
    if let Some(socket_path) = runtime.server.socket_path {
        println!("  socket_path = {socket_path}");
    }
    if runtime.server.cors_allowed_origins.is_empty() {
        println!("  cors_allowed_origins = []");
    } else {
        println!(
            "  cors_allowed_origins = {}",
            runtime.server.cors_allowed_origins.join(", ")
        );
    }

    println!("\n[storage]");
    println!("  data_dir = {}", runtime.engine.data_dir);

    println!("\n[profile]");
    println!("  display_name = {}", runtime.engine.profile.display_name);
    println!(
        "  primary_email = {}",
        runtime
            .engine
            .profile
            .primary_email
            .as_deref()
            .unwrap_or("<unset>")
    );
    println!("  timezone = {}", runtime.engine.profile.timezone);
    println!(
        "  signature = {}",
        runtime
            .engine
            .profile
            .signature
            .as_deref()
            .unwrap_or("<unset>")
    );

    println!("\n[embedding]");
    println!("  provider = {}", runtime.engine.embedding.provider);
    println!("  model = {}", runtime.engine.embedding.model);
    println!("  dimensions = {}", runtime.engine.embedding.dimensions);

    println!("\n[search]");
    println!("  default_limit = {}", runtime.engine.search.default_limit);
    println!(
        "  default_strategy = {}",
        runtime.engine.search.default_strategy
    );
    println!("  min_score = {}", runtime.engine.search.min_score);
    println!("  vector_weight = {}", runtime.engine.search.vector_weight);
    println!(
        "  fulltext_weight = {}",
        runtime.engine.search.fulltext_weight
    );
    println!("  rrf_k = {}", runtime.engine.search.rrf_k);

    println!("\n[graph]");
    println!(
        "  default_traversal_depth = {}",
        runtime.engine.graph.default_traversal_depth
    );
    println!(
        "  graph_boost_factor = {}",
        runtime.engine.graph.graph_boost_factor
    );

    println!("\n[ai]");
    println!(
        "  auto_tagging_enabled = {}",
        runtime.engine.ai.auto_tagging_enabled
    );
    println!(
        "  auto_tagging_max_generated_tags = {}",
        runtime.engine.ai.auto_tagging_max_generated_tags
    );
    println!(
        "  auto_tagging_max_total_tags = {}",
        runtime.engine.ai.auto_tagging_max_total_tags
    );
    println!(
        "  auto_tagging_similarity_seed_limit = {}",
        runtime.engine.ai.auto_tagging_similarity_seed_limit
    );
    println!(
        "  auto_tagging_min_token_length = {}",
        runtime.engine.ai.auto_tagging_min_token_length
    );

    println!("\n[linking]");
    println!(
        "  auto_backlinks_enabled = {}",
        runtime.engine.linking.auto_backlinks_enabled
    );
    println!(
        "  auto_backlinks_scan_limit = {}",
        runtime.engine.linking.auto_backlinks_scan_limit
    );
    println!(
        "  auto_backlinks_max_targets = {}",
        runtime.engine.linking.auto_backlinks_max_targets
    );

    println!("\n[daily_notes]");
    println!("  enabled = {}", runtime.engine.daily_notes.enabled);
    println!(
        "  midnight_scheduler_enabled = {}",
        runtime.engine.daily_notes.midnight_scheduler_enabled
    );
    println!("  namespace = {}", runtime.engine.daily_notes.namespace);
    println!(
        "  title_template = {}",
        runtime.engine.daily_notes.title_template
    );
    println!(
        "  default_importance = {}",
        runtime.engine.daily_notes.default_importance
    );

    println!("\n[recurrence]");
    println!("  enabled = {}", runtime.engine.recurrence.enabled);
    println!(
        "  scheduler_interval_secs = {}",
        runtime.engine.recurrence.scheduler_interval_secs
    );
    println!(
        "  max_instances_per_template = {}",
        runtime.engine.recurrence.max_instances_per_template
    );

    println!("\n[encryption]");
    println!("  enabled = {}", runtime.engine.encryption.enabled);
    println!(
        "  argon2_memory_kib = {}",
        runtime.engine.encryption.argon2_memory_kib
    );
    println!(
        "  argon2_iterations = {}",
        runtime.engine.encryption.argon2_iterations
    );
    println!(
        "  argon2_parallelism = {}",
        runtime.engine.encryption.argon2_parallelism
    );

    println!("\n[llm]");
    println!("  enabled = {}", runtime.engine.llm.enabled);
    println!("  base_url = {}", runtime.engine.llm.base_url);
    println!("  model = {}", runtime.engine.llm.model);
    println!("  max_tokens = {}", runtime.engine.llm.max_tokens);
    println!("  temperature = {}", runtime.engine.llm.temperature);
    println!("  timeout_secs = {}", runtime.engine.llm.timeout_secs);

    println!("\n[email]");
    println!("  enabled = {}", runtime.engine.email.enabled);
    println!("  namespace = {}", runtime.engine.email.namespace);
    println!(
        "  poll_interval_secs = {}",
        runtime.engine.email.poll_interval_secs
    );
    println!("  max_fetch = {}", runtime.engine.email.max_fetch);
    println!(
        "  max_attachment_bytes = {}",
        runtime.engine.email.max_attachment_bytes
    );
    println!("  mark_seen = {}", runtime.engine.email.mark_seen);
    println!(
        "  imap_host = {}",
        runtime.engine.email.imap_host.as_deref().unwrap_or("<unset>")
    );
    println!("  imap_port = {}", runtime.engine.email.imap_port);
    println!(
        "  imap_username = {}",
        runtime
            .engine
            .email
            .imap_username
            .as_deref()
            .unwrap_or("<unset>")
    );
    println!("  imap_folder = {}", runtime.engine.email.imap_folder);
    println!("  imap_starttls = {}", runtime.engine.email.imap_starttls);
    println!(
        "  smtp_host = {}",
        runtime.engine.email.smtp_host.as_deref().unwrap_or("<unset>")
    );
    println!("  smtp_port = {}", runtime.engine.email.smtp_port);
    println!(
        "  smtp_username = {}",
        runtime
            .engine
            .email
            .smtp_username
            .as_deref()
            .unwrap_or("<unset>")
    );
    println!(
        "  smtp_from = {}",
        runtime.engine.email.smtp_from.as_deref().unwrap_or("<unset>")
    );
    println!("  smtp_starttls = {}", runtime.engine.email.smtp_starttls);

    println!("\n[watcher]");
    println!("  enabled = {}", runtime.engine.watcher.enabled);
    println!("  interval_secs = {}", runtime.engine.watcher.interval_secs);
    println!(
        "  lookback_hours = {}",
        runtime.engine.watcher.lookback_hours
    );
    println!(
        "  max_nodes_per_cycle = {}",
        runtime.engine.watcher.max_nodes_per_cycle
    );

    Ok(())
}

pub async fn set(config_path: &str, key: &str, value: &str) -> Result<()> {
    let path = super::shellexpand(config_path);

    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut root: toml::Value = if std::path::Path::new(&path).exists() {
        let content = std::fs::read_to_string(&path)?;
        toml::from_str(&content)?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };

    let parsed_value = parse_value(value);
    set_nested_value(&mut root, key, parsed_value)?;

    let content = toml::to_string_pretty(&root)?;
    std::fs::write(&path, content)?;

    println!("set {key} = {value} in {path}");
    Ok(())
}

pub async fn validate(config_path: &str) -> Result<()> {
    let path = super::shellexpand(config_path);

    if !std::path::Path::new(&path).exists() {
        println!("no config file at {path} — using defaults (valid)");
        return Ok(());
    }

    match super::load_runtime_config(config_path) {
        Ok(runtime) => {
            println!("config is valid:");
            println!("  server.bind_host: {}", runtime.server.bind_host);
            println!("  server.rest_port: {}", runtime.server.rest_port);
            println!("  server.grpc_port: {}", runtime.server.grpc_port);
            println!("  data_dir: {}", runtime.engine.data_dir);
            println!("  embedding.model: {}", runtime.engine.embedding.model);

            if std::path::Path::new(&runtime.engine.data_dir).exists() {
                println!("  data_dir exists: yes");
            } else {
                println!("  data_dir exists: no (will be created on first use)");
            }

            let creds = hx_core::credentials::CredentialStore::new("helix");
            match creds.get("OPENAI_API_KEY") {
                Ok(Some(sv)) => println!("  OPENAI_API_KEY: set (via {})", sv.source()),
                _ => println!("  OPENAI_API_KEY: not set (vector search will be disabled)"),
            }
            match creds.get("HELIX_EMAIL_IMAP_PASSWORD") {
                Ok(Some(sv)) => println!("  HELIX_EMAIL_IMAP_PASSWORD: set (via {})", sv.source()),
                _ => println!("  HELIX_EMAIL_IMAP_PASSWORD: not set"),
            }
            match creds.get("HELIX_EMAIL_SMTP_PASSWORD") {
                Ok(Some(sv)) => println!("  HELIX_EMAIL_SMTP_PASSWORD: set (via {})", sv.source()),
                _ => println!("  HELIX_EMAIL_SMTP_PASSWORD: not set"),
            }
        }
        Err(e) => {
            println!("config validation failed: {e}");
            return Err(e);
        }
    }

    Ok(())
}

fn parse_value(value: &str) -> toml::Value {
    if let Ok(boolean) = value.parse::<bool>() {
        return toml::Value::Boolean(boolean);
    }
    if let Ok(integer) = value.parse::<i64>() {
        return toml::Value::Integer(integer);
    }
    if let Ok(float) = value.parse::<f64>() {
        return toml::Value::Float(float);
    }
    if value.contains(',') {
        let items: Vec<toml::Value> = value
            .split(',')
            .map(|item| toml::Value::String(item.trim().to_string()))
            .filter(|item| matches!(item, toml::Value::String(s) if !s.is_empty()))
            .collect();
        if !items.is_empty() {
            return toml::Value::Array(items);
        }
    }

    toml::Value::String(value.to_string())
}

fn set_nested_value(root: &mut toml::Value, key: &str, value: toml::Value) -> Result<()> {
    let mut current = root;
    let mut parts = key.split('.').peekable();

    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            let table = current
                .as_table_mut()
                .ok_or_else(|| anyhow::anyhow!("key path points to non-table: {key}"))?;
            table.insert(part.to_string(), value);
            return Ok(());
        }

        let table = current
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("key path points to non-table: {key}"))?;

        current = table
            .entry(part.to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }

    Ok(())
}
