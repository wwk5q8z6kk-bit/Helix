//! Interactive setup wizard for Helix platform configuration.
//!
//! `hx setup` -- 7-step guided wizard
//! `hx setup --quick` -- accept all defaults

use anyhow::Result;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Run the setup wizard.
pub async fn run_setup(quick: bool, _config_path: &str) -> Result<()> {
    let helix_home = resolve_helix_home();

    if quick {
        return run_quick_setup(&helix_home);
    }

    println!("=== Helix Setup Wizard ===\n");

    // Step 1: Init workspace
    step_init_workspace(&helix_home)?;

    // Step 2: AI providers (prompt for API keys)
    step_ai_providers(&helix_home)?;

    // Step 3: Channels (configure adapters)
    step_channels(&helix_home)?;

    // Step 4: Tunnel (optional)
    step_tunnel(&helix_home)?;

    // Step 5: Tool mode (sovereign/composio/hybrid)
    step_tool_mode(&helix_home)?;

    // Step 6: Personalization
    step_personalization(&helix_home)?;

    // Step 7: Scaffold files
    step_scaffold(&helix_home)?;

    println!("\nSetup complete! Run `hx server start` to begin.");
    Ok(())
}

fn resolve_helix_home() -> PathBuf {
    std::env::var("HELIX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".helix")
        })
}

fn step_init_workspace(home: &Path) -> Result<()> {
    println!("Step 1/7: Initializing workspace...");
    std::fs::create_dir_all(home.join("data"))?;
    std::fs::create_dir_all(home.join("skills"))?;
    std::fs::create_dir_all(home.join("backups"))?;
    println!("  Created {}", home.display());
    Ok(())
}

fn step_ai_providers(home: &Path) -> Result<()> {
    println!("\nStep 2/7: AI Provider Configuration");
    println!("  Enter API keys (press Enter to skip):\n");

    let providers = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT)"),
        ("GOOGLE_API_KEY", "Google (Gemini)"),
        ("XAI_API_KEY", "xAI (Grok)"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("MISTRAL_API_KEY", "Mistral"),
        ("TOGETHER_API_KEY", "Together AI"),
    ];

    for (env_var, name) in &providers {
        let key = prompt_line(&format!("  {name} ({env_var}): "))?;
        if !key.is_empty() {
            append_env_var(home, env_var, &key)?;
            println!("    Stored");
        }
    }
    Ok(())
}

fn step_channels(home: &Path) -> Result<()> {
    println!("\nStep 3/7: Channel Configuration");
    let choice = prompt_choice("  Enable Discord adapter?", &["yes", "no"], 1)?;
    if choice == 0 {
        let token = prompt_line("  Discord bot token: ")?;
        if !token.is_empty() {
            append_env_var(home, "DISCORD_BOT_TOKEN", &token)?;
            println!("    Stored");
        }
    }

    let choice = prompt_choice("  Enable Telegram adapter?", &["yes", "no"], 1)?;
    if choice == 0 {
        let token = prompt_line("  Telegram bot token: ")?;
        if !token.is_empty() {
            append_env_var(home, "TELEGRAM_BOT_TOKEN", &token)?;
            println!("    Stored");
        }
    }

    Ok(())
}

fn step_tunnel(home: &Path) -> Result<()> {
    println!("\nStep 4/7: Tunnel Configuration");
    let choice = prompt_choice(
        "  Enable tunnel (Cloudflare/ngrok)?",
        &["skip", "cloudflare", "ngrok"],
        0,
    )?;
    match choice {
        1 => {
            let token = prompt_line("  Cloudflare tunnel token: ")?;
            if !token.is_empty() {
                append_env_var(home, "CLOUDFLARE_TUNNEL_TOKEN", &token)?;
                println!("    Stored");
            }
        }
        2 => {
            let token = prompt_line("  ngrok auth token: ")?;
            if !token.is_empty() {
                append_env_var(home, "NGROK_AUTH_TOKEN", &token)?;
                println!("    Stored");
            }
        }
        _ => println!("  Skipping tunnel configuration."),
    }
    let _ = home; // suppress unused warning in skip path
    Ok(())
}

fn step_tool_mode(home: &Path) -> Result<()> {
    println!("\nStep 5/7: Tool Mode");
    let choice = prompt_choice(
        "  Select tool execution mode",
        &["sovereign (local-only)", "composio (cloud)", "hybrid"],
        0,
    )?;
    let mode = match choice {
        1 => "composio",
        2 => "hybrid",
        _ => "sovereign",
    };
    append_env_var(home, "HELIX_TOOL_MODE", mode)?;
    println!("    Set tool mode to: {mode}");
    Ok(())
}

fn step_personalization(home: &Path) -> Result<()> {
    println!("\nStep 6/7: Personalization");
    let name = prompt_line("  Display name (default: Helix): ")?;
    let display_name = if name.is_empty() {
        "Helix".to_string()
    } else {
        name
    };
    let timezone = prompt_line("  Timezone (default: UTC): ")?;
    let tz = if timezone.is_empty() {
        "UTC".to_string()
    } else {
        timezone
    };
    append_env_var(home, "HELIX_PROFILE_DISPLAY_NAME", &display_name)?;
    append_env_var(home, "HELIX_PROFILE_TIMEZONE", &tz)?;
    println!("    Set display_name={display_name}, timezone={tz}");
    Ok(())
}

fn step_scaffold(home: &Path) -> Result<()> {
    println!("\nStep 7/7: Scaffolding files...");

    let config_path = home.join("config.toml");
    if !config_path.exists() {
        std::fs::write(&config_path, default_config_toml())?;
        println!("  Created config.toml");
    } else {
        println!("  config.toml already exists, skipping");
    }

    let heartbeat_path = home.join("HEARTBEAT.md");
    if !heartbeat_path.exists() {
        std::fs::write(&heartbeat_path, default_heartbeat_md())?;
        println!("  Created HEARTBEAT.md");
    } else {
        println!("  HEARTBEAT.md already exists, skipping");
    }

    Ok(())
}

fn run_quick_setup(home: &Path) -> Result<()> {
    println!("Running quick setup with defaults...");
    step_init_workspace(home)?;
    step_scaffold(home)?;
    println!("Quick setup complete!");
    Ok(())
}

fn default_config_toml() -> &'static str {
    r#"# Helix configuration
# See https://github.com/anthropics/helix for documentation

[server]
bind_host = "127.0.0.1"
rest_port = 9470
grpc_port = 50051

[storage]
data_dir = "~/.helix/data"

[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimensions = 1536

[search]
default_limit = 10
default_strategy = "hybrid"

[graph]
default_traversal_depth = 2
"#
}

fn default_heartbeat_md() -> &'static str {
    r#"# Helix Heartbeat

This file tracks the health of your Helix instance.
It is automatically updated by the Helix server.

## Status
- **State**: initialized
- **Last check**: never
"#
}

fn append_env_var(home: &Path, key: &str, value: &str) -> Result<()> {
    let env_path = home.join(".env");
    if let Some(parent) = env_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Read existing content to check for duplicates
    let existing = std::fs::read_to_string(&env_path).unwrap_or_default();
    let prefix = format!("{key}=");

    // If key already exists, replace the line; otherwise append
    if existing.lines().any(|line| line.starts_with(&prefix)) {
        let updated: Vec<String> = existing
            .lines()
            .map(|line| {
                if line.starts_with(&prefix) {
                    format!("{key}={value}")
                } else {
                    line.to_string()
                }
            })
            .collect();
        std::fs::write(&env_path, updated.join("\n") + "\n")?;
    } else {
        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&env_path)?;
        writeln!(file, "{key}={value}")?;
    }

    Ok(())
}

fn prompt_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn prompt_choice(prompt: &str, options: &[&str], default: usize) -> Result<usize> {
    println!("{prompt}");
    for (i, option) in options.iter().enumerate() {
        let marker = if i == default { " (default)" } else { "" };
        println!("    [{i}] {option}{marker}");
    }
    let input = prompt_line("  Choice: ")?;
    if input.is_empty() {
        return Ok(default);
    }
    match input.parse::<usize>() {
        Ok(n) if n < options.len() => Ok(n),
        _ => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_home() -> PathBuf {
        let id = uuid::Uuid::now_v7();
        std::env::temp_dir().join(format!("helix-setup-test-{id}"))
    }

    #[test]
    fn init_creates_dirs() {
        let home = temp_home();
        step_init_workspace(&home).unwrap();

        assert!(home.join("data").is_dir());
        assert!(home.join("skills").is_dir());
        assert!(home.join("backups").is_dir());

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn scaffold_generates_files() {
        let home = temp_home();
        fs::create_dir_all(&home).unwrap();

        step_scaffold(&home).unwrap();

        let config_path = home.join("config.toml");
        let heartbeat_path = home.join("HEARTBEAT.md");

        assert!(config_path.exists());
        assert!(heartbeat_path.exists());

        let config_content = fs::read_to_string(&config_path).unwrap();
        assert!(config_content.contains("[server]"));
        assert!(config_content.contains("rest_port"));

        let heartbeat_content = fs::read_to_string(&heartbeat_path).unwrap();
        assert!(heartbeat_content.contains("Heartbeat"));

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn quick_mode_defaults() {
        let home = temp_home();

        run_quick_setup(&home).unwrap();

        assert!(home.join("data").is_dir());
        assert!(home.join("skills").is_dir());
        assert!(home.join("backups").is_dir());
        assert!(home.join("config.toml").exists());
        assert!(home.join("HEARTBEAT.md").exists());

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn rerun_does_not_overwrite() {
        let home = temp_home();
        fs::create_dir_all(&home).unwrap();

        step_scaffold(&home).unwrap();

        // Write custom content to config.toml
        let config_path = home.join("config.toml");
        fs::write(&config_path, "# custom config\n").unwrap();

        // Run scaffold again — should not overwrite
        step_scaffold(&home).unwrap();

        let content = fs::read_to_string(&config_path).unwrap();
        assert_eq!(content, "# custom config\n");

        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn env_var_append() {
        let home = temp_home();
        fs::create_dir_all(&home).unwrap();

        append_env_var(&home, "TEST_KEY", "value1").unwrap();
        let content = fs::read_to_string(home.join(".env")).unwrap();
        assert!(content.contains("TEST_KEY=value1"));

        // Append a different key
        append_env_var(&home, "OTHER_KEY", "value2").unwrap();
        let content = fs::read_to_string(home.join(".env")).unwrap();
        assert!(content.contains("TEST_KEY=value1"));
        assert!(content.contains("OTHER_KEY=value2"));

        // Replace existing key
        append_env_var(&home, "TEST_KEY", "updated").unwrap();
        let content = fs::read_to_string(home.join(".env")).unwrap();
        assert!(content.contains("TEST_KEY=updated"));
        assert!(!content.contains("TEST_KEY=value1"));

        let _ = fs::remove_dir_all(&home);
    }
}
