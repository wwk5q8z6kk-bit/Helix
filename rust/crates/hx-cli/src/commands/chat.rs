//! Interactive chat REPL connecting to the Helix server.
//!
//! `hx chat [--server URL] [--name NAME]`

use anyhow::Result;
use std::io::{self, BufRead, Write};

/// Run the interactive chat REPL.
pub async fn run_chat(server_url: &str, name: &str) -> Result<()> {
    println!("Helix Chat -- connected to {server_url}");
    println!("Type your message and press Enter. Type /quit to exit.\n");

    let client = reqwest::Client::new();
    let stdin = io::stdin();

    loop {
        print!("{name}> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/quit" || trimmed == "/exit" {
            println!("Goodbye!");
            break;
        }

        match send_message(&client, server_url, trimmed).await {
            Ok(response) => println!("helix> {response}\n"),
            Err(e) => eprintln!("Error: {e}\n"),
        }
    }

    Ok(())
}

async fn send_message(client: &reqwest::Client, server_url: &str, message: &str) -> Result<String> {
    let url = format!("{server_url}/api/generate");
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": message,
            "max_tokens": 2048,
            "temperature": 0.7,
        }))
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("Server returned {}", resp.status());
    }

    let body: serde_json::Value = resp.json().await?;
    Ok(body["text"].as_str().unwrap_or("(no response)").to_string())
}
