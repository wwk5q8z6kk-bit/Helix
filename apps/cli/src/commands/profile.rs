use anyhow::Result;
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "http://127.0.0.1:9470";

#[derive(Deserialize)]
struct CreateConsumerResponse {
    id: String,
    name: String,
    token: String,
}

#[derive(Deserialize)]
struct ConsumerSummary {
    id: String,
    name: String,
    description: Option<String>,
    created_at: String,
    last_used_at: Option<String>,
    revoked_at: Option<String>,
}

#[derive(Serialize)]
struct CreateConsumerRequest {
    name: String,
    description: Option<String>,
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to create HTTP client")
}

/// Create a consumer profile.
pub async fn create(name: &str, description: Option<&str>) -> Result<()> {
    let resp = client()
        .post(format!("{BASE_URL}/api/v1/consumers"))
        .json(&CreateConsumerRequest {
            name: name.to_string(),
            description: description.map(|d| d.to_string()),
        })
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("server returned {status}: {body}");
    }

    let result: CreateConsumerResponse = resp.json().await?;

    println!("Consumer created:");
    println!("  ID:    {}", result.id);
    println!("  Name:  {}", result.name);
    println!();
    println!("  Token: {}", result.token);
    println!();
    println!("IMPORTANT: Save this token now. It cannot be retrieved again.");

    Ok(())
}

/// List all consumer profiles.
pub async fn list() -> Result<()> {
    let resp = client()
        .get(format!("{BASE_URL}/api/v1/consumers"))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("server returned {status}: {body}");
    }

    let consumers: Vec<ConsumerSummary> = resp.json().await?;

    if consumers.is_empty() {
        println!("no consumer profiles found");
        return Ok(());
    }

    println!(
        "{:<38} {:<20} {:<10} {}",
        "ID", "Name", "Status", "Created"
    );
    println!("{}", "-".repeat(90));

    for c in &consumers {
        let status = if c.revoked_at.is_some() {
            "revoked"
        } else {
            "active"
        };
        println!("{:<38} {:<20} {:<10} {}", c.id, c.name, status, c.created_at);
    }

    Ok(())
}

/// Revoke a consumer profile by ID.
pub async fn revoke(id: &str) -> Result<()> {
    let resp = client()
        .delete(format!("{BASE_URL}/api/v1/consumers/{id}"))
        .send()
        .await?;

    if resp.status().as_u16() == 204 {
        println!("consumer {id} revoked");
    } else if resp.status().as_u16() == 404 {
        anyhow::bail!("consumer {id} not found");
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("server returned {status}: {body}");
    }

    Ok(())
}

/// Show which consumer a token corresponds to.
pub async fn whoami(token: &str) -> Result<()> {
    let resp = client()
        .get(format!("{BASE_URL}/api/v1/consumers/whoami"))
        .bearer_auth(token)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("server returned {status}: {body}");
    }

    let consumer: ConsumerSummary = resp.json().await?;

    println!("Consumer Profile:");
    println!("  ID:          {}", consumer.id);
    println!("  Name:        {}", consumer.name);
    if let Some(desc) = &consumer.description {
        println!("  Description: {desc}");
    }
    println!("  Created:     {}", consumer.created_at);
    if let Some(last_used) = &consumer.last_used_at {
        println!("  Last Used:   {last_used}");
    }
    if let Some(revoked) = &consumer.revoked_at {
        println!("  Revoked At:  {revoked}");
    }

    Ok(())
}
