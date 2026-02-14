use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::LlmConfig;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionParams {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for CompletionParams {
    fn default() -> Self {
        Self {
            model: None,
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("LLM not configured or disabled")]
    NotConfigured,
    #[error("LLM request failed: {0}")]
    RequestFailed(String),
    #[error("LLM response parse error: {0}")]
    ParseError(String),
    #[error("LLM request timed out")]
    Timeout,
    #[error("LLM server unreachable: {0}")]
    Unreachable(String),
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Provider-agnostic LLM interface. Works with any OpenAI-compatible API.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a chat completion request.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        params: &CompletionParams,
    ) -> Result<String, LlmError>;

    /// Provider name for logging/diagnostics.
    fn name(&self) -> &str;

    /// Check if the provider is reachable (non-blocking best-effort).
    async fn is_available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// OpenAI-Compatible Provider
// ---------------------------------------------------------------------------

/// Works with any OpenAI-compatible chat completions API:
/// Ollama, vLLM, LMStudio, llama.cpp, OpenAI, Together, Groq, Fireworks, etc.
pub struct OpenAiCompatibleLlm {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    max_tokens: u32,
    temperature: f32,
}

impl OpenAiCompatibleLlm {
    pub fn from_config(config: &LlmConfig, api_key: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .unwrap_or_default();

        Self {
            client,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key,
            model: config.model.clone(),
            max_tokens: config.max_tokens,
            temperature: config.temperature,
        }
    }
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiCompatibleLlm {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        params: &CompletionParams,
    ) -> Result<String, LlmError> {
        let model = params.model.clone().unwrap_or_else(|| self.model.clone());
        let max_tokens = params.max_tokens.unwrap_or(self.max_tokens);
        let temperature = params.temperature.unwrap_or(self.temperature);

        let url = format!("{}/chat/completions", self.base_url);

        let request = ChatCompletionRequest {
            model,
            messages: messages.to_vec(),
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
        };

        let mut req_builder = self.client.post(&url).json(&request);
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let response = req_builder.send().await.map_err(|e| {
            if e.is_timeout() {
                LlmError::Timeout
            } else if e.is_connect() {
                LlmError::Unreachable(e.to_string())
            } else {
                LlmError::RequestFailed(e.to_string())
            }
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::RequestFailed(format!("HTTP {status}: {body}")));
        }

        let chat_resp: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::ParseError("no content in response".to_string()))
    }

    fn name(&self) -> &str {
        "openai-compatible"
    }

    async fn is_available(&self) -> bool {
        let url = format!("{}/models", self.base_url);
        let mut req = self.client.get(&url);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        match req.timeout(Duration::from_secs(3)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback Provider
// ---------------------------------------------------------------------------

/// Tries each provider in order until one succeeds.
pub struct FallbackLlmProvider {
    providers: Vec<Arc<dyn LlmProvider>>,
    name: String,
}

impl FallbackLlmProvider {
    pub fn new(providers: Vec<Arc<dyn LlmProvider>>) -> Self {
        let name = providers
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
            .join(" -> ");
        Self { providers, name }
    }
}

#[async_trait::async_trait]
impl LlmProvider for FallbackLlmProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        params: &CompletionParams,
    ) -> Result<String, LlmError> {
        let mut last_err = LlmError::NotConfigured;
        for provider in &self.providers {
            match provider.complete(messages, params).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    info!(provider = provider.name(), error = %e, "LLM provider failed, trying next");
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn is_available(&self) -> bool {
        for provider in &self.providers {
            if provider.is_available().await {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// LLM-powered assist functions
// ---------------------------------------------------------------------------

/// Generate a summary using the LLM with search context.
pub async fn llm_summarize(
    llm: &dyn LlmProvider,
    input: &str,
    context_snippets: &[String],
    sentence_limit: usize,
) -> Result<String, LlmError> {
    let context = if context_snippets.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRelevant context from the knowledge vault:\n{}",
            context_snippets
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a concise summarizer for a personal knowledge management system. \
             Produce a clear, factual summary that captures the key points. \
             Do not add information not present in the input or context. \
             Keep it to a few sentences.",
        ),
        ChatMessage::user(format!(
            "Summarize the following in at most {sentence_limit} sentences:\n\n{input}{context}"
        )),
    ];

    llm.complete(&messages, &CompletionParams::default()).await
}

/// Generate action items using the LLM.
pub async fn llm_action_items(
    llm: &dyn LlmProvider,
    input: &str,
    context_snippets: &[String],
    item_limit: usize,
) -> Result<Vec<String>, LlmError> {
    let context = if context_snippets.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRelevant context:\n{}",
            context_snippets
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a task extraction assistant for a personal knowledge management system. \
             Extract concrete, actionable next steps from the given text. \
             Return ONLY a plain list, one action item per line, no numbering, no bullets, no extra text.",
        ),
        ChatMessage::user(format!(
            "Extract up to {item_limit} action items from:\n\n{input}{context}"
        )),
    ];

    let result = llm
        .complete(&messages, &CompletionParams::default())
        .await?;

    let items: Vec<String> = result
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches(|c: char| {
                    c == '-' || c == '*' || c == '•' || c.is_ascii_digit()
                })
                .trim_start_matches('.')
                .trim_start_matches(')')
                .trim()
                .to_string()
        })
        .filter(|line| line.len() >= 8)
        .take(item_limit)
        .collect();

    if items.is_empty() {
        Err(LlmError::ParseError("no action items extracted".to_string()))
    } else {
        Ok(items)
    }
}

/// Generate completion suggestions using the LLM.
pub async fn llm_completion_suggestions(
    llm: &dyn LlmProvider,
    input: &str,
    context_snippets: &[String],
    limit: usize,
) -> Result<Vec<String>, LlmError> {
    let context = if context_snippets.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRelevant context:\n{}",
            context_snippets
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a writing completion assistant for a personal knowledge management system. \
             Propose concise continuation suggestions. \
             Return ONLY the suggestions, one per line, no numbering or bullets. \
             Keep each suggestion under 200 characters.",
        ),
        ChatMessage::user(format!(
            "Provide up to {limit} completion suggestions for:\n\n{input}{context}"
        )),
    ];

    let result = llm
        .complete(&messages, &CompletionParams::default())
        .await?;

    let mut seen = std::collections::HashSet::new();
    let suggestions: Vec<String> = result
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches(|c: char| c == '-' || c == '*' || c == '•')
                .trim()
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(|c| c == '.' || c == ')')
                .trim()
                .to_string()
        })
        .filter(|line| !line.is_empty())
        .filter(|line| {
            let key = line.to_ascii_lowercase();
            if seen.contains(&key) {
                false
            } else {
                seen.insert(key);
                true
            }
        })
        .map(|line| {
            if line.len() > 200 {
                line[..200].to_string()
            } else {
                line
            }
        })
        .take(limit)
        .collect();

    if suggestions.is_empty() {
        Err(LlmError::ParseError("no completion suggestions extracted".to_string()))
    } else {
        Ok(suggestions)
    }
}

/// Generate a refined version (summary + action items) using the LLM.
pub async fn llm_refine(
    llm: &dyn LlmProvider,
    input: &str,
    context_snippets: &[String],
    limit: usize,
) -> Result<String, LlmError> {
    let context = if context_snippets.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRelevant context:\n{}",
            context_snippets
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a writing assistant for a personal knowledge management system. \
             Refine the given text by producing: \
             1) A concise summary under '### Refined Summary' \
             2) Concrete next steps under '### Suggested Next Steps' as a markdown bullet list. \
             Do not add information not present in the input or context.",
        ),
        ChatMessage::user(format!(
            "Refine the following (summary: {limit} sentences max, next steps: {limit} items max):\n\n{input}{context}"
        )),
    ];

    llm.complete(&messages, &CompletionParams::default()).await
}

/// Generate a meeting notes summary with decisions and action items.
pub async fn llm_meeting_notes(
    llm: &dyn LlmProvider,
    input: &str,
    context_snippets: &[String],
    limit: usize,
) -> Result<String, LlmError> {
    let context = if context_snippets.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRelevant context:\n{}",
            context_snippets
                .iter()
                .take(6)
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a meeting note assistant for a personal knowledge management system. \
             Produce structured markdown with the following sections: \
             ### Meeting Summary, ### Decisions, ### Action Items, ### Open Questions. \
             Keep summaries concise, list decisions and questions as bullets, and action items as markdown task list items. \
             Do not add information not present in the input or context.",
        ),
        ChatMessage::user(format!(
            "Summarize the following meeting notes (summary: {limit} sentences max, decisions: {limit} items max, action items: {limit} items max, questions: {limit} items max):\n\n{input}{context}"
        )),
    ];

    llm.complete(&messages, &CompletionParams::default()).await
}

/// Generate a daily briefing summary using the LLM.
pub async fn llm_briefing_summary(
    llm: &dyn LlmProvider,
    due_today_count: usize,
    overdue_count: usize,
    in_progress_count: usize,
    habits_done: usize,
    habits_total: usize,
    task_titles: &[String],
    recent_note_titles: &[String],
) -> Result<String, LlmError> {
    let tasks_section = if task_titles.is_empty() {
        "No tasks due today.".to_string()
    } else {
        format!(
            "Tasks due today: {}",
            task_titles
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let notes_section = if recent_note_titles.is_empty() {
        String::new()
    } else {
        format!(
            "\nRecent notes: {}",
            recent_note_titles
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let messages = vec![
        ChatMessage::system(
            "You are a friendly daily briefing assistant for a personal knowledge management system. \
             Generate a brief, motivating daily summary (2-3 sentences). \
             Mention the most important items. Be concise and actionable.",
        ),
        ChatMessage::user(format!(
            "Generate a daily briefing:\n\
             - {due_today_count} tasks due today, {overdue_count} overdue, {in_progress_count} in progress\n\
             - Habits: {habits_done}/{habits_total} completed\n\
             - {tasks_section}\n\
             {notes_section}"
        )),
    ];

    llm.complete(&messages, &CompletionParams::default()).await
}

// ---------------------------------------------------------------------------
// Provider initialization
// ---------------------------------------------------------------------------

/// Probe local Ollama for availability.
pub async fn probe_ollama(base_url: &str, timeout_secs: u64) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs.min(5)))
        .build()
        .unwrap_or_default();
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    match client
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Initialize the LLM provider from config. Returns None if no provider available.
///
/// Resolution order (fallback chain):
/// 1. Local llama.cpp (if `local_llm.enabled` and model available)
/// 2. Explicitly configured provider (if `llm.enabled`)
/// 3. Auto-detected Ollama (if `llm.auto_detect`)
/// 4. OpenAI API (if API key available)
///
/// When multiple providers are available, they form a `FallbackLlmProvider`
/// chain that tries each in order until one succeeds.
pub async fn init_llm_provider(
    config: &LlmConfig,
    api_key: Option<String>,
) -> Option<Arc<dyn LlmProvider>> {
    init_llm_provider_with_local(config, &crate::config::LocalLlmConfig::default(), api_key).await
}

/// Extended LLM provider init that includes local llama.cpp configuration.
pub async fn init_llm_provider_with_local(
    config: &LlmConfig,
    local_config: &crate::config::LocalLlmConfig,
    api_key: Option<String>,
) -> Option<Arc<dyn LlmProvider>> {
    let mut providers: Vec<Arc<dyn LlmProvider>> = Vec::new();

    // 1. Local llama.cpp provider (highest priority — fully offline)
    if let Some(local_provider) = crate::llm_local::init_local_provider(local_config) {
        info!("Local llama.cpp provider added to fallback chain");
        providers.push(local_provider);
    }

    // 2. If explicitly enabled, use configured provider
    if config.enabled {
        let provider = OpenAiCompatibleLlm::from_config(config, api_key.clone());
        info!(
            provider = "openai-compatible",
            base_url = %config.base_url,
            model = %config.model,
            "LLM provider initialized (explicit)"
        );
        providers.push(Arc::new(provider));
    }

    // 3. Auto-detect local Ollama if enabled
    if config.auto_detect && !config.enabled {
        let ollama_url = "http://localhost:11434/v1";
        if probe_ollama(ollama_url, config.timeout_secs).await {
            let ollama_config = LlmConfig {
                enabled: true,
                base_url: ollama_url.into(),
                model: config.model.clone(),
                ..config.clone()
            };
            let provider = OpenAiCompatibleLlm::from_config(&ollama_config, None);
            info!("Auto-detected local Ollama at {}", ollama_url);
            providers.push(Arc::new(provider));
        }
    }

    // 4. Add OpenAI as fallback if API key present and not already the primary
    if let Some(key) = api_key {
        if !config.enabled || config.base_url != "https://api.openai.com/v1" {
            let openai_config = LlmConfig {
                enabled: true,
                base_url: "https://api.openai.com/v1".to_string(),
                model: "gpt-4o-mini".to_string(),
                ..config.clone()
            };
            let provider = OpenAiCompatibleLlm::from_config(&openai_config, Some(key));
            info!("Added OpenAI as fallback LLM provider");
            providers.push(Arc::new(provider));
        }
    }

    match providers.len() {
        0 => {
            info!("LLM provider disabled — using heuristic fallback for assist/briefing");
            None
        }
        1 => Some(providers.remove(0)),
        _ => Some(Arc::new(FallbackLlmProvider::new(providers))),
    }
}

/// Extract context snippets from search results for LLM prompts.
pub fn extract_context_snippets(results: &[hx_core::SearchResult], limit: usize) -> Vec<String> {
    results
        .iter()
        .take(limit)
        .filter_map(|r| {
            let title = r.node.title.as_deref().unwrap_or("Untitled");
            let content = r.node.content.trim();
            if content.is_empty() {
                None
            } else {
                let preview = if content.len() > 300 {
                    format!("{}...", &content[..300])
                } else {
                    content.to_string()
                };
                Some(format!("[{title}] {preview}"))
            }
        })
        .collect()
}
