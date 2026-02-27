//! Local LLM inference via llama.cpp (GGUF models).
//!
//! Implements `LlmProvider` using the `llama-cpp-2` crate for fully offline
//! inference on CPU. Models must be GGUF-quantized (Q4_K_M recommended for
//! 3B-8B parameter models on 16GB RAM machines).
//!
//! This module is behind the `local-llm` feature flag and is not compiled
//! by default.

#[cfg(feature = "local-llm")]
mod inner {
    use std::path::PathBuf;
    use std::sync::Arc;

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel, Special};
    use llama_cpp_2::sampling::LlamaSampler;
    use tokio::sync::Mutex;
    use tracing::{debug, info, warn};

    use crate::config::LocalLlmConfig;
    use crate::llm::{ChatMessage, CompletionParams, LlmError, LlmProvider};

    /// Local LLM provider using llama.cpp for GGUF model inference.
    pub struct LlamaCppProvider {
        model: Arc<LlamaModel>,
        backend: Arc<LlamaBackend>,
        config: LocalLlmConfig,
        /// Serialize inference requests — llama.cpp context is not thread-safe.
        inference_lock: Mutex<()>,
    }

    impl LlamaCppProvider {
        /// Load a GGUF model from the given path.
        pub fn load(model_path: PathBuf, config: &LocalLlmConfig) -> Result<Self, LlmError> {
            if !model_path.exists() {
                return Err(LlmError::RequestFailed(format!(
                    "model file not found: {}",
                    model_path.display()
                )));
            }

            let file_size = std::fs::metadata(&model_path)
                .map(|m| m.len())
                .unwrap_or(0);
            if file_size > config.max_ram_bytes {
                return Err(LlmError::RequestFailed(format!(
                    "model file ({:.1} GB) exceeds max_ram_bytes ({:.1} GB)",
                    file_size as f64 / 1e9,
                    config.max_ram_bytes as f64 / 1e9,
                )));
            }

            info!(
                path = %model_path.display(),
                size_gb = format!("{:.1}", file_size as f64 / 1e9),
                "loading local GGUF model"
            );

            let backend =
                LlamaBackend::init().map_err(|e| LlmError::RequestFailed(e.to_string()))?;

            let model_params = {
                let mut p = LlamaModelParams::default();
                p = p.with_n_gpu_layers(config.gpu_layers);
                p
            };

            let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
                .map_err(|e| LlmError::RequestFailed(format!("failed to load model: {e}")))?;

            info!("local GGUF model loaded successfully");

            Ok(Self {
                model: Arc::new(model),
                backend: Arc::new(backend),
                config: config.clone(),
                inference_lock: Mutex::new(()),
            })
        }

        /// Format chat messages into a prompt string.
        ///
        /// Uses ChatML format which is widely supported by instruction-tuned
        /// GGUF models (Llama 3, Mistral, Phi, Qwen, etc.).
        fn format_prompt(messages: &[ChatMessage]) -> String {
            let mut prompt = String::new();
            for msg in messages {
                prompt.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    msg.role, msg.content
                ));
            }
            prompt.push_str("<|im_start|>assistant\n");
            prompt
        }

        /// Run inference synchronously (called from within the lock).
        fn generate_sync(
            &self,
            prompt: &str,
            max_tokens: u32,
            temperature: f32,
        ) -> Result<String, LlmError> {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(self.config.context_size))
                .with_n_threads(self.config.threads)
                .with_n_threads_batch(self.config.threads);

            let mut ctx = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| LlmError::RequestFailed(format!("context creation failed: {e}")))?;

            // Tokenize the prompt
            let tokens = self
                .model
                .str_to_token(prompt, AddBos::Always)
                .map_err(|e| LlmError::RequestFailed(format!("tokenization failed: {e}")))?;

            let n_tokens = tokens.len();
            if n_tokens == 0 {
                return Ok(String::new());
            }

            // Check context size
            let ctx_size = self.config.context_size as usize;
            if n_tokens >= ctx_size {
                return Err(LlmError::RequestFailed(format!(
                    "prompt ({n_tokens} tokens) exceeds context size ({ctx_size})"
                )));
            }

            // Create batch and evaluate prompt tokens
            let mut batch = LlamaBatch::new(ctx_size, 1);

            // Add all prompt tokens to the batch
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == n_tokens - 1;
                batch
                    .add(token, i as i32, &[0], is_last)
                    .map_err(|e| LlmError::RequestFailed(format!("batch add failed: {e}")))?;
            }

            // Evaluate the prompt
            ctx.decode(&mut batch)
                .map_err(|e| LlmError::RequestFailed(format!("decode failed: {e}")))?;

            // Set up sampler chain: temperature + greedy
            let sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::greedy(),
            ]);

            // Generate tokens
            let mut output = String::new();
            let mut n_decoded = 0;
            let max_gen = max_tokens.min((ctx_size - n_tokens) as u32);

            loop {
                if n_decoded >= max_gen {
                    break;
                }

                let token = sampler.sample(&ctx, batch.n_tokens() - 1);

                // Check for end-of-generation
                if self.model.is_eog_token(token) {
                    break;
                }

                let piece = self
                    .model
                    .token_to_str(token, Special::Tokenize)
                    .map_err(|e| {
                        LlmError::RequestFailed(format!("token to string failed: {e}"))
                    })?;

                // Stop on ChatML end token
                if piece.contains("<|im_end|>") || piece.contains("<|endoftext|>") {
                    break;
                }

                output.push_str(&piece);
                n_decoded += 1;

                // Prepare next batch
                batch.clear();
                batch
                    .add(token, (n_tokens + n_decoded as usize) as i32, &[0], true)
                    .map_err(|e| LlmError::RequestFailed(format!("batch add failed: {e}")))?;

                ctx.decode(&mut batch)
                    .map_err(|e| LlmError::RequestFailed(format!("decode failed: {e}")))?;
            }

            debug!(
                prompt_tokens = n_tokens,
                generated_tokens = n_decoded,
                "local LLM generation complete"
            );

            Ok(output.trim().to_string())
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for LlamaCppProvider {
        async fn complete(
            &self,
            messages: &[ChatMessage],
            params: &CompletionParams,
        ) -> Result<String, LlmError> {
            let _guard = self.inference_lock.lock().await;

            let prompt = Self::format_prompt(messages);
            let max_tokens = params.max_tokens.unwrap_or(self.config.context_size / 2);
            let temperature = params.temperature.unwrap_or(0.3);

            // Run inference on a blocking thread since llama.cpp is synchronous
            let model = self.model.clone();
            let backend = self.backend.clone();
            let config = self.config.clone();

            // We need to create a new provider-like struct for the blocking call
            // since we can't send &self across threads easily.
            let prompt_clone = prompt.clone();
            tokio::task::spawn_blocking(move || {
                // Re-create a minimal context for this blocking call
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(std::num::NonZeroU32::new(config.context_size))
                    .with_n_threads(config.threads)
                    .with_n_threads_batch(config.threads);

                let mut ctx = model
                    .new_context(&backend, ctx_params)
                    .map_err(|e| {
                        LlmError::RequestFailed(format!("context creation failed: {e}"))
                    })?;

                let tokens = model
                    .str_to_token(&prompt_clone, AddBos::Always)
                    .map_err(|e| LlmError::RequestFailed(format!("tokenization failed: {e}")))?;

                let n_tokens = tokens.len();
                if n_tokens == 0 {
                    return Ok(String::new());
                }

                let ctx_size = config.context_size as usize;
                if n_tokens >= ctx_size {
                    return Err(LlmError::RequestFailed(format!(
                        "prompt ({n_tokens} tokens) exceeds context size ({ctx_size})"
                    )));
                }

                let mut batch = LlamaBatch::new(ctx_size, 1);
                for (i, &token) in tokens.iter().enumerate() {
                    let is_last = i == n_tokens - 1;
                    batch
                        .add(token, i as i32, &[0], is_last)
                        .map_err(|e| {
                            LlmError::RequestFailed(format!("batch add failed: {e}"))
                        })?;
                }

                ctx.decode(&mut batch)
                    .map_err(|e| LlmError::RequestFailed(format!("decode failed: {e}")))?;

                let sampler = LlamaSampler::chain_simple([
                    LlamaSampler::temp(temperature),
                    LlamaSampler::greedy(),
                ]);

                let mut output = String::new();
                let mut n_decoded: u32 = 0;
                let max_gen = max_tokens.min((ctx_size - n_tokens) as u32);

                loop {
                    if n_decoded >= max_gen {
                        break;
                    }

                    let token = sampler.sample(&ctx, batch.n_tokens() - 1);

                    if model.is_eog_token(token) {
                        break;
                    }

                    let piece = model
                        .token_to_str(token, Special::Tokenize)
                        .map_err(|e| {
                            LlmError::RequestFailed(format!("token to string failed: {e}"))
                        })?;

                    if piece.contains("<|im_end|>") || piece.contains("<|endoftext|>") {
                        break;
                    }

                    output.push_str(&piece);
                    n_decoded += 1;

                    batch.clear();
                    batch
                        .add(
                            token,
                            (n_tokens + n_decoded as usize) as i32,
                            &[0],
                            true,
                        )
                        .map_err(|e| {
                            LlmError::RequestFailed(format!("batch add failed: {e}"))
                        })?;

                    ctx.decode(&mut batch)
                        .map_err(|e| {
                            LlmError::RequestFailed(format!("decode failed: {e}"))
                        })?;
                }

                Ok(output.trim().to_string())
            })
            .await
            .map_err(|e| LlmError::RequestFailed(format!("spawn_blocking failed: {e}")))?
        }

        fn name(&self) -> &str {
            "local-llama-cpp"
        }

        async fn is_available(&self) -> bool {
            true
        }
    }

    /// Try to initialize a local LlamaCpp provider from config.
    /// Returns None if disabled or no model is available.
    pub fn init_local_provider(
        config: &LocalLlmConfig,
    ) -> Option<Arc<dyn LlmProvider>> {
        if !config.enabled {
            return None;
        }

        // Resolve model path: explicit path > model_id in models_dir > scan models_dir
        let model_path = if let Some(ref path) = config.model_path {
            PathBuf::from(path)
        } else if let Some(ref model_id) = config.model_id {
            // Look for the model in models_dir
            let dir = PathBuf::from(&config.models_dir);
            let candidate = dir.join(model_id);
            if candidate.exists() {
                candidate
            } else {
                // Try with .gguf extension
                let with_ext = dir.join(format!("{model_id}.gguf"));
                if with_ext.exists() {
                    with_ext
                } else {
                    warn!(
                        model_id = model_id,
                        models_dir = %config.models_dir,
                        "local LLM model not found"
                    );
                    return None;
                }
            }
        } else {
            // Scan models_dir for any .gguf file
            let dir = PathBuf::from(&config.models_dir);
            if !dir.exists() {
                info!("local LLM models directory does not exist: {}", dir.display());
                return None;
            }
            match find_first_gguf(&dir) {
                Some(path) => {
                    info!(path = %path.display(), "auto-detected GGUF model");
                    path
                }
                None => {
                    info!("no GGUF models found in {}", dir.display());
                    return None;
                }
            }
        };

        match LlamaCppProvider::load(model_path, config) {
            Ok(provider) => Some(Arc::new(provider)),
            Err(e) => {
                warn!(error = %e, "failed to initialize local LLM provider");
                None
            }
        }
    }

    /// Find the first .gguf file in a directory (non-recursive).
    fn find_first_gguf(dir: &std::path::Path) -> Option<PathBuf> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut gguf_files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .collect();
        // Sort by name for deterministic selection
        gguf_files.sort();
        gguf_files.into_iter().next()
    }
}

#[cfg(feature = "local-llm")]
pub use inner::*;

// When the feature is disabled, provide a stub init function so that
// engine.rs can always call init_local_provider() without cfg guards.
#[cfg(not(feature = "local-llm"))]
pub fn init_local_provider(
    _config: &crate::config::LocalLlmConfig,
) -> Option<std::sync::Arc<dyn crate::llm::LlmProvider>> {
    if _config.enabled {
        tracing::warn!("local-llm feature is not enabled at compile time; ignoring local_llm.enabled=true");
    }
    None
}

/// Returns true if the `local-llm` feature was enabled at compile time.
pub fn is_feature_enabled() -> bool {
    cfg!(feature = "local-llm")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_none_when_disabled() {
        let config = crate::config::LocalLlmConfig::default();
        assert!(!config.enabled);
        let result = init_local_provider(&config);
        assert!(result.is_none());
    }

    #[cfg(not(feature = "local-llm"))]
    #[test]
    fn stub_warns_when_enabled_without_feature() {
        let config = crate::config::LocalLlmConfig {
            enabled: true,
            ..Default::default()
        };
        let result = init_local_provider(&config);
        assert!(result.is_none());
    }
}
