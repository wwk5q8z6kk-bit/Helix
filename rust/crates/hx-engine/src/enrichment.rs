use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, info_span, warn};
use uuid::Uuid;

use crate::config::AiConfig;
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};
use hx_core::*;
use hx_storage::unified::UnifiedStore;

/// Event emitted when a node has been enriched.
#[derive(Debug, Clone)]
pub struct EnrichmentEvent {
    pub node_id: Uuid,
    pub success: bool,
}

/// The pipeline responsible for asynchronously enriching knowledge nodes with AI-extracted metadata.
pub struct EnrichmentPipeline {
    config: AiConfig,
    sender: mpsc::UnboundedSender<Uuid>,
}

impl EnrichmentPipeline {
    pub fn new(
        store: Arc<UnifiedStore>,
        config: AiConfig,
        llm: Option<Arc<dyn LlmProvider>>,
        change_tx: tokio::sync::broadcast::Sender<ChangeNotification>,
    ) -> (Self, EnrichmentWorker) {
        let (tx, rx) = mpsc::unbounded_channel();

        let pipeline = Self {
            config: config.clone(),
            sender: tx,
        };

        let worker = EnrichmentWorker::new(store, config, llm, rx, change_tx);

        (pipeline, worker)
    }

    /// Queue a node for enrichment.
    pub fn queue_enrichment(&self, node_id: Uuid) {
        if !self.config.enrichment_enabled {
            return;
        }

        if let Err(e) = self.sender.send(node_id) {
            error!("Failed to queue node {} for enrichment: {}", node_id, e);
        }
    }
}

/// Background worker that processes the enrichment queue.
pub struct EnrichmentWorker {
    store: Arc<UnifiedStore>,
    config: AiConfig,
    llm: Option<Arc<dyn LlmProvider>>,
    receiver: mpsc::UnboundedReceiver<Uuid>,
    change_tx: tokio::sync::broadcast::Sender<ChangeNotification>,
}

impl EnrichmentWorker {
    fn new(
        store: Arc<UnifiedStore>,
        config: AiConfig,
        llm: Option<Arc<dyn LlmProvider>>,
        receiver: mpsc::UnboundedReceiver<Uuid>,
        change_tx: tokio::sync::broadcast::Sender<ChangeNotification>,
    ) -> Self {
        Self {
            store,
            config,
            llm,
            receiver,
            change_tx,
        }
    }

    /// Start the worker loop.
    pub async fn run(mut self) {
        info!("EnrichmentWorker started");

        while let Some(node_id) = self.receiver.recv().await {
            let span = info_span!("enrich_node", %node_id);
            let _enter = span.enter();

            match self.process_node(node_id).await {
                Ok(_) => {
                    info!("Node successfully enriched");
                    let _ = self.change_tx.send(ChangeNotification {
                        operation: "enriched".into(),
                        node_id: node_id.to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        namespace: None,
                    });
                }
                Err(e) => warn!("Node enrichment failed: {}", e),
            }
        }

        info!("EnrichmentWorker shutting down");
    }

    async fn process_node(&self, node_id: Uuid) -> MvResult<()> {
        // 1. Fetch the node
        let Some(mut node) = self.store.nodes.get(node_id).await? else {
            return Err(HxError::NodeNotFound(node_id));
        };

        info!(
            "Enriching node: {}",
            node.title.as_deref().unwrap_or("[No Title]")
        );

        // 2. Perform AI Enrichment via LlmProvider
        self.perform_llm_enrichment(&mut node).await?;

        // 3. Save the enriched node
        self.store.nodes.update(&node).await?;

        Ok(())
    }

    async fn perform_llm_enrichment(&self, node: &mut KnowledgeNode) -> MvResult<()> {
        let llm = match &self.llm {
            Some(llm) => llm,
            None => {
                warn!("No LLM provider available for enrichment, skipping");
                return Ok(());
            }
        };

        let prompt = format!(
            "Analyze the following content and extract: \n\
			1. Up to 5 relevant tags (e.g. #project-x, #research)\n\
			2. A short title if missing\n\
			3. Importance score (0.0 to 1.0)\n\n\
			Content: {content}\n\n\
			Return JSON: {{\"tags\": [], \"title\": \"\", \"importance\": 0.0}}",
            content = node.content
        );

        let messages = vec![
            ChatMessage::system(
                "You are a metadata extraction assistant for a knowledge management system. \
				 Always respond with valid JSON matching the requested schema.",
            ),
            ChatMessage::user(prompt),
        ];

        let params = CompletionParams {
            model: Some(self.config.enrichment_model.clone()),
            ..CompletionParams::default()
        };

        let response = llm
            .complete(&messages, &params)
            .await
            .map_err(|e| HxError::Embedding(format!("LLM enrichment failed: {e}")))?;

        #[derive(serde::Deserialize)]
        struct EnrichmentData {
            tags: Vec<String>,
            title: String,
            importance: f64,
        }

        let data: Option<EnrichmentData> = serde_json::from_str(&response).ok();

        if let Some(data) = data {
            // Apply enriched tags (merge)
            for tag in data.tags {
                if !node.tags.contains(&tag) {
                    node.tags.push(tag);
                }
            }

            // Apply title if missing or very short
            if (node.title.is_none() || node.title.as_ref().is_some_and(|t| t.len() < 5))
                && !data.title.is_empty()
            {
                node.title = Some(data.title);
            }

            // Apply importance (weighted average)
            node.importance = (node.importance + data.importance) / 2.0;

            info!(
                "Enrichment applied: title='{}', tags_added={}",
                node.title.as_deref().unwrap_or(""),
                node.tags.len()
            );
        } else {
            warn!("Failed to parse enrichment response as JSON, skipping");
        }

        Ok(())
    }
}
