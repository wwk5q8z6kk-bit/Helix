use super::*;
use tempfile::TempDir;

pub async fn create_test_engine() -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    config.embedding.provider = "noop".into();
    let engine = HelixEngine::init(config).await.unwrap();
    (engine, temp_dir)
}

pub async fn create_test_engine_with_ai_auto_tagging() -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    config.embedding.provider = "noop".into();
    config.ai.auto_tagging_enabled = true;
    config.ai.auto_tagging_similarity_seed_limit = 8;
    config.ai.auto_tagging_max_generated_tags = 6;
    config.ai.auto_tagging_max_total_tags = 12;
    let engine = HelixEngine::init(config).await.unwrap();
    (engine, temp_dir)
}

pub async fn create_test_engine_with_local_embedding_provider(
    provider: &str,
    model: &str,
) -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    config.embedding.provider = provider.to_string();
    config.embedding.model = model.to_string();
    let engine = HelixEngine::init(config).await.unwrap();
    (engine, temp_dir)
}

pub async fn create_test_engine_with_backlinks_disabled() -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    config.embedding.provider = "noop".into();
    config.linking.auto_backlinks_enabled = false;
    let engine = HelixEngine::init(config).await.unwrap();
    (engine, temp_dir)
}

pub async fn create_test_sealed_engine(unseal_vault: bool) -> (HelixEngine, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let mut config = EngineConfig {
        data_dir: temp_dir.path().to_string_lossy().to_string(),
        ..Default::default()
    };
    config.embedding.provider = "noop".into();
    config.sealed_mode = true;
    let engine = HelixEngine::init(config).await.unwrap();
    engine
        .keychain
        .initialize_vault("test-password", false, "test-suite")
        .await
        .unwrap();
    if unseal_vault {
        engine
            .keychain
            .unseal("test-password", "test-suite")
            .await
            .unwrap();
    } else {
        engine.keychain.seal("test-suite").await.unwrap();
    }
    (engine, temp_dir)
}

#[tokio::test]
async fn test_store_and_retrieve_node() {
    let (engine, _tmp_dir) = create_test_engine().await;
    let node = KnowledgeNode::new(NodeKind::Fact, "Test content".to_string())
        .with_title("Test Title")
        .with_tags(vec!["test".to_string()]);

    let stored_node = engine.store_node(node.clone()).await.unwrap();
    assert_eq!(stored_node.content, "Test content");
    assert_eq!(stored_node.title, Some("Test Title".to_string()));

    let retrieved = engine.get_node(stored_node.id).await.unwrap().unwrap();
    assert_eq!(retrieved.content, "Test content");
    assert_eq!(retrieved.tags, vec!["test"]);
}

#[tokio::test]
async fn test_update_node() {
    let (engine, _tmp_dir) = create_test_engine().await;
    let node = KnowledgeNode::new(NodeKind::Fact, "Original content".to_string());
    let stored = engine.store_node(node).await.unwrap();

    let mut updated = stored.clone();
    updated.content = "Updated content".to_string();
    updated.title = Some("Updated title".to_string());

    let updated_node = engine.update_node(updated).await.unwrap();
    assert_eq!(updated_node.content, "Updated content");
    assert_eq!(updated_node.title, Some("Updated title".to_string()));
}

#[tokio::test]
async fn test_delete_node() {
    let (engine, _tmp_dir) = create_test_engine().await;
    let node = KnowledgeNode::new(NodeKind::Fact, "To delete".to_string());
    let stored = engine.store_node(node).await.unwrap();

    assert!(engine.delete_node(stored.id).await.unwrap());
    assert!(engine.get_node(stored.id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_node_count() {
    let (engine, _tmp_dir) = create_test_engine().await;

    let count = engine.node_count().await.unwrap();
    assert_eq!(count, 0);

    engine
        .store_node(KnowledgeNode::new(NodeKind::Fact, "First".to_string()))
        .await
        .unwrap();
    engine
        .store_node(KnowledgeNode::new(NodeKind::Fact, "Second".to_string()))
        .await
        .unwrap();

    let count = engine.node_count().await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_sealed_engine_blocks_node_io_while_sealed() {
    let (engine, _tmp_dir) = create_test_sealed_engine(false).await;
    let err = engine
        .store_node(KnowledgeNode::new(NodeKind::Fact, "blocked".to_string()))
        .await
        .expect_err("sealed store_node must fail");
    assert!(matches!(err, HxError::VaultSealed));
}

#[tokio::test]
async fn test_sealed_migrate_and_rebuild_require_unseal_then_succeed() {
    let (engine, _tmp_dir) = create_test_sealed_engine(false).await;

    let rebuild_err = engine
        .rebuild_runtime_indexes()
        .await
        .expect_err("sealed rebuild should fail");
    assert!(matches!(rebuild_err, HxError::VaultSealed));

    let migrate_err = engine
        .migrate_sealed_storage()
        .await
        .expect_err("sealed migrate should fail");
    assert!(matches!(migrate_err, HxError::VaultSealed));

    engine
        .keychain
        .unseal("test-password", "test-suite")
        .await
        .unwrap();

    let node = engine
        .store_node(KnowledgeNode::new(
            NodeKind::Fact,
            "sealed migration validation".to_string(),
        ))
        .await
        .unwrap();

    engine
        .rebuild_runtime_indexes()
        .await
        .expect("rebuild after unseal");
    engine
        .migrate_sealed_storage()
        .await
        .expect("migrate after unseal");

    let loaded = engine
        .get_node(node.id)
        .await
        .expect("load node")
        .expect("node exists");
    assert_eq!(loaded.content, "sealed migration validation");
}

#[tokio::test]
async fn test_sealed_restart_cycle_recovers_data_after_unseal_and_rebuild() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_string_lossy().to_string();

    let mut config = EngineConfig {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    config.embedding.provider = "noop".into();
    config.sealed_mode = true;

    let engine = HelixEngine::init(config.clone()).await.unwrap();
    engine
        .keychain
        .initialize_vault("restart-password", false, "test-suite")
        .await
        .unwrap();

    let stored = engine
        .store_node(KnowledgeNode::new(
            NodeKind::Fact,
            "restart lifecycle keeps encrypted knowledge".to_string(),
        ))
        .await
        .unwrap();

    engine.keychain.seal("test-suite").await.unwrap();
    drop(engine);

    let restarted = HelixEngine::init(config).await.unwrap();
    assert!(
        restarted.is_sealed(),
        "engine should start sealed after restart"
    );

    let sealed_err = restarted
        .get_node(stored.id)
        .await
        .expect_err("sealed restart should block node reads");
    assert!(matches!(sealed_err, HxError::VaultSealed));

    restarted
        .keychain
        .unseal("restart-password", "test-suite")
        .await
        .unwrap();
    restarted
        .migrate_sealed_storage()
        .await
        .expect("migrate after restart");
    restarted
        .rebuild_runtime_indexes()
        .await
        .expect("rebuild after restart");

    let loaded = restarted
        .get_node(stored.id)
        .await
        .expect("load node")
        .expect("node exists");
    assert_eq!(
        loaded.content,
        "restart lifecycle keeps encrypted knowledge"
    );

    let recall_results = restarted
        .recall(
            &MemoryQuery::new("restart lifecycle")
                .with_strategy(SearchStrategy::FullText)
                .with_limit(10)
                .with_min_score(0.0),
        )
        .await
        .expect("recall should work after restart rebuild");
    assert!(
        recall_results
            .iter()
            .any(|result| result.node.id == stored.id),
        "recalled results should include the stored node"
    );
}

#[tokio::test]
async fn test_relationships() {
    let (engine, _tmp_dir) = create_test_engine().await;

    let node1 = engine
        .store_node(KnowledgeNode::new(NodeKind::Fact, "Node 1".to_string()))
        .await
        .unwrap();
    let node2 = engine
        .store_node(KnowledgeNode::new(NodeKind::Fact, "Node 2".to_string()))
        .await
        .unwrap();

    let relation = Relationship {
        id: uuid::Uuid::new_v4(),
        from_node: node1.id,
        to_node: node2.id,
        kind: RelationKind::RelatesTo,
        weight: 1.0,
        metadata: Default::default(),
        created_at: chrono::Utc::now(),
    };

    engine.add_relationship(&relation).await.unwrap();

    let neighbors = engine.get_neighbors(node1.id, 1).await.unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0], node2.id);
}

#[tokio::test]
async fn test_update_profile_syncs_owner_relay_contact() {
    let (engine, _tmp_dir) = create_test_engine().await;

    let updated = engine
        .update_profile(&UpdateProfileRequest {
            display_name: Some("Owner".to_string()),
            email: Some("owner@example.com".to_string()),
            signature_public_key: Some("pk-owner".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();

    let contact_id = updated
        .metadata
        .get("relay_contact_id")
        .and_then(|value| value.as_str())
        .and_then(|value| uuid::Uuid::parse_str(value).ok())
        .expect("relay_contact_id set");

    let contact = engine.relay.get_contact(contact_id).await.unwrap().unwrap();
    assert_eq!(contact.display_name, "Owner");
    assert_eq!(contact.public_key, "pk-owner");
    assert_eq!(
        contact.vault_address.as_deref(),
        Some("mailto:owner@example.com")
    );
}

#[tokio::test]
async fn test_relay_inbound_creates_reply_proposal() {
    let (engine, _tmp_dir) = create_test_engine().await;

    let contact = RelayContact::new("Alice", "pk-alice").with_trust(TrustLevel::ContextInject);
    engine.relay.add_contact(&contact).await.unwrap();

    let channel = RelayChannel::direct(contact.id);
    engine.relay.create_channel(&channel).await.unwrap();

    let node = KnowledgeNode::new(
        NodeKind::Fact,
        "Project Atlas roadmap lives in the Q2 plan.".to_string(),
    );
    engine.store_node(node).await.unwrap();

    let message =
        RelayMessage::inbound(channel.id, contact.id, "Can you share the Atlas roadmap?");
    let outcome = engine
        .receive_relay_message(message, "default")
        .await
        .unwrap();

    assert!(outcome.proposal_id.is_some());
    assert!(outcome.auto_reply.is_none());
    assert_eq!(outcome.message.status, MessageStatus::Deferred);

    let proposals = engine
        .list_proposals(Some(ProposalState::Pending), 10, 0)
        .await
        .unwrap();
    assert!(proposals.iter().any(|proposal| {
        proposal.action == ProposalAction::Custom("relay.reply".to_string())
    }));
}
