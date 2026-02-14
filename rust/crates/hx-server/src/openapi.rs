//! OpenAPI specification and Swagger UI for Helix API.
#![allow(dead_code)]

use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Helix API OpenAPI specification.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Helix API",
        version = env!("CARGO_PKG_VERSION"),
        description = "Helix is a local-first knowledge management system with hybrid semantic retrieval.",
        license(name = "MIT"),
        contact(name = "Helix Contributors")
    ),
    servers(
        (url = "http://localhost:9470", description = "Local development server")
    ),
    tags(
        (name = "health", description = "Health check endpoints"),
        (name = "nodes", description = "Knowledge node CRUD operations"),
        (name = "recall", description = "Semantic recall and search"),
        (name = "graph", description = "Relationship graph operations"),
        (name = "daily-notes", description = "Daily notes management"),
        (name = "calendar", description = "Calendar and iCal integration"),
        (name = "tasks", description = "Task management"),
        (name = "templates", description = "Template management"),
        (name = "files", description = "File attachments"),
        (name = "voice", description = "Voice notes and transcription"),
        (name = "assist", description = "AI writing assistance"),
        (name = "export-import", description = "Data export and import"),
        (name = "audit", description = "Audit logging"),
        (name = "metrics", description = "Prometheus metrics"),
        (name = "saved-searches", description = "Saved search presets"),
        (name = "saved-views", description = "Saved view presets"),
        (name = "permissions", description = "Permission templates and access keys"),
        (name = "oauth", description = "OAuth2 client credentials"),
        (name = "profile", description = "Owner profile settings"),
        (name = "exchange", description = "Exchange proposal inbox"),
        (name = "autonomy", description = "Autonomy rules and evaluation"),
        (name = "keychain", description = "Encryption vault and credential management"),
        (name = "relay", description = "Messaging relay contacts and channels"),
        (name = "safeguards", description = "Blocked senders, auto-approve rules, undo"),
        (name = "plugins", description = "WASM plugin lifecycle"),
        (name = "sync", description = "Device sync export/import"),
        (name = "federation", description = "Federated peer queries"),
        (name = "adapters", description = "External service adapters (email, etc.)"),
        (name = "agent", description = "Agent context, intents, insights, feedback"),
        (name = "proactive", description = "Proactive insight generation"),
        (name = "graph-extra", description = "Graph relationship management"),
        (name = "clips", description = "Web clip import and enrichment"),
        (name = "secrets", description = "Server-side secret storage"),
        (name = "ai", description = "AI sidecar proxy"),
        (name = "sharing", description = "Public share links"),
        (name = "comments", description = "Node comments and annotations"),
        (name = "mcp-marketplace", description = "MCP connector registry")
    ),
    paths(
        // Health
        health,
        embedding_diagnostics,
        diagnostics_health,
        diagnostics_performance,
        // Nodes
        store_node,
        list_nodes,
        get_node,
        update_node,
        delete_node,
        // Comments
        create_node_comment,
        list_node_comments,
        resolve_node_comment,
        delete_node_comment,
        // Recall & Search
        recall,
        search,
        // Graph
        add_relationship,
        get_node_backlinks,
        get_neighbors,
        // Daily notes
        list_daily_notes,
        ensure_daily_note,
        // Calendar
        list_calendar_items,
        export_calendar_ical,
        import_calendar_ical,
        google_calendar_status,
        google_calendar_sync,
        // AI sidecar proxy
        ai_health,
        ai_models,
        ai_embeddings,
        ai_chat_completions,
        // Sharing
        create_public_share,
        list_public_shares,
        revoke_public_share,
        get_public_share,
        // MCP Marketplace
        list_mcp_connectors,
        get_mcp_connector,
        create_mcp_connector,
        update_mcp_connector,
        delete_mcp_connector,
        // Tasks
        list_due_tasks,
        prioritize_tasks,
        complete_task,
        reopen_task,
        snooze_task_reminder,
        // Templates
        list_templates,
        create_template,
        update_template,
        delete_template,
        instantiate_template,
        apply_template,
        // Files
        upload_file,
        list_attachments_index,
        list_node_attachments,
        get_attachment_chunks,
        reindex_attachment,
        download_attachment,
        delete_attachment,
        // Voice
        upload_voice_note,
        // Assist
        assist_completion,
        assist_autocomplete,
        assist_links,
        assist_transform,
        // Export/Import
        export_bundle,
        import_bundle,
        // Audit
        list_audit_logs,
        // Saved searches
        list_saved_searches,
        create_saved_search,
        update_saved_search,
        delete_saved_search,
        run_saved_search,
        // Saved views
        list_saved_views,
        create_saved_view,
        update_saved_view,
        delete_saved_view,
        // Permissions
        list_permission_templates,
        create_permission_template,
        update_permission_template,
        delete_permission_template,
        list_access_keys,
        create_access_key,
        revoke_access_key,
        // OAuth
        list_oauth_clients,
        create_oauth_client,
        revoke_oauth_client,
        oauth_token,
        // Profile
        get_profile,
        update_profile,
        // Exchange
        list_proposals,
        submit_proposal,
        get_proposal,
        approve_proposal,
        reject_proposal,
        inbox_count,
        batch_proposals,
        // Autonomy
        list_autonomy_rules,
        create_autonomy_rule,
        get_autonomy_rule,
        update_autonomy_rule,
        delete_autonomy_rule,
        list_autonomy_action_log,
        evaluate_autonomy,
        // Keychain
        keychain_init,
        keychain_unseal,
        keychain_seal,
        keychain_status,
        keychain_rotate,
        keychain_list_epochs,
        keychain_store_credential,
        keychain_list_credentials,
        keychain_read_credential,
        keychain_update_credential,
        keychain_destroy_credential,
        keychain_create_domain,
        keychain_list_domains,
        keychain_create_delegation,
        keychain_list_delegations,
        keychain_revoke_delegation,
        keychain_list_audit,
        keychain_backup,
        keychain_restore,
        // Relay
        relay_list_contacts,
        relay_create_contact,
        relay_get_contact,
        relay_update_contact,
        relay_delete_contact,
        relay_list_channels,
        relay_create_channel,
        relay_delete_channel,
        relay_list_messages,
        relay_send_message,
        relay_mark_read,
        relay_unread_count,
        // Safeguards
        list_blocked_senders,
        add_blocked_sender,
        remove_blocked_sender,
        list_auto_approve_rules,
        add_auto_approve_rule,
        update_auto_approve_rule,
        remove_auto_approve_rule,
        undo_proposal,
        // Plugins
        list_plugins_api,
        install_plugin_api,
        list_hook_points_api,
        reload_plugins_api,
        uninstall_plugin_api,
        plugins_runtime_list,
        plugins_runtime_get,
        plugins_runtime_reload,
        plugins_runtime_unload,
        plugins_runtime_hooks,
        // Sync
        sync_export,
        sync_import,
        sync_status,
        // Federation
        federation_list_peers,
        federation_add_peer,
        federation_remove_peer,
        federation_peer_health,
        federation_identity,
        federation_query,
        // Adapters
        adapters_list,
        adapters_register,
        adapters_statuses,
        adapter_get_status,
        adapter_remove,
        adapter_send,
        adapter_health,
        // Agent
        agent_context,
        agent_chronicle,
        agent_list_intents,
        agent_apply_intent,
        agent_dismiss_intent,
        agent_list_models,
        agent_watcher_status,
        agent_list_insights,
        agent_record_feedback,
        agent_list_feedback,
        agent_reflection_stats,
        // Proactive
        proactive_list_insights,
        proactive_generate,
        proactive_delete_insight,
        // Clips
        clips_import,
        clips_enrich,
        clips_create_note,
        // Graph extra
        delete_relationship,
        // Secrets
        set_secret,
        secret_status,
        delete_secret,
    ),
    components(
        schemas(
            HealthResponse,
            ProfileResponse,
            UpdateProfileRequest,
            KnowledgeNode,
            NodeKind,
            RecallRequest,
            RecallResult,
            StoreNodeRequest,
            UpdateNodeRequest,
            RelationshipRequest,
            AuditEntry,
            PermissionTemplateResponse,
            AccessKeyResponse,
            AccessKeyCreateResponse,
            OAuthClientCreateRequest,
            OAuthClientResponse,
            OAuthClientCreateResponse,
            OAuthTokenRequest,
            OAuthTokenResponse,
            FederationIdentityResponse,
            NodeCommentResponse,
            CreateNodeCommentRequest,
            McpConnectorResponse,
            CreateMcpConnectorRequest,
            UpdateMcpConnectorRequest,
        )
    )
)]
pub struct ApiDoc;

// Placeholder path operations - these need to be defined with #[utoipa::path] in rest.rs
// For now, we'll create a minimal spec

#[utoipa::path(
    get,
    path = "/api/v1/health",
    tag = "health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
async fn health() {}

#[utoipa::path(
    get,
    path = "/api/v1/diagnostics/embedding",
    tag = "health",
    responses(
        (status = 200, description = "Embedding provider diagnostics")
    )
)]
async fn embedding_diagnostics() {}

#[utoipa::path(
    get,
    path = "/api/v1/diagnostics/health",
    tag = "health",
    responses(
        (status = 200, description = "Health check with performance info")
    )
)]
async fn diagnostics_health() {}

#[utoipa::path(
    get,
    path = "/api/v1/diagnostics/performance",
    tag = "metrics",
    responses(
        (status = 200, description = "Histogram-based performance stats")
    )
)]
async fn diagnostics_performance() {}

#[utoipa::path(
    post,
    path = "/api/v1/nodes",
    tag = "nodes",
    request_body = StoreNodeRequest,
    responses(
        (status = 201, description = "Node created", body = KnowledgeNode)
    )
)]
async fn store_node() {}

#[utoipa::path(
    get,
    path = "/api/v1/nodes",
    tag = "nodes",
    params(
        ("limit" = Option<usize>, Query, description = "Max results"),
        ("offset" = Option<usize>, Query, description = "Pagination offset"),
        ("namespace" = Option<String>, Query, description = "Filter by namespace"),
        ("kind" = Option<String>, Query, description = "Filter by node kind")
    ),
    responses(
        (status = 200, description = "List of nodes", body = Vec<KnowledgeNode>)
    )
)]
async fn list_nodes() {}

#[utoipa::path(
    get,
    path = "/api/v1/nodes/{id}",
    tag = "nodes",
    params(
        ("id" = String, Path, description = "Node UUID")
    ),
    responses(
        (status = 200, description = "Node found", body = KnowledgeNode),
        (status = 404, description = "Node not found")
    )
)]
async fn get_node() {}

#[utoipa::path(
    put,
    path = "/api/v1/nodes/{id}",
    tag = "nodes",
    params(
        ("id" = String, Path, description = "Node UUID")
    ),
    request_body = UpdateNodeRequest,
    responses(
        (status = 200, description = "Node updated", body = KnowledgeNode),
        (status = 404, description = "Node not found")
    )
)]
async fn update_node() {}

#[utoipa::path(
    delete,
    path = "/api/v1/nodes/{id}",
    tag = "nodes",
    params(
        ("id" = String, Path, description = "Node UUID")
    ),
    responses(
        (status = 200, description = "Node deleted"),
        (status = 404, description = "Node not found")
    )
)]
async fn delete_node() {}

#[utoipa::path(
    post,
    path = "/api/v1/nodes/{id}/comments",
    tag = "comments",
    params(
        ("id" = String, Path, description = "Node UUID")
    ),
    responses(
        (status = 200, description = "Comment created", body = NodeCommentResponse),
        (status = 404, description = "Node not found")
    )
)]
async fn create_node_comment() {}

#[utoipa::path(
    get,
    path = "/api/v1/nodes/{id}/comments",
    tag = "comments",
    params(
        ("id" = String, Path, description = "Node UUID"),
        ("include_resolved" = Option<bool>, Query, description = "Include resolved comments")
    ),
    responses(
        (status = 200, description = "Node comments", body = Vec<NodeCommentResponse>),
        (status = 404, description = "Node not found")
    )
)]
async fn list_node_comments() {}

#[utoipa::path(
    put,
    path = "/api/v1/nodes/{id}/comments/{comment_id}/resolve",
    tag = "comments",
    params(
        ("id" = String, Path, description = "Node UUID"),
        ("comment_id" = String, Path, description = "Comment UUID")
    ),
    responses(
        (status = 200, description = "Comment resolved", body = NodeCommentResponse),
        (status = 404, description = "Comment not found")
    )
)]
async fn resolve_node_comment() {}

#[utoipa::path(
    delete,
    path = "/api/v1/nodes/{id}/comments/{comment_id}",
    tag = "comments",
    params(
        ("id" = String, Path, description = "Node UUID"),
        ("comment_id" = String, Path, description = "Comment UUID")
    ),
    responses(
        (status = 200, description = "Comment deleted", body = NodeCommentResponse),
        (status = 404, description = "Comment not found")
    )
)]
async fn delete_node_comment() {}

#[utoipa::path(
    post,
    path = "/api/v1/recall",
    tag = "recall",
    request_body = RecallRequest,
    responses(
        (status = 200, description = "Recall results", body = Vec<RecallResult>)
    )
)]
async fn recall() {}

#[utoipa::path(
    get,
    path = "/api/v1/search",
    tag = "recall",
    params(
        ("q" = String, Query, description = "Search query"),
        ("limit" = Option<usize>, Query, description = "Max results"),
        ("type" = Option<String>, Query, description = "Search type: fulltext, vector, hybrid")
    ),
    responses(
        (status = 200, description = "Search results")
    )
)]
async fn search() {}

#[utoipa::path(
    post,
    path = "/api/v1/graph/relationships",
    tag = "graph",
    request_body = RelationshipRequest,
    responses(
        (status = 201, description = "Relationship created")
    )
)]
async fn add_relationship() {}

#[utoipa::path(
    get,
    path = "/api/v1/nodes/{id}/backlinks",
    tag = "graph",
    params(
        ("id" = String, Path, description = "Node UUID"),
        ("limit" = Option<usize>, Query, description = "Page size"),
        ("offset" = Option<usize>, Query, description = "Pagination offset"),
        ("include_auto" = Option<bool>, Query, description = "Include auto-managed backlinks"),
        ("include_manual" = Option<bool>, Query, description = "Include manual backlinks"),
        ("source" = Option<String>, Query, description = "Filter auto backlink source (e.g. wikilink, mention)")
    ),
    responses(
        (status = 200, description = "Backlinks for the node")
    )
)]
async fn get_node_backlinks() {}

#[utoipa::path(
    get,
    path = "/api/v1/graph/neighbors/{id}",
    tag = "graph",
    params(
        ("id" = String, Path, description = "Node UUID"),
        ("depth" = Option<usize>, Query, description = "Traversal depth")
    ),
    responses(
        (status = 200, description = "Neighbor node IDs")
    )
)]
async fn get_neighbors() {}

#[utoipa::path(get, path = "/api/v1/daily-notes", tag = "daily-notes", responses((status = 200)))]
async fn list_daily_notes() {}

#[utoipa::path(post, path = "/api/v1/daily-notes/ensure", tag = "daily-notes", responses((status = 200)))]
async fn ensure_daily_note() {}

#[utoipa::path(get, path = "/api/v1/calendar/items", tag = "calendar", responses((status = 200)))]
async fn list_calendar_items() {}

#[utoipa::path(get, path = "/api/v1/calendar/ical", tag = "calendar", responses((status = 200)))]
async fn export_calendar_ical() {}

#[utoipa::path(post, path = "/api/v1/calendar/ical/import", tag = "calendar", responses((status = 200)))]
async fn import_calendar_ical() {}

#[utoipa::path(get, path = "/api/v1/calendar/google/status", tag = "calendar", responses((status = 200)))]
async fn google_calendar_status() {}

#[utoipa::path(post, path = "/api/v1/calendar/google/sync", tag = "calendar", responses((status = 200)))]
async fn google_calendar_sync() {}

#[utoipa::path(get, path = "/api/v1/ai/health", tag = "ai", responses((status = 200)))]
async fn ai_health() {}

#[utoipa::path(get, path = "/api/v1/ai/models", tag = "ai", responses((status = 200)))]
async fn ai_models() {}

#[utoipa::path(post, path = "/api/v1/ai/embeddings", tag = "ai", responses((status = 200)))]
async fn ai_embeddings() {}

#[utoipa::path(post, path = "/api/v1/ai/chat/completions", tag = "ai", responses((status = 200)))]
async fn ai_chat_completions() {}

#[utoipa::path(post, path = "/api/v1/shares", tag = "sharing", responses((status = 200)))]
async fn create_public_share() {}

#[utoipa::path(get, path = "/api/v1/shares", tag = "sharing", responses((status = 200)))]
async fn list_public_shares() {}

#[utoipa::path(delete, path = "/api/v1/shares/{id}", tag = "sharing", params(("id" = String, Path)), responses((status = 200)))]
async fn revoke_public_share() {}

#[utoipa::path(get, path = "/public/shares/{token}", tag = "sharing", params(("token" = String, Path)), responses((status = 200), (status = 404)))]
async fn get_public_share() {}

#[utoipa::path(
    get,
    path = "/api/v1/mcp/connectors",
    tag = "mcp-marketplace",
    params(
        ("publisher" = Option<String>, Query, description = "Filter by publisher"),
        ("verified" = Option<bool>, Query, description = "Filter by verification"),
        ("limit" = Option<usize>, Query, description = "Page size"),
        ("offset" = Option<usize>, Query, description = "Pagination offset")
    ),
    responses((status = 200, description = "MCP connectors", body = Vec<McpConnectorResponse>))
)]
async fn list_mcp_connectors() {}

#[utoipa::path(
    get,
    path = "/api/v1/mcp/connectors/{id}",
    tag = "mcp-marketplace",
    params(("id" = String, Path, description = "Connector UUID")),
    responses(
        (status = 200, description = "Connector found", body = McpConnectorResponse),
        (status = 404, description = "Connector not found")
    )
)]
async fn get_mcp_connector() {}

#[utoipa::path(
    post,
    path = "/api/v1/mcp/connectors",
    tag = "mcp-marketplace",
    request_body = CreateMcpConnectorRequest,
    responses((status = 200, description = "Connector created", body = McpConnectorResponse))
)]
async fn create_mcp_connector() {}

#[utoipa::path(
    put,
    path = "/api/v1/mcp/connectors/{id}",
    tag = "mcp-marketplace",
    params(("id" = String, Path, description = "Connector UUID")),
    request_body = UpdateMcpConnectorRequest,
    responses(
        (status = 200, description = "Connector updated", body = McpConnectorResponse),
        (status = 404, description = "Connector not found")
    )
)]
async fn update_mcp_connector() {}

#[utoipa::path(
    delete,
    path = "/api/v1/mcp/connectors/{id}",
    tag = "mcp-marketplace",
    params(("id" = String, Path, description = "Connector UUID")),
    responses((status = 204, description = "Connector deleted"), (status = 404, description = "Connector not found"))
)]
async fn delete_mcp_connector() {}

#[utoipa::path(get, path = "/api/v1/tasks/due", tag = "tasks", responses((status = 200)))]
async fn list_due_tasks() {}

#[utoipa::path(post, path = "/api/v1/tasks/prioritize", tag = "tasks", responses((status = 200)))]
async fn prioritize_tasks() {}

#[utoipa::path(post, path = "/api/v1/tasks/{id}/complete", tag = "tasks", responses((status = 200)))]
async fn complete_task() {}

#[utoipa::path(post, path = "/api/v1/tasks/{id}/reopen", tag = "tasks", responses((status = 200)))]
async fn reopen_task() {}

#[utoipa::path(post, path = "/api/v1/tasks/{id}/snooze", tag = "tasks", responses((status = 200)))]
async fn snooze_task_reminder() {}

#[utoipa::path(get, path = "/api/v1/templates", tag = "templates", responses((status = 200)))]
async fn list_templates() {}

#[utoipa::path(post, path = "/api/v1/templates", tag = "templates", responses((status = 201)))]
async fn create_template() {}

#[utoipa::path(patch, path = "/api/v1/templates/{id}", tag = "templates", responses((status = 200)))]
async fn update_template() {}

#[utoipa::path(delete, path = "/api/v1/templates/{id}", tag = "templates", responses((status = 200)))]
async fn delete_template() {}

#[utoipa::path(post, path = "/api/v1/templates/{id}/instantiate", tag = "templates", responses((status = 201)))]
async fn instantiate_template() {}

#[utoipa::path(post, path = "/api/v1/templates/{id}/apply", tag = "templates", responses((status = 200)))]
async fn apply_template() {}

#[utoipa::path(post, path = "/api/v1/files/upload", tag = "files", responses((status = 201)))]
async fn upload_file() {}

#[utoipa::path(
    get,
    path = "/api/v1/files",
    tag = "files",
    params(
        ("q" = Option<String>, Query, description = "Search text filter"),
        ("status" = Option<String>, Query, description = "Filter by extraction status"),
        ("failed_only" = Option<bool>, Query, description = "Only include failed extractions"),
        ("limit" = Option<usize>, Query, description = "Page size"),
        ("offset" = Option<usize>, Query, description = "Pagination offset"),
        ("sort" = Option<String>, Query, description = "Sort key (uploaded_desc/uploaded_asc/name_asc/name_desc/status_asc/status_desc)"),
        ("namespace" = Option<String>, Query, description = "Namespace filter"),
        ("kind" = Option<String>, Query, description = "Node kind filter")
    ),
    responses((status = 200))
)]
async fn list_attachments_index() {}

#[utoipa::path(get, path = "/api/v1/files/{node_id}", tag = "files", responses((status = 200)))]
async fn list_node_attachments() {}

#[utoipa::path(get, path = "/api/v1/files/{node_id}/{attachment_id}/chunks", tag = "files", responses((status = 200)))]
async fn get_attachment_chunks() {}

#[utoipa::path(post, path = "/api/v1/files/{node_id}/{attachment_id}/reindex", tag = "files", responses((status = 200)))]
async fn reindex_attachment() {}

#[utoipa::path(
    get,
    path = "/api/v1/files/{node_id}/{attachment_id}",
    tag = "files",
    params(
        ("node_id" = String, Path, description = "Node id"),
        ("attachment_id" = String, Path, description = "Attachment id"),
        ("inline" = Option<bool>, Query, description = "When true, return inline Content-Disposition")
    ),
    responses((status = 200))
)]
async fn download_attachment() {}

#[utoipa::path(delete, path = "/api/v1/files/{node_id}/{attachment_id}", tag = "files", responses((status = 200)))]
async fn delete_attachment() {}

#[utoipa::path(post, path = "/api/v1/voice/upload", tag = "voice", responses((status = 201)))]
async fn upload_voice_note() {}

#[utoipa::path(post, path = "/api/v1/assist/completion", tag = "assist", responses((status = 200)))]
async fn assist_completion() {}

#[utoipa::path(post, path = "/api/v1/assist/autocomplete", tag = "assist", responses((status = 200)))]
async fn assist_autocomplete() {}

#[utoipa::path(post, path = "/api/v1/assist/links", tag = "assist", responses((status = 200)))]
async fn assist_links() {}

#[utoipa::path(post, path = "/api/v1/assist/transform", tag = "assist", responses((status = 200)))]
async fn assist_transform() {}

#[utoipa::path(get, path = "/api/v1/export", tag = "export-import", responses((status = 200)))]
async fn export_bundle() {}

#[utoipa::path(post, path = "/api/v1/import", tag = "export-import", responses((status = 200)))]
async fn import_bundle() {}

#[utoipa::path(get, path = "/api/v1/audit", tag = "audit", responses((status = 200, body = Vec<AuditEntry>)))]
async fn list_audit_logs() {}

#[utoipa::path(get, path = "/api/v1/search/saved", tag = "saved-searches", responses((status = 200)))]
async fn list_saved_searches() {}

#[utoipa::path(post, path = "/api/v1/search/saved", tag = "saved-searches", responses((status = 201)))]
async fn create_saved_search() {}

#[utoipa::path(put, path = "/api/v1/search/saved/{id}", tag = "saved-searches", responses((status = 200)))]
async fn update_saved_search() {}

#[utoipa::path(delete, path = "/api/v1/search/saved/{id}", tag = "saved-searches", responses((status = 200)))]
async fn delete_saved_search() {}

#[utoipa::path(post, path = "/api/v1/search/saved/{id}/run", tag = "saved-searches", responses((status = 200)))]
async fn run_saved_search() {}

#[utoipa::path(get, path = "/api/v1/saved_views", tag = "saved-views", responses((status = 200)))]
async fn list_saved_views() {}

#[utoipa::path(post, path = "/api/v1/saved_views", tag = "saved-views", responses((status = 201)))]
async fn create_saved_view() {}

#[utoipa::path(patch, path = "/api/v1/saved_views/{id}", tag = "saved-views", responses((status = 200)))]
async fn update_saved_view() {}

#[utoipa::path(delete, path = "/api/v1/saved_views/{id}", tag = "saved-views", responses((status = 200)))]
async fn delete_saved_view() {}

#[utoipa::path(get, path = "/api/v1/permission-templates", tag = "permissions", responses((status = 200, body = Vec<PermissionTemplateResponse>)))]
async fn list_permission_templates() {}

#[utoipa::path(post, path = "/api/v1/permission-templates", tag = "permissions", responses((status = 201, body = PermissionTemplateResponse)))]
async fn create_permission_template() {}

#[utoipa::path(put, path = "/api/v1/permission-templates/{id}", tag = "permissions", responses((status = 200, body = PermissionTemplateResponse)))]
async fn update_permission_template() {}

#[utoipa::path(delete, path = "/api/v1/permission-templates/{id}", tag = "permissions", responses((status = 204)))]
async fn delete_permission_template() {}

#[utoipa::path(get, path = "/api/v1/access-keys", tag = "permissions", responses((status = 200, body = Vec<AccessKeyResponse>)))]
async fn list_access_keys() {}

#[utoipa::path(post, path = "/api/v1/access-keys", tag = "permissions", responses((status = 201, body = AccessKeyCreateResponse)))]
async fn create_access_key() {}

#[utoipa::path(delete, path = "/api/v1/access-keys/{id}", tag = "permissions", responses((status = 204)))]
async fn revoke_access_key() {}

#[utoipa::path(get, path = "/api/v1/oauth/clients", tag = "oauth", responses((status = 200, body = Vec<OAuthClientResponse>)))]
async fn list_oauth_clients() {}

#[utoipa::path(post, path = "/api/v1/oauth/clients", tag = "oauth", responses((status = 201, body = OAuthClientCreateResponse)))]
async fn create_oauth_client() {}

#[utoipa::path(delete, path = "/api/v1/oauth/clients/{id}", tag = "oauth", responses((status = 204)))]
async fn revoke_oauth_client() {}

#[utoipa::path(post, path = "/api/v1/oauth/token", tag = "oauth", responses((status = 200, body = OAuthTokenResponse)))]
async fn oauth_token() {}

#[utoipa::path(
    get,
    path = "/api/v1/profile",
    tag = "profile",
    responses(
        (status = 200, description = "Owner profile", body = ProfileResponse)
    )
)]
async fn get_profile() {}

#[utoipa::path(
    put,
    path = "/api/v1/profile",
    tag = "profile",
    request_body = UpdateProfileRequest,
    responses(
        (status = 200, description = "Updated owner profile", body = ProfileResponse)
    )
)]
async fn update_profile() {}

#[utoipa::path(get, path = "/api/v1/exchange/proposals", tag = "exchange", responses((status = 200)))]
async fn list_proposals() {}

#[utoipa::path(post, path = "/api/v1/exchange/proposals", tag = "exchange", responses((status = 201)))]
async fn submit_proposal() {}

#[utoipa::path(get, path = "/api/v1/exchange/proposals/{id}", tag = "exchange", responses((status = 200)))]
async fn get_proposal() {}

#[utoipa::path(post, path = "/api/v1/exchange/proposals/{id}/approve", tag = "exchange", responses((status = 200)))]
async fn approve_proposal() {}

#[utoipa::path(post, path = "/api/v1/exchange/proposals/{id}/reject", tag = "exchange", responses((status = 200)))]
async fn reject_proposal() {}

#[utoipa::path(get, path = "/api/v1/exchange/inbox/count", tag = "exchange", responses((status = 200)))]
async fn inbox_count() {}

#[utoipa::path(post, path = "/api/v1/exchange/proposals/batch", tag = "exchange", responses((status = 200)))]
async fn batch_proposals() {}

#[utoipa::path(get, path = "/api/v1/autonomy/rules", tag = "autonomy", responses((status = 200)))]
async fn list_autonomy_rules() {}

#[utoipa::path(post, path = "/api/v1/autonomy/rules", tag = "autonomy", responses((status = 201)))]
async fn create_autonomy_rule() {}

#[utoipa::path(get, path = "/api/v1/autonomy/rules/{id}", tag = "autonomy", responses((status = 200)))]
async fn get_autonomy_rule() {}

#[utoipa::path(put, path = "/api/v1/autonomy/rules/{id}", tag = "autonomy", responses((status = 200)))]
async fn update_autonomy_rule() {}

#[utoipa::path(delete, path = "/api/v1/autonomy/rules/{id}", tag = "autonomy", responses((status = 204)))]
async fn delete_autonomy_rule() {}

#[utoipa::path(get, path = "/api/v1/autonomy/action-log", tag = "autonomy", responses((status = 200)))]
async fn list_autonomy_action_log() {}

#[utoipa::path(post, path = "/api/v1/autonomy/evaluate", tag = "autonomy", responses((status = 200)))]
async fn evaluate_autonomy() {}

// --- Keychain ---
#[utoipa::path(post, path = "/api/v1/keychain/init", tag = "keychain", responses((status = 200, description = "Vault initialized")))]
async fn keychain_init() {}
#[utoipa::path(post, path = "/api/v1/keychain/unseal", tag = "keychain", responses((status = 200, description = "Vault unsealed")))]
async fn keychain_unseal() {}
#[utoipa::path(post, path = "/api/v1/keychain/seal", tag = "keychain", responses((status = 200, description = "Vault sealed")))]
async fn keychain_seal() {}
#[utoipa::path(get, path = "/api/v1/keychain/status", tag = "keychain", responses((status = 200, description = "Vault status")))]
async fn keychain_status() {}
#[utoipa::path(post, path = "/api/v1/keychain/rotate", tag = "keychain", responses((status = 200, description = "Key rotated")))]
async fn keychain_rotate() {}
#[utoipa::path(get, path = "/api/v1/keychain/epochs", tag = "keychain", responses((status = 200, description = "List key epochs")))]
async fn keychain_list_epochs() {}
#[utoipa::path(post, path = "/api/v1/keychain/credentials", tag = "keychain", responses((status = 201, description = "Credential stored")))]
async fn keychain_store_credential() {}
#[utoipa::path(get, path = "/api/v1/keychain/credentials", tag = "keychain", responses((status = 200, description = "List credentials")))]
async fn keychain_list_credentials() {}
#[utoipa::path(get, path = "/api/v1/keychain/credentials/{id}", tag = "keychain", params(("id" = String, Path)), responses((status = 200, description = "Credential value")))]
async fn keychain_read_credential() {}
#[utoipa::path(put, path = "/api/v1/keychain/credentials/{id}", tag = "keychain", params(("id" = String, Path)), responses((status = 200, description = "Credential updated")))]
async fn keychain_update_credential() {}
#[utoipa::path(delete, path = "/api/v1/keychain/credentials/{id}", tag = "keychain", params(("id" = String, Path)), responses((status = 204, description = "Credential destroyed")))]
async fn keychain_destroy_credential() {}
#[utoipa::path(post, path = "/api/v1/keychain/domains", tag = "keychain", responses((status = 201, description = "Domain created")))]
async fn keychain_create_domain() {}
#[utoipa::path(get, path = "/api/v1/keychain/domains", tag = "keychain", responses((status = 200, description = "List domains")))]
async fn keychain_list_domains() {}
#[utoipa::path(post, path = "/api/v1/keychain/delegations", tag = "keychain", responses((status = 201, description = "Delegation created")))]
async fn keychain_create_delegation() {}
#[utoipa::path(get, path = "/api/v1/keychain/delegations", tag = "keychain", responses((status = 200, description = "List delegations")))]
async fn keychain_list_delegations() {}
#[utoipa::path(delete, path = "/api/v1/keychain/delegations/{id}", tag = "keychain", params(("id" = String, Path)), responses((status = 204, description = "Delegation revoked")))]
async fn keychain_revoke_delegation() {}
#[utoipa::path(get, path = "/api/v1/keychain/audit", tag = "keychain", responses((status = 200, description = "Audit entries")))]
async fn keychain_list_audit() {}
#[utoipa::path(post, path = "/api/v1/keychain/backup", tag = "keychain", responses((status = 200, description = "Vault backup")))]
async fn keychain_backup() {}
#[utoipa::path(post, path = "/api/v1/keychain/restore", tag = "keychain", responses((status = 200, description = "Vault restored")))]
async fn keychain_restore() {}

// --- Relay ---
#[utoipa::path(get, path = "/api/v1/relay/contacts", tag = "relay", responses((status = 200)))]
async fn relay_list_contacts() {}
#[utoipa::path(post, path = "/api/v1/relay/contacts", tag = "relay", responses((status = 201)))]
async fn relay_create_contact() {}
#[utoipa::path(get, path = "/api/v1/relay/contacts/{id}", tag = "relay", params(("id" = String, Path)), responses((status = 200)))]
async fn relay_get_contact() {}
#[utoipa::path(put, path = "/api/v1/relay/contacts/{id}", tag = "relay", params(("id" = String, Path)), responses((status = 200)))]
async fn relay_update_contact() {}
#[utoipa::path(delete, path = "/api/v1/relay/contacts/{id}", tag = "relay", params(("id" = String, Path)), responses((status = 204)))]
async fn relay_delete_contact() {}
#[utoipa::path(get, path = "/api/v1/relay/channels", tag = "relay", responses((status = 200)))]
async fn relay_list_channels() {}
#[utoipa::path(post, path = "/api/v1/relay/channels", tag = "relay", responses((status = 201)))]
async fn relay_create_channel() {}
#[utoipa::path(delete, path = "/api/v1/relay/channels/{id}", tag = "relay", params(("id" = String, Path)), responses((status = 204)))]
async fn relay_delete_channel() {}
#[utoipa::path(get, path = "/api/v1/relay/channels/{id}/messages", tag = "relay", params(("id" = String, Path)), responses((status = 200)))]
async fn relay_list_messages() {}
#[utoipa::path(post, path = "/api/v1/relay/channels/{id}/messages", tag = "relay", params(("id" = String, Path)), responses((status = 201)))]
async fn relay_send_message() {}
#[utoipa::path(post, path = "/api/v1/relay/messages/{id}/read", tag = "relay", params(("id" = String, Path)), responses((status = 200)))]
async fn relay_mark_read() {}
#[utoipa::path(get, path = "/api/v1/relay/unread", tag = "relay", responses((status = 200)))]
async fn relay_unread_count() {}

// --- Safeguards ---
#[utoipa::path(get, path = "/api/v1/exchange/blocked-senders", tag = "safeguards", responses((status = 200)))]
async fn list_blocked_senders() {}
#[utoipa::path(post, path = "/api/v1/exchange/blocked-senders", tag = "safeguards", responses((status = 201)))]
async fn add_blocked_sender() {}
#[utoipa::path(delete, path = "/api/v1/exchange/blocked-senders/{id}", tag = "safeguards", params(("id" = String, Path)), responses((status = 204)))]
async fn remove_blocked_sender() {}
#[utoipa::path(get, path = "/api/v1/exchange/auto-approve-rules", tag = "safeguards", responses((status = 200)))]
async fn list_auto_approve_rules() {}
#[utoipa::path(post, path = "/api/v1/exchange/auto-approve-rules", tag = "safeguards", responses((status = 201)))]
async fn add_auto_approve_rule() {}
#[utoipa::path(put, path = "/api/v1/exchange/auto-approve-rules/{id}", tag = "safeguards", params(("id" = String, Path)), responses((status = 200)))]
async fn update_auto_approve_rule() {}
#[utoipa::path(delete, path = "/api/v1/exchange/auto-approve-rules/{id}", tag = "safeguards", params(("id" = String, Path)), responses((status = 204)))]
async fn remove_auto_approve_rule() {}
#[utoipa::path(post, path = "/api/v1/exchange/proposals/{id}/undo", tag = "safeguards", params(("id" = String, Path)), responses((status = 200, description = "Proposal undone")))]
async fn undo_proposal() {}

// --- Plugins ---
#[utoipa::path(get, path = "/api/v1/plugins", tag = "plugins", responses((status = 200, description = "List installed plugins")))]
async fn list_plugins_api() {}
#[utoipa::path(post, path = "/api/v1/plugins", tag = "plugins", responses((status = 201, description = "Plugin installed")))]
async fn install_plugin_api() {}
#[utoipa::path(get, path = "/api/v1/plugins/hooks", tag = "plugins", responses((status = 200, description = "Available hook points")))]
async fn list_hook_points_api() {}
#[utoipa::path(post, path = "/api/v1/plugins/reload", tag = "plugins", responses((status = 200, description = "Plugins reloaded")))]
async fn reload_plugins_api() {}
#[utoipa::path(delete, path = "/api/v1/plugins/{name}", tag = "plugins", params(("name" = String, Path)), responses((status = 204, description = "Plugin uninstalled")))]
async fn uninstall_plugin_api() {}

#[utoipa::path(get, path = "/api/v1/plugins/runtime", tag = "plugins", responses((status = 200, description = "List runtime plugins")))]
async fn plugins_runtime_list() {}
#[utoipa::path(get, path = "/api/v1/plugins/runtime/{name}", tag = "plugins", params(("name" = String, Path)), responses((status = 200, description = "Runtime plugin details")))]
async fn plugins_runtime_get() {}
#[utoipa::path(post, path = "/api/v1/plugins/runtime/{name}/reload", tag = "plugins", params(("name" = String, Path)), responses((status = 200, description = "Runtime plugin reloaded")))]
async fn plugins_runtime_reload() {}
#[utoipa::path(delete, path = "/api/v1/plugins/runtime/{name}", tag = "plugins", params(("name" = String, Path)), responses((status = 200, description = "Runtime plugin unloaded")))]
async fn plugins_runtime_unload() {}
#[utoipa::path(get, path = "/api/v1/plugins/runtime/{name}/hooks", tag = "plugins", params(("name" = String, Path)), responses((status = 200, description = "Runtime plugin hooks")))]
async fn plugins_runtime_hooks() {}

// --- Sync ---
#[utoipa::path(post, path = "/api/v1/sync/export", tag = "sync", responses((status = 200, description = "Sync export bundle")))]
async fn sync_export() {}
#[utoipa::path(post, path = "/api/v1/sync/import", tag = "sync", responses((status = 200, description = "Sync import result")))]
async fn sync_import() {}
#[utoipa::path(get, path = "/api/v1/sync/status", tag = "sync", responses((status = 200, description = "Sync status")))]
async fn sync_status() {}

// --- Federation ---
#[utoipa::path(get, path = "/api/v1/federation/peers", tag = "federation", responses((status = 200)))]
async fn federation_list_peers() {}
#[utoipa::path(post, path = "/api/v1/federation/peers", tag = "federation", responses((status = 201)))]
async fn federation_add_peer() {}
#[utoipa::path(delete, path = "/api/v1/federation/peers/{id}", tag = "federation", params(("id" = String, Path)), responses((status = 204)))]
async fn federation_remove_peer() {}
#[utoipa::path(get, path = "/api/v1/federation/peers/{id}/health", tag = "federation", params(("id" = String, Path)), responses((status = 200)))]
async fn federation_peer_health() {}
#[utoipa::path(get, path = "/api/v1/federation/identity", tag = "federation", responses((status = 200, body = FederationIdentityResponse)))]
async fn federation_identity() {}
#[utoipa::path(post, path = "/api/v1/federation/query", tag = "federation", responses((status = 200, description = "Federated query results")))]
async fn federation_query() {}

// --- Adapters ---
#[utoipa::path(get, path = "/api/v1/adapters", tag = "adapters", responses((status = 200)))]
async fn adapters_list() {}
#[utoipa::path(post, path = "/api/v1/adapters", tag = "adapters", responses((status = 201)))]
async fn adapters_register() {}
#[utoipa::path(get, path = "/api/v1/adapters/statuses", tag = "adapters", responses((status = 200)))]
async fn adapters_statuses() {}
#[utoipa::path(get, path = "/api/v1/adapters/{id}", tag = "adapters", params(("id" = String, Path)), responses((status = 200)))]
async fn adapter_get_status() {}
#[utoipa::path(delete, path = "/api/v1/adapters/{id}", tag = "adapters", params(("id" = String, Path)), responses((status = 204)))]
async fn adapter_remove() {}
#[utoipa::path(post, path = "/api/v1/adapters/{id}/send", tag = "adapters", params(("id" = String, Path)), responses((status = 200)))]
async fn adapter_send() {}
#[utoipa::path(post, path = "/api/v1/adapters/{id}/health", tag = "adapters", params(("id" = String, Path)), responses((status = 200)))]
async fn adapter_health() {}

// --- Agent ---
#[utoipa::path(get, path = "/api/v1/agent/context", tag = "agent", responses((status = 200)))]
async fn agent_context() {}
#[utoipa::path(get, path = "/api/v1/agent/chronicle", tag = "agent", responses((status = 200)))]
async fn agent_chronicle() {}
#[utoipa::path(get, path = "/api/v1/agent/intents", tag = "agent", responses((status = 200)))]
async fn agent_list_intents() {}
#[utoipa::path(post, path = "/api/v1/agent/intents/{id}/apply", tag = "agent", params(("id" = String, Path)), responses((status = 200)))]
async fn agent_apply_intent() {}
#[utoipa::path(post, path = "/api/v1/agent/intents/{id}/dismiss", tag = "agent", params(("id" = String, Path)), responses((status = 200)))]
async fn agent_dismiss_intent() {}
#[utoipa::path(get, path = "/api/v1/agent/models", tag = "agent", responses((status = 200)))]
async fn agent_list_models() {}
#[utoipa::path(get, path = "/api/v1/agent/watcher/status", tag = "agent", responses((status = 200)))]
async fn agent_watcher_status() {}
#[utoipa::path(get, path = "/api/v1/agent/insights", tag = "agent", responses((status = 200)))]
async fn agent_list_insights() {}
#[utoipa::path(post, path = "/api/v1/agent/feedback", tag = "agent", responses((status = 201, description = "Feedback recorded")))]
async fn agent_record_feedback() {}
#[utoipa::path(get, path = "/api/v1/agent/feedback", tag = "agent", responses((status = 200)))]
async fn agent_list_feedback() {}
#[utoipa::path(get, path = "/api/v1/agent/reflection/stats", tag = "agent", responses((status = 200)))]
async fn agent_reflection_stats() {}

// --- Proactive ---
#[utoipa::path(get, path = "/api/v1/proactive/insights", tag = "proactive", responses((status = 200)))]
async fn proactive_list_insights() {}
#[utoipa::path(post, path = "/api/v1/proactive/generate", tag = "proactive", responses((status = 200)))]
async fn proactive_generate() {}
#[utoipa::path(delete, path = "/api/v1/proactive/insights/{id}", tag = "proactive", params(("id" = String, Path)), responses((status = 204)))]
async fn proactive_delete_insight() {}

// --- Clips ---
#[utoipa::path(post, path = "/api/v1/clips/import", tag = "clips", responses((status = 201)))]
async fn clips_import() {}
#[utoipa::path(post, path = "/api/v1/clips/enrich", tag = "clips", responses((status = 200)))]
async fn clips_enrich() {}
#[utoipa::path(post, path = "/api/v1/clips/{id}/note", tag = "clips", params(("id" = String, Path)), responses((status = 201)))]
async fn clips_create_note() {}

// --- Graph extra ---
#[utoipa::path(delete, path = "/api/v1/graph/relationships/{id}", tag = "graph-extra", params(("id" = String, Path)), responses((status = 204)))]
async fn delete_relationship() {}

// --- Secrets ---
#[utoipa::path(post, path = "/api/v1/secrets", tag = "secrets", responses((status = 200, description = "Secret stored")))]
async fn set_secret() {}
#[utoipa::path(get, path = "/api/v1/secrets/status", tag = "secrets", responses((status = 200)))]
async fn secret_status() {}
#[utoipa::path(delete, path = "/api/v1/secrets/{key}", tag = "secrets", params(("key" = String, Path)), responses((status = 204)))]
async fn delete_secret() {}

// Schema types for OpenAPI
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub node_count: usize,
    pub version: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ProfileResponse {
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub bio: Option<String>,
    pub email: Option<String>,
    pub preferred_namespace: String,
    pub default_node_kind: String,
    pub preferred_llm_provider: Option<String>,
    pub timezone: String,
    pub signature_name: Option<String>,
    pub signature_public_key: Option<String>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Serialize, ToSchema)]
pub struct FederationIdentityResponse {
    pub vault_id: String,
    pub display_name: String,
    pub public_key: Option<String>,
    pub vault_address: Option<String>,
    pub updated_at: String,
}

#[derive(Deserialize, ToSchema)]
pub struct UpdateProfileRequest {
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub bio: Option<String>,
    pub email: Option<String>,
    pub preferred_namespace: Option<String>,
    pub default_node_kind: Option<String>,
    pub preferred_llm_provider: Option<String>,
    pub timezone: Option<String>,
    pub signature_name: Option<String>,
    pub signature_public_key: Option<String>,
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct KnowledgeNode {
    pub id: String,
    pub kind: NodeKind,
    pub title: Option<String>,
    pub content: String,
    pub namespace: String,
    pub tags: Vec<String>,
    pub importance: f64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub enum NodeKind {
    Fact,
    Task,
    Event,
    Decision,
    Preference,
    Entity,
    CodeSnippet,
    Project,
    Conversation,
    Procedure,
    Observation,
    Bookmark,
    Template,
    SavedView,
}

#[derive(Deserialize, ToSchema)]
pub struct StoreNodeRequest {
    pub kind: String,
    pub content: String,
    pub title: Option<String>,
    pub namespace: Option<String>,
    pub tags: Option<Vec<String>>,
    pub importance: Option<f64>,
}

#[derive(Deserialize, ToSchema)]
pub struct UpdateNodeRequest {
    pub kind: Option<String>,
    pub content: Option<String>,
    pub title: Option<String>,
    pub tags: Option<Vec<String>>,
    pub importance: Option<f64>,
}

#[derive(Deserialize, ToSchema)]
pub struct RecallRequest {
    pub text: String,
    pub strategy: Option<String>,
    pub limit: Option<usize>,
    pub min_score: Option<f64>,
    pub namespace: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct RecallResult {
    pub node: KnowledgeNode,
    pub score: f64,
}

#[derive(Deserialize, ToSchema)]
pub struct RelationshipRequest {
    pub source_id: String,
    pub target_id: String,
    pub kind: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct AuditEntry {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub subject: Option<String>,
    pub role: String,
    pub namespace: Option<String>,
    pub method: String,
    pub path: String,
    pub action: String,
    pub resource_id: Option<String>,
    pub status_code: u16,
    pub success: bool,
    pub latency_ms: u64,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct PermissionTemplateResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub tier: String,
    pub scope_namespace: Option<String>,
    pub scope_tags: Vec<String>,
    pub allow_kinds: Vec<String>,
    pub allow_actions: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct AccessKeyResponse {
    pub id: String,
    pub name: Option<String>,
    pub template_id: String,
    pub template_name: Option<String>,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub expires_at: Option<String>,
    pub revoked_at: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct AccessKeyCreateResponse {
    pub token: String,
    pub access_key: AccessKeyResponse,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OAuthClientCreateRequest {
    pub name: String,
    pub template_id: String,
    pub description: Option<String>,
    pub token_ttl_seconds: Option<u64>,
    pub expires_at: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OAuthClientResponse {
    pub client_id: String,
    pub name: String,
    pub template_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub last_used_at: Option<String>,
    pub expires_at: Option<String>,
    pub revoked_at: Option<String>,
    pub token_ttl_seconds: u64,
    pub description: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OAuthClientCreateResponse {
    pub client: OAuthClientResponse,
    pub client_secret: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OAuthTokenRequest {
    pub grant_type: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scope: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OAuthTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub scope: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub struct CreateNodeCommentRequest {
    pub body: String,
    pub author: Option<String>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct NodeCommentResponse {
    pub id: String,
    pub node_id: String,
    pub author: Option<String>,
    pub body: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize, ToSchema)]
pub struct CreateMcpConnectorRequest {
    pub name: String,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: String,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: Option<serde_json::Value>,
    pub capabilities: Option<Vec<String>>,
    pub verified: Option<bool>,
}

#[derive(Deserialize, ToSchema)]
pub struct UpdateMcpConnectorRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: Option<String>,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: Option<serde_json::Value>,
    pub capabilities: Option<Vec<String>>,
    pub verified: Option<bool>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct McpConnectorResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: String,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: serde_json::Value,
    pub capabilities: Vec<String>,
    pub verified: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Create the Swagger UI router.
pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/api/docs").url("/api/openapi.json", ApiDoc::openapi())
}

/// Get the OpenAPI JSON spec.
pub fn openapi_json() -> String {
    ApiDoc::openapi().to_pretty_json().unwrap_or_default()
}
