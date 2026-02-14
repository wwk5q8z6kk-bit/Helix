use std::sync::Arc;

use hx_core::{KnowledgeNode, NodeKind, PermissionTemplate, PermissionTier, QueryFilters};
use hx_engine::engine::HelixEngine;

const DEFAULT_RESOURCE_LIMIT: usize = 1000;

#[derive(Debug, Clone)]
pub struct McpContext {
    scope: McpScope,
    template_name: Option<String>,
    key_id: Option<String>,
}

impl McpContext {
    pub async fn from_access_key(
        engine: &Arc<HelixEngine>,
        token: &str,
    ) -> Result<Self, String> {
        let (key, template) = engine
            .resolve_access_key(token)
            .await
            .map_err(|e| format!("access key lookup failed: {e}"))?
            .ok_or("access key not found or expired".to_string())?;

        let scope = McpScope::from_template(&template);
        let key_id = Some(key.id.to_string());
        let template_name = Some(format!("{}:{}", template.name, key.id));

        Ok(Self {
            scope,
            template_name,
            key_id,
        })
    }

    pub fn unscoped_read_only() -> Self {
        Self {
            scope: McpScope::read_only(),
            template_name: None,
            key_id: None,
        }
    }

    /// Test-only constructor that allows arbitrary scope configuration.
    #[cfg(test)]
    pub(crate) fn with_scope(scope: McpScope) -> Self {
        Self {
            scope,
            template_name: Some("test-template".into()),
            key_id: Some("test-key".into()),
        }
    }

    pub fn scope(&self) -> &McpScope {
        &self.scope
    }

    pub fn key_id(&self) -> Option<&str> {
        self.key_id.as_deref()
    }

    pub fn can_read(&self) -> bool {
        true
    }

    pub fn can_write(&self) -> bool {
        self.scope.allow_write
    }

    pub fn summary(&self) -> String {
        match &self.template_name {
            Some(name) => format!("scoped({name})"),
            None => "unscoped(read-only)".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct McpScope {
    pub namespace: Option<String>,
    pub tags: Vec<String>,
    pub kinds: Vec<NodeKind>,
    pub allow_write: bool,
    pub allow_actions: Vec<String>,
    pub resource_limit: usize,
    pub tier: PermissionTier,
}

impl McpScope {
    pub fn from_template(template: &PermissionTemplate) -> Self {
        let allow_write = matches!(
            template.tier,
            PermissionTier::Edit | PermissionTier::Action | PermissionTier::Admin
        );

        Self {
            namespace: template.scope_namespace.clone(),
            tags: template.scope_tags.clone(),
            kinds: template.allow_kinds.clone(),
            allow_write,
            allow_actions: template.allow_actions.clone(),
            resource_limit: DEFAULT_RESOURCE_LIMIT,
            tier: template.tier,
        }
    }

    pub fn read_only() -> Self {
        Self {
            namespace: None,
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: false,
            allow_actions: Vec::new(),
            resource_limit: DEFAULT_RESOURCE_LIMIT,
            tier: PermissionTier::View,
        }
    }

    pub fn is_unscoped(&self) -> bool {
        self.namespace.is_none() && self.tags.is_empty() && self.kinds.is_empty()
    }

    pub fn is_admin(&self) -> bool {
        self.tier == PermissionTier::Admin
    }

    pub fn ensure_action(&self, action: &str) -> Result<(), String> {
        if self.allows_action(action) {
            Ok(())
        } else {
            Err(format!("action '{action}' not permitted by access scope"))
        }
    }

    pub fn apply_filters(&self, filters: &mut QueryFilters) -> Result<(), String> {
        if let Some(ns) = &self.namespace {
            if let Some(req) = &filters.namespace {
                if req != ns {
                    return Err(format!("namespace '{req}' not permitted"));
                }
            }
            filters.namespace = Some(ns.clone());
        }

        if !self.kinds.is_empty() {
            if let Some(req) = &filters.kinds {
                if !req.iter().all(|k| self.kinds.contains(k)) {
                    return Err("one or more kinds not permitted".into());
                }
            } else {
                filters.kinds = Some(self.kinds.clone());
            }
        }

        if !self.tags.is_empty() {
            if let Some(req) = &filters.tags {
                if !req.iter().all(|tag| self.contains_tag(tag)) {
                    return Err("one or more tags not permitted".into());
                }
            } else {
                filters.tags = Some(self.tags.clone());
            }
        }

        Ok(())
    }

    pub fn check_node(&self, node: &KnowledgeNode) -> Result<(), String> {
        if let Some(ns) = &self.namespace {
            if &node.namespace != ns {
                return Err("node not permitted by namespace scope".into());
            }
        }

        if !self.kinds.is_empty() && !self.kinds.contains(&node.kind) {
            return Err("node kind not permitted".into());
        }

        if !self.tags.is_empty()
            && !node.tags.iter().any(|tag| self.contains_tag(tag))
        {
            return Err("node tags not permitted".into());
        }

        Ok(())
    }

    pub fn normalize_namespace(&self, requested: Option<&str>) -> Result<Option<String>, String> {
        if let Some(ns) = &self.namespace {
            if let Some(req) = requested {
                if req != ns {
                    return Err(format!("namespace '{req}' not permitted"));
                }
            }
            return Ok(Some(ns.clone()));
        }

        Ok(requested.map(|s| s.to_string()))
    }

    pub fn ensure_kind(&self, kind: NodeKind) -> Result<(), String> {
        if self.kinds.is_empty() || self.kinds.contains(&kind) {
            Ok(())
        } else {
            Err(format!("kind '{}' not permitted", kind.as_str()))
        }
    }

    pub fn ensure_write_allowed(&self) -> Result<(), String> {
        if self.allow_write {
            Ok(())
        } else {
            Err("write/propose operations are not permitted".into())
        }
    }

    pub fn ensure_tags_for_proposal(&self, mut tags: Vec<String>) -> Vec<String> {
        if self.tags.is_empty() {
            return tags;
        }

        for scope_tag in &self.tags {
            if !tags.iter().any(|t| t.eq_ignore_ascii_case(scope_tag)) {
                tags.push(scope_tag.clone());
            }
        }

        tags
    }

    fn allows_action(&self, action: &str) -> bool {
        if self.allow_actions.is_empty() {
            return true;
        }

        self.allow_actions
            .iter()
            .any(|allowed| action_matches(allowed, action))
    }

    fn contains_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t.eq_ignore_ascii_case(tag))
    }
}

fn action_matches(allowed: &str, action: &str) -> bool {
    if allowed == "*" || allowed == "mcp:*" {
        return true;
    }
    if allowed == action {
        return true;
    }
    if let Some(prefix) = allowed.strip_suffix(".*") {
        return action.starts_with(prefix);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::{NodeKind, PermissionTemplate, PermissionTier, QueryFilters};

    fn scope_read_only() -> McpScope {
        McpScope::read_only()
    }

    fn scope_with_namespace(ns: &str) -> McpScope {
        McpScope {
            namespace: Some(ns.to_string()),
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: Vec::new(),
            resource_limit: 1000,
            tier: PermissionTier::Edit,
        }
    }

    fn scope_with_kinds(kinds: Vec<NodeKind>) -> McpScope {
        McpScope {
            namespace: None,
            tags: Vec::new(),
            kinds,
            allow_write: true,
            allow_actions: Vec::new(),
            resource_limit: 1000,
            tier: PermissionTier::Edit,
        }
    }

    fn scope_with_tags(tags: Vec<&str>) -> McpScope {
        McpScope {
            namespace: None,
            tags: tags.into_iter().map(String::from).collect(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: Vec::new(),
            resource_limit: 1000,
            tier: PermissionTier::Edit,
        }
    }

    fn scope_with_actions(actions: Vec<&str>) -> McpScope {
        McpScope {
            namespace: None,
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: actions.into_iter().map(String::from).collect(),
            resource_limit: 1000,
            tier: PermissionTier::Action,
        }
    }

    // --- read_only_scope_blocks_write ---

    #[test]
    fn read_only_scope_blocks_write() {
        let scope = scope_read_only();
        assert!(!scope.allow_write);
        assert!(scope.ensure_write_allowed().is_err());
    }

    // --- namespace_isolation ---

    #[test]
    fn namespace_isolation() {
        let scope = scope_with_namespace("private");
        let mut node = KnowledgeNode::new(NodeKind::Fact, "test");
        node.namespace = "default".to_string();
        assert!(scope.check_node(&node).is_err());
    }

    #[test]
    fn namespace_allows_matching() {
        let scope = scope_with_namespace("private");
        let mut node = KnowledgeNode::new(NodeKind::Fact, "test");
        node.namespace = "private".to_string();
        assert!(scope.check_node(&node).is_ok());
    }

    // --- kind_restriction ---

    #[test]
    fn kind_restriction_blocks_disallowed() {
        let scope = scope_with_kinds(vec![NodeKind::Fact]);
        let node = KnowledgeNode::new(NodeKind::Task, "test");
        assert!(scope.check_node(&node).is_err());
    }

    #[test]
    fn kind_restriction_allows_permitted() {
        let scope = scope_with_kinds(vec![NodeKind::Fact, NodeKind::Task]);
        let node = KnowledgeNode::new(NodeKind::Fact, "test");
        assert!(scope.check_node(&node).is_ok());
    }

    // --- tag_restriction ---

    #[test]
    fn tag_restriction_blocks_untagged() {
        let scope = scope_with_tags(vec!["project-a"]);
        let node = KnowledgeNode::new(NodeKind::Fact, "test");
        // node has no tags
        assert!(scope.check_node(&node).is_err());
    }

    #[test]
    fn tag_restriction_allows_matching() {
        let scope = scope_with_tags(vec!["project-a"]);
        let mut node = KnowledgeNode::new(NodeKind::Fact, "test");
        node.tags = vec!["project-a".to_string()];
        assert!(scope.check_node(&node).is_ok());
    }

    // --- apply_filters_enforces_namespace ---

    #[test]
    fn apply_filters_enforces_namespace() {
        let scope = scope_with_namespace("work");
        let mut filters = QueryFilters::default();
        filters.namespace = Some("personal".to_string());
        assert!(scope.apply_filters(&mut filters).is_err());
    }

    #[test]
    fn apply_filters_injects_namespace() {
        let scope = scope_with_namespace("work");
        let mut filters = QueryFilters::default();
        scope.apply_filters(&mut filters).unwrap();
        assert_eq!(filters.namespace, Some("work".to_string()));
    }

    // --- apply_filters_injects_kinds ---

    #[test]
    fn apply_filters_injects_kinds() {
        let scope = scope_with_kinds(vec![NodeKind::Fact, NodeKind::Task]);
        let mut filters = QueryFilters::default();
        scope.apply_filters(&mut filters).unwrap();
        assert_eq!(filters.kinds, Some(vec![NodeKind::Fact, NodeKind::Task]));
    }

    #[test]
    fn apply_filters_rejects_unpermitted_kinds() {
        let scope = scope_with_kinds(vec![NodeKind::Fact]);
        let mut filters = QueryFilters::default();
        filters.kinds = Some(vec![NodeKind::Task]);
        assert!(scope.apply_filters(&mut filters).is_err());
    }

    // --- ensure_action_blocks_unpermitted ---

    #[test]
    fn ensure_action_blocks_unpermitted() {
        let scope = scope_with_actions(vec!["mcp.read"]);
        assert!(scope.ensure_action("mcp.propose").is_err());
    }

    #[test]
    fn ensure_action_allows_permitted() {
        let scope = scope_with_actions(vec!["mcp.read", "mcp.propose"]);
        assert!(scope.ensure_action("mcp.read").is_ok());
        assert!(scope.ensure_action("mcp.propose").is_ok());
    }

    // --- action_wildcard ---

    #[test]
    fn action_wildcard_allows_all() {
        let scope = scope_with_actions(vec!["mcp:*"]);
        assert!(scope.ensure_action("mcp.read").is_ok());
        assert!(scope.ensure_action("mcp.propose").is_ok());
        assert!(scope.ensure_action("anything").is_ok());
    }

    #[test]
    fn action_star_allows_all() {
        let scope = scope_with_actions(vec!["*"]);
        assert!(scope.ensure_action("mcp.read").is_ok());
    }

    #[test]
    fn action_empty_allows_all() {
        // Empty allow_actions means "allow everything"
        let scope = McpScope {
            allow_actions: Vec::new(),
            ..scope_read_only()
        };
        assert!(scope.ensure_action("anything").is_ok());
    }

    // --- ensure_tags_injects_scope_tags ---

    #[test]
    fn ensure_tags_injects_scope_tags() {
        let scope = scope_with_tags(vec!["scope-tag"]);
        let tags = scope.ensure_tags_for_proposal(vec!["user-tag".to_string()]);
        assert!(tags.contains(&"user-tag".to_string()));
        assert!(tags.contains(&"scope-tag".to_string()));
    }

    #[test]
    fn ensure_tags_no_duplicates() {
        let scope = scope_with_tags(vec!["shared"]);
        let tags = scope.ensure_tags_for_proposal(vec!["shared".to_string()]);
        assert_eq!(tags.len(), 1);
    }

    // --- from_template_tiers ---

    #[test]
    fn from_template_view_tier_no_write() {
        let now = chrono::Utc::now();
        let template = PermissionTemplate {
            id: uuid::Uuid::now_v7(),
            name: "viewer".into(),
            description: None,
            tier: PermissionTier::View,
            scope_namespace: None,
            scope_tags: Vec::new(),
            allow_kinds: Vec::new(),
            allow_actions: Vec::new(),
            created_at: now,
            updated_at: now,
        };
        let scope = McpScope::from_template(&template);
        assert!(!scope.allow_write);
        assert_eq!(scope.tier, PermissionTier::View);
    }

    #[test]
    fn from_template_edit_tier_allows_write() {
        let now = chrono::Utc::now();
        let template = PermissionTemplate {
            id: uuid::Uuid::now_v7(),
            name: "editor".into(),
            description: None,
            tier: PermissionTier::Edit,
            scope_namespace: Some("work".into()),
            scope_tags: vec!["team".into()],
            allow_kinds: vec![NodeKind::Fact],
            allow_actions: vec!["mcp.read".into(), "mcp.propose".into()],
            created_at: now,
            updated_at: now,
        };
        let scope = McpScope::from_template(&template);
        assert!(scope.allow_write);
        assert_eq!(scope.namespace, Some("work".into()));
        assert_eq!(scope.tags, vec!["team".to_string()]);
        assert_eq!(scope.kinds, vec![NodeKind::Fact]);
        assert_eq!(scope.tier, PermissionTier::Edit);
    }

    #[test]
    fn from_template_admin_tier_allows_write() {
        let now = chrono::Utc::now();
        let template = PermissionTemplate {
            id: uuid::Uuid::now_v7(),
            name: "admin".into(),
            description: None,
            tier: PermissionTier::Admin,
            scope_namespace: None,
            scope_tags: Vec::new(),
            allow_kinds: Vec::new(),
            allow_actions: Vec::new(),
            created_at: now,
            updated_at: now,
        };
        let scope = McpScope::from_template(&template);
        assert!(scope.allow_write);
        assert!(scope.is_admin());
    }

    // --- action_matches helper ---

    #[test]
    fn action_matches_exact() {
        assert!(action_matches("mcp.read", "mcp.read"));
        assert!(!action_matches("mcp.read", "mcp.write"));
    }

    #[test]
    fn action_matches_prefix_wildcard() {
        assert!(action_matches("mcp.*", "mcp.read"));
        assert!(action_matches("mcp.*", "mcp.propose"));
    }

    // --- McpContext ---

    #[test]
    fn unscoped_read_only_context() {
        let ctx = McpContext::unscoped_read_only();
        assert!(ctx.can_read());
        assert!(!ctx.can_write());
        assert!(ctx.key_id().is_none());
        assert!(ctx.summary().contains("unscoped"));
    }

    // --- McpScope helpers ---

    #[test]
    fn is_unscoped_when_empty() {
        let scope = scope_read_only();
        assert!(scope.is_unscoped());
    }

    #[test]
    fn is_scoped_with_namespace() {
        let scope = scope_with_namespace("ns");
        assert!(!scope.is_unscoped());
    }

    #[test]
    fn ensure_kind_allows_when_empty() {
        let scope = scope_read_only();
        assert!(scope.ensure_kind(NodeKind::Task).is_ok());
    }

    #[test]
    fn ensure_kind_blocks_when_restricted() {
        let scope = scope_with_kinds(vec![NodeKind::Fact]);
        assert!(scope.ensure_kind(NodeKind::Task).is_err());
    }

    #[test]
    fn normalize_namespace_uses_scope() {
        let scope = scope_with_namespace("scoped");
        let result = scope.normalize_namespace(None).unwrap();
        assert_eq!(result, Some("scoped".to_string()));
    }

    #[test]
    fn normalize_namespace_rejects_mismatch() {
        let scope = scope_with_namespace("scoped");
        let result = scope.normalize_namespace(Some("other"));
        assert!(result.is_err());
    }

    #[test]
    fn normalize_namespace_allows_match() {
        let scope = scope_with_namespace("scoped");
        let result = scope.normalize_namespace(Some("scoped")).unwrap();
        assert_eq!(result, Some("scoped".to_string()));
    }

    #[test]
    fn normalize_namespace_passthrough_unscoped() {
        let scope = scope_read_only();
        let result = scope.normalize_namespace(Some("anything")).unwrap();
        assert_eq!(result, Some("anything".to_string()));
    }
}
