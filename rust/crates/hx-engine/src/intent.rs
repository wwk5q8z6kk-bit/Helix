use chrono::{Datelike, Utc, Weekday};
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};
use hx_core::*;
use hx_storage::unified::UnifiedStore;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use serde::Deserialize;

/// The IntentEngine analyzes knowledge nodes to suggest autonomous actions.
#[derive(Clone)]
pub struct IntentEngine {
    store: Arc<UnifiedStore>,
    llm: Option<Arc<dyn LlmProvider>>,
}

impl IntentEngine {
    pub fn new(store: Arc<UnifiedStore>) -> Self {
        Self { store, llm: None }
    }

    pub fn with_llm(mut self, llm: Option<Arc<dyn LlmProvider>>) -> Self {
        self.llm = llm;
        self
    }

    /// Extract possible intents from a node and store them.
    /// Applies confidence adjustments from learned feedback.
    pub async fn extract_intents_and_store(
        &self,
        node: &KnowledgeNode,
    ) -> MvResult<Vec<CapturedIntent>> {
        let mut detected = Vec::new();

        if let Some(intent) = self.detect_reminder_intent(node) {
            detected.push(intent);
        }

        if let Some(intent) = self.detect_task_intent(node) {
            detected.push(intent);
        }

        detected.extend(self.detect_link_intents(node));
        detected.extend(self.detect_tag_intents(node));

        let has_primary_intent = detected.iter().any(|intent| {
            matches!(intent.intent_type, IntentType::ScheduleReminder | IntentType::ExtractTask)
        });

        if !has_primary_intent {
            if let Some(llm) = self.llm.clone() {
                let llm_intents = self.detect_llm_intents(node, llm.as_ref()).await;
                detected.extend(llm_intents);
            }
        }

        // Apply learned confidence adjustments
        for intent in &mut detected {
            let type_str = intent.intent_type.to_string();
            if let Ok(Some(override_)) = self.store.nodes.get_confidence_override(&type_str).await {
                // Adjust confidence based on feedback history
                intent.confidence =
                    (intent.confidence + override_.base_adjustment).clamp(0.0, 1.0);

                // Suppress intents below the learned floor
                if intent.confidence < override_.suppress_below {
                    continue; // will be filtered out below
                }
            }
        }

        // Filter out suppressed intents (confidence set to 0 above)
        detected.retain(|i| i.confidence > 0.0);

        let existing_intents = self
            .store
            .nodes
            .list_intents(Some(node.id), Some(IntentStatus::Suggested), 200, 0)
            .await?;
        let mut seen_signatures: HashSet<String> = existing_intents
            .iter()
            .map(Self::intent_signature)
            .collect();

        let mut intents = Vec::new();
        for intent in detected {
            let signature = Self::intent_signature(&intent);
            if !seen_signatures.insert(signature) {
                continue;
            }
            self.store.nodes.log_intent(&intent).await?;
            intents.push(intent);
        }

        Ok(intents)
    }

    fn intent_signature(intent: &CapturedIntent) -> String {
        let params = serde_json::to_string(&intent.parameters).unwrap_or_default();
        format!("{}|{params}", intent.intent_type)
    }

    /// Detect reminder-related intents with rich date parsing
    fn detect_reminder_intent(&self, node: &KnowledgeNode) -> Option<CapturedIntent> {
        let content_lower = node.content.to_lowercase();

        let is_reminder = content_lower.contains("remind me")
            || content_lower.contains("don't forget")
            || content_lower.contains("dont forget")
            || content_lower.contains("reminder:");

        // Also trigger if time expressions are present with action verbs
        let has_time_expr = self.parse_relative_date(&content_lower).is_some();
        let has_action_verb = content_lower.contains("need to")
            || content_lower.contains("should")
            || content_lower.contains("call")
            || content_lower.contains("email")
            || content_lower.contains("send")
            || content_lower.contains("check")
            || content_lower.contains("review")
            || content_lower.contains("follow up");

        if !is_reminder && !(has_time_expr && has_action_verb) {
            return None;
        }

        let mut intent = CapturedIntent::new(node.id, IntentType::ScheduleReminder);

        // Adjust confidence based on keyword strength
        if content_lower.contains("remind me") || content_lower.contains("don't forget") {
            intent.confidence = 0.9;
        } else if has_time_expr && has_action_verb {
            intent.confidence = 0.8;
        } else if has_time_expr {
            intent.confidence = 0.7;
        } else {
            intent.confidence = 0.6;
        }

        // Extract date hints as parameters
        let mut params = serde_json::Map::new();
        if let Some(relative) = self.parse_relative_date(&content_lower) {
            params.insert("relative_time".to_string(), relative.into());
        }

        // Extract the reminder subject (what to be reminded about)
        if let Some(subject) = self.extract_reminder_subject(&content_lower) {
            params.insert("subject".to_string(), subject.into());
        }

        if !params.is_empty() {
            intent.parameters = serde_json::Value::Object(params);
        }

        Some(intent)
    }

    /// Detect task-related intents with priority and deadline extraction
    fn detect_task_intent(&self, node: &KnowledgeNode) -> Option<CapturedIntent> {
        let content = &node.content;
        let content_lower = content.to_lowercase();

        let is_task = content.contains("- [ ]")
            || content.contains("- []")
            || content_lower.contains("todo:")
            || content_lower.contains("task:")
            || content_lower.contains("need to ")
            || content_lower.contains("must ")
            || content_lower.contains("have to ")
            || content_lower.contains("action item")
            || content_lower.contains("action items:");

        if !is_task {
            return None;
        }

        let mut intent = CapturedIntent::new(node.id, IntentType::ExtractTask);

        // Checkbox syntax is highest confidence
        if content.contains("- [ ]") || content.contains("- []") {
            intent.confidence = 0.95;
        } else if content_lower.contains("todo:") || content_lower.contains("task:") {
            intent.confidence = 0.85;
        } else if content_lower.contains("need to") || content_lower.contains("must") {
            intent.confidence = 0.7;
        } else {
            intent.confidence = 0.6;
        }

        let mut params = serde_json::Map::new();

        // Extract priority
        if let Some((priority, label)) = self.detect_priority(&content_lower) {
            params.insert("priority".to_string(), serde_json::json!(priority));
            params.insert("priority_label".to_string(), label.into());
        }

        // Extract deadline
        if let Some(deadline) = self.detect_deadline(&content_lower) {
            params.insert("deadline_relative".to_string(), deadline.into());
        }

        // Extract dependency hints
        if let Some(dep) = self.detect_dependency(&content_lower) {
            params.insert("depends_on".to_string(), dep.into());
        }

        if !params.is_empty() {
            intent.parameters = serde_json::Value::Object(params);
        }

        Some(intent)
    }

    /// Detect wikilink and mention-style link intents
    fn detect_link_intents(&self, node: &KnowledgeNode) -> Vec<CapturedIntent> {
        let mut intents = Vec::new();

        // Detect [[wikilinks]] - simple bracket matching
        let mut i = 0;
        let chars: Vec<char> = node.content.chars().collect();
        while i < chars.len().saturating_sub(3) {
            if chars[i] == '[' && chars.get(i + 1) == Some(&'[') {
                let start = i + 2;
                let mut end = start;
                while end < chars.len().saturating_sub(1) {
                    if chars[end] == ']' && chars.get(end + 1) == Some(&']') {
                        let target: String = chars[start..end].iter().collect();
                        if !target.is_empty() {
                            let mut intent = CapturedIntent::new(node.id, IntentType::SuggestLink)
                                .with_confidence(0.85);

                            let mut params = serde_json::Map::new();
                            params.insert("target".to_string(), target.into());
                            params.insert("link_type".to_string(), serde_json::Value::String("wikilink".to_string()));
                            intent.parameters = serde_json::Value::Object(params);

                            intents.push(intent);
                        }
                        i = end + 2;
                        break;
                    }
                    end += 1;
                }
                if end >= chars.len().saturating_sub(1) {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        // Detect @mentions
        let words: Vec<&str> = node.content.split_whitespace().collect();
        for word in words {
            if word.starts_with('@') && word.len() > 1 {
                let mention = word[1..].trim_matches(|c: char| !c.is_alphanumeric());
                if !mention.is_empty() {
                    let mut intent = CapturedIntent::new(node.id, IntentType::LinkToProject)
                        .with_confidence(0.75);

                    let mut params = serde_json::Map::new();
                    params.insert("target".to_string(), mention.to_string().into());
                    params.insert("link_type".to_string(), serde_json::Value::String("mention".to_string()));
                    intent.parameters = serde_json::Value::Object(params);

                    intents.push(intent);
                }
            }
        }

        intents
    }

    /// Detect potential tag suggestions based on hashtags in content
    fn detect_tag_intents(&self, node: &KnowledgeNode) -> Vec<CapturedIntent> {
        let mut intents = Vec::new();

        let words: Vec<&str> = node.content.split_whitespace().collect();
        for word in words {
            if word.starts_with('#') && word.len() > 1 {
                let tag = word[1..]
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if tag.is_empty() {
                    continue;
                }

                if node.tags.iter().any(|t| t.to_lowercase() == tag) {
                    continue;
                }

                let mut intent =
                    CapturedIntent::new(node.id, IntentType::SuggestTag).with_confidence(0.8);

                let mut params = serde_json::Map::new();
                params.insert("tag".to_string(), tag.into());
                intent.parameters = serde_json::Value::Object(params);

                intents.push(intent);
            }
        }

        intents
    }

    /// Analyze a single node for intents without storing
    pub fn analyze_node(&self, node: &KnowledgeNode) -> Vec<CapturedIntent> {
        let mut intents = Vec::new();

        if let Some(intent) = self.detect_reminder_intent(node) {
            intents.push(intent);
        }
        if let Some(intent) = self.detect_task_intent(node) {
            intents.push(intent);
        }
        intents.extend(self.detect_link_intents(node));
        intents.extend(self.detect_tag_intents(node));

        intents
    }

    async fn detect_llm_intents(
        &self,
        node: &KnowledgeNode,
        llm: &dyn LlmProvider,
    ) -> Vec<CapturedIntent> {
        let content = node.content.trim();
        if content.len() < 20 {
            return Vec::new();
        }

        let truncated = if content.len() > 4000 {
            content.chars().take(4000).collect::<String>()
        } else {
            content.to_string()
        };

        let prompt = format!(
            "Extract intents from the note. Return JSON only in this schema:\n\
{{\"intents\":[{{\"type\":\"extract_task|schedule_reminder\",\"confidence\":0.0-1.0,\"parameters\":{{...}}}}]}}\n\
Use schedule_reminder parameters: relative_time (today, tomorrow, next_week, next_month, in_2_days, in_3_days, in_N_days) \
or reminder_at (RFC3339). Optionally include subject.\n\
Use extract_task parameters: task (short text), priority (low|medium|high), deadline_relative, deadline (RFC3339).\n\
If none, return {{\"intents\":[]}}.\n\nNote:\n{truncated}"
        );

        let messages = vec![
            ChatMessage::system(
                "You are a strict JSON generator for a personal knowledge system. \
                 Only output valid JSON. No extra text.",
            ),
            ChatMessage::user(prompt),
        ];

        let params = CompletionParams {
            max_tokens: Some(300),
            temperature: Some(0.0),
            ..CompletionParams::default()
        };

        let response = match tokio::time::timeout(Duration::from_secs(8), llm.complete(&messages, &params)).await {
            Ok(Ok(response)) => response,
            Ok(Err(err)) => {
                tracing::warn!(error = %err, node_id = %node.id, "LLM intent detection failed");
                return Vec::new();
            }
            Err(_) => {
                tracing::warn!(node_id = %node.id, "LLM intent detection timed out");
                return Vec::new();
            }
        };

        #[derive(Deserialize)]
        struct LlmIntentResponse {
            intents: Vec<LlmIntent>,
        }

        #[derive(Deserialize)]
        struct LlmIntent {
            #[serde(rename = "type")]
            intent_type: String,
            confidence: Option<f32>,
            parameters: Option<serde_json::Value>,
        }

        let trimmed = response.trim();
        let json_str = if trimmed.starts_with('{') {
            trimmed
        } else if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
            &trimmed[start..=end]
        } else {
            tracing::warn!(node_id = %node.id, "LLM intent detection returned non-JSON response");
            return Vec::new();
        };

        let parsed: LlmIntentResponse = match serde_json::from_str(json_str) {
            Ok(parsed) => parsed,
            Err(err) => {
                tracing::warn!(error = %err, node_id = %node.id, "Failed to parse LLM intent JSON");
                return Vec::new();
            }
        };

        let content_lower = content.to_lowercase();
        let mut intents = Vec::new();

        for candidate in parsed.intents {
            let intent_type: IntentType = match candidate.intent_type.parse() {
                Ok(intent_type) => intent_type,
                Err(_) => continue,
            };

            if !matches!(intent_type, IntentType::ExtractTask | IntentType::ScheduleReminder) {
                continue;
            }

            let mut intent = CapturedIntent::new(node.id, intent_type.clone());
            intent.confidence = candidate.confidence.unwrap_or(0.6).clamp(0.0, 1.0);

            let mut params = match candidate.parameters {
                Some(serde_json::Value::Object(map)) => map,
                _ => serde_json::Map::new(),
            };

            match intent_type {
                IntentType::ScheduleReminder => {
                    if !params.contains_key("relative_time") && !params.contains_key("reminder_at") {
                        if let Some(relative) = self.parse_relative_date(&content_lower) {
                            params.insert("relative_time".to_string(), relative.into());
                        }
                    }
                    if !params.contains_key("subject") {
                        if let Some(subject) = self.extract_reminder_subject(&content_lower) {
                            params.insert("subject".to_string(), subject.into());
                        }
                    }
                }
                IntentType::ExtractTask => {
                    if !params.contains_key("priority") {
                        if let Some((priority, label)) = self.detect_priority(&content_lower) {
                            params.insert("priority".to_string(), serde_json::json!(priority));
                            params.insert("priority_label".to_string(), label.into());
                        }
                    }
                    if !params.contains_key("deadline_relative") && !params.contains_key("deadline") {
                        if let Some(deadline) = self.detect_deadline(&content_lower) {
                            params.insert("deadline_relative".to_string(), deadline.into());
                        }
                    }
                    if !params.contains_key("depends_on") {
                        if let Some(dep) = self.detect_dependency(&content_lower) {
                            params.insert("depends_on".to_string(), dep.into());
                        }
                    }
                }
                _ => {}
            }

            if !params.is_empty() {
                intent.parameters = serde_json::Value::Object(params);
            }

            intents.push(intent);
        }

        intents
    }

    // --- Enhanced parsing helpers ---

    /// Parse relative date expressions from content. Returns a normalized key
    /// like "tomorrow", "next_week", "next_monday", "in_3_days", etc.
    fn parse_relative_date(&self, content: &str) -> Option<String> {
        // Exact matches (most specific first)
        if content.contains("this evening") || content.contains("tonight") {
            return Some("today".to_string());
        }
        if content.contains("tomorrow morning") || content.contains("tomorrow") {
            return Some("tomorrow".to_string());
        }
        if content.contains("day after tomorrow") {
            return Some("in_2_days".to_string());
        }
        if content.contains("this weekend") {
            let now = Utc::now();
            let days_to_saturday = (Weekday::Sat.num_days_from_monday() as i64
                - now.weekday().num_days_from_monday() as i64
                + 7)
                % 7;
            let days = if days_to_saturday == 0 { 7 } else { days_to_saturday };
            return Some(format!("in_{}_days", days));
        }

        // "next [day]" patterns
        let days = [
            ("monday", Weekday::Mon),
            ("tuesday", Weekday::Tue),
            ("wednesday", Weekday::Wed),
            ("thursday", Weekday::Thu),
            ("friday", Weekday::Fri),
            ("saturday", Weekday::Sat),
            ("sunday", Weekday::Sun),
        ];
        for (name, weekday) in &days {
            let pattern = format!("next {}", name);
            if content.contains(&pattern) {
                let now = Utc::now();
                let current = now.weekday().num_days_from_monday() as i64;
                let target = weekday.num_days_from_monday() as i64;
                let delta = (target - current + 7) % 7;
                let delta = if delta == 0 { 7 } else { delta };
                return Some(format!("in_{}_days", delta));
            }
        }

        // "by [day]" patterns (same logic)
        for (name, weekday) in &days {
            let pattern = format!("by {}", name);
            if content.contains(&pattern) {
                let now = Utc::now();
                let current = now.weekday().num_days_from_monday() as i64;
                let target = weekday.num_days_from_monday() as i64;
                let delta = (target - current + 7) % 7;
                let delta = if delta == 0 { 7 } else { delta };
                return Some(format!("in_{}_days", delta));
            }
        }

        // "in N days/weeks" patterns
        if let Some(n) = self.parse_in_n_time(content) {
            return Some(n);
        }

        // Generic relative periods
        if content.contains("next week") {
            return Some("next_week".to_string());
        }
        if content.contains("next month") {
            return Some("next_month".to_string());
        }
        if content.contains("end of week") || content.contains("eow") {
            let now = Utc::now();
            let days_to_friday = (Weekday::Fri.num_days_from_monday() as i64
                - now.weekday().num_days_from_monday() as i64
                + 7)
                % 7;
            let days = if days_to_friday == 0 { 7 } else { days_to_friday };
            return Some(format!("in_{}_days", days));
        }
        if content.contains("end of month") || content.contains("eom") {
            return Some("end_of_month".to_string());
        }

        None
    }

    /// Parse "in N days/weeks/hours" patterns
    fn parse_in_n_time(&self, content: &str) -> Option<String> {
        // Look for "in X days", "in X weeks", "in X hours"
        let words: Vec<&str> = content.split_whitespace().collect();
        for window in words.windows(3) {
            if window[0] == "in" {
                if let Ok(n) = window[1].parse::<i64>() {
                    match window[2].trim_end_matches(|c: char| !c.is_alphabetic()) {
                        "day" | "days" => return Some(format!("in_{}_days", n)),
                        "week" | "weeks" => return Some(format!("in_{}_days", n * 7)),
                        "hour" | "hours" => return Some("today".to_string()),
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Extract the subject of a reminder ("remind me to ...")
    fn extract_reminder_subject(&self, content: &str) -> Option<String> {
        // Try "remind me to ..."
        for prefix in &["remind me to ", "don't forget to ", "dont forget to ", "remember to "] {
            if let Some(pos) = content.find(prefix) {
                let rest = &content[pos + prefix.len()..];
                let subject = rest
                    .lines()
                    .next()
                    .unwrap_or(rest)
                    .trim()
                    .trim_end_matches('.')
                    .to_string();
                if !subject.is_empty() {
                    return Some(subject);
                }
            }
        }

        // Try "reminder: ..."
        if let Some(pos) = content.find("reminder:") {
            let rest = &content[pos + 9..];
            let subject = rest
                .lines()
                .next()
                .unwrap_or(rest)
                .trim()
                .trim_end_matches('.')
                .to_string();
            if !subject.is_empty() {
                return Some(subject);
            }
        }

        None
    }

    /// Detect priority level from content. Returns (numeric_priority, label).
    fn detect_priority(&self, content: &str) -> Option<(i32, String)> {
        // Explicit priority markers
        if content.contains("p0") || content.contains("critical") {
            return Some((0, "critical".to_string()));
        }
        if content.contains("p1")
            || content.contains("urgent")
            || content.contains("asap")
            || content.contains("immediately")
            || content.contains("right away")
        {
            return Some((1, "high".to_string()));
        }
        if content.contains("p2") || content.contains("high priority") {
            return Some((1, "high".to_string()));
        }
        if content.contains("low priority") || content.contains("p3") || content.contains("when possible") {
            return Some((3, "low".to_string()));
        }

        // Infer from urgency signals
        if content.contains("!!")
            || content.contains("important")
            || content.contains("blocking")
            || content.contains("blocker")
        {
            return Some((1, "high".to_string()));
        }

        None
    }

    /// Detect deadline expressions. Returns a relative key like "tomorrow", "this_week".
    fn detect_deadline(&self, content: &str) -> Option<String> {
        // Explicit deadline markers
        for prefix in &["due by ", "deadline: ", "deadline ", "due: ", "due date: ", "before "] {
            if let Some(pos) = content.find(prefix) {
                let rest = &content[pos + prefix.len()..];
                let fragment = rest.split(|c: char| c == '.' || c == ',' || c == '\n').next().unwrap_or("");
                if let Some(date) = self.parse_relative_date(fragment.trim()) {
                    return Some(date);
                }
                // If the fragment itself is a day name
                let trimmed = fragment.trim().to_lowercase();
                if let Some(date) = self.parse_relative_date(&format!("next {}", trimmed)) {
                    return Some(date);
                }
            }
        }

        // Pattern: "by tomorrow", "by friday", "by next week"
        if let Some(pos) = content.find(" by ") {
            let rest = &content[pos + 4..];
            let fragment = rest.split(|c: char| c == '.' || c == ',' || c == '\n').next().unwrap_or("");
            if let Some(date) = self.parse_relative_date(fragment.trim()) {
                return Some(date);
            }
        }

        None
    }

    /// Detect dependency hints like "after X", "depends on Y", "blocked by Z"
    fn detect_dependency(&self, content: &str) -> Option<String> {
        for prefix in &["depends on ", "blocked by ", "waiting on ", "after completing "] {
            if let Some(pos) = content.find(prefix) {
                let rest = &content[pos + prefix.len()..];
                let dep = rest
                    .split(|c: char| c == '.' || c == ',' || c == '\n')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !dep.is_empty() {
                    return Some(dep);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(content: &str) -> KnowledgeNode {
        KnowledgeNode::new(NodeKind::Fact, content.to_string())
    }

    fn make_engine() -> IntentEngine {
        let store = Arc::new(hx_storage::unified::UnifiedStore::in_memory(384).unwrap());
        IntentEngine::new(store)
    }

    #[test]
    fn test_reminder_detection() {
        let engine = make_engine();

        let node = make_node("Remind me to call John tomorrow");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::ScheduleReminder)));

        let node = make_node("Don't forget to buy groceries");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::ScheduleReminder)));
    }

    #[test]
    fn test_reminder_extracts_subject() {
        let engine = make_engine();

        let node = make_node("Remind me to call John tomorrow");
        let intents = engine.analyze_node(&node);
        let reminder = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ScheduleReminder))
            .unwrap();
        assert_eq!(
            reminder.parameters.get("subject").and_then(|v| v.as_str()),
            Some("call john tomorrow")
        );
    }

    #[test]
    fn test_reminder_relative_dates() {
        let engine = make_engine();

        let node = make_node("Remind me to review the PR next week");
        let intents = engine.analyze_node(&node);
        let reminder = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ScheduleReminder))
            .unwrap();
        assert_eq!(
            reminder
                .parameters
                .get("relative_time")
                .and_then(|v| v.as_str()),
            Some("next_week")
        );
    }

    #[test]
    fn test_reminder_in_n_days() {
        let engine = make_engine();

        let node = make_node("Don't forget to check in 3 days");
        let intents = engine.analyze_node(&node);
        let reminder = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ScheduleReminder))
            .unwrap();
        assert_eq!(
            reminder
                .parameters
                .get("relative_time")
                .and_then(|v| v.as_str()),
            Some("in_3_days")
        );
    }

    #[test]
    fn test_task_detection() {
        let engine = make_engine();

        let node = make_node("- [ ] Complete the report");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::ExtractTask)));

        let node = make_node("TODO: Fix the bug in login");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::ExtractTask)));
    }

    #[test]
    fn test_task_priority_detection() {
        let engine = make_engine();

        let node = make_node("TODO: Fix the urgent login bug ASAP");
        let intents = engine.analyze_node(&node);
        let task = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ExtractTask))
            .unwrap();
        assert_eq!(
            task.parameters.get("priority").and_then(|v| v.as_i64()),
            Some(1)
        );
        assert_eq!(
            task.parameters
                .get("priority_label")
                .and_then(|v| v.as_str()),
            Some("high")
        );
    }

    #[test]
    fn test_task_deadline_detection() {
        let engine = make_engine();

        let node = make_node("TODO: Submit report. Due by tomorrow.");
        let intents = engine.analyze_node(&node);
        let task = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ExtractTask))
            .unwrap();
        assert_eq!(
            task.parameters
                .get("deadline_relative")
                .and_then(|v| v.as_str()),
            Some("tomorrow")
        );
    }

    #[test]
    fn test_task_dependency_detection() {
        let engine = make_engine();

        let node = make_node("TODO: Deploy to prod. Depends on code review.");
        let intents = engine.analyze_node(&node);
        let task = intents
            .iter()
            .find(|i| matches!(i.intent_type, IntentType::ExtractTask))
            .unwrap();
        assert_eq!(
            task.parameters
                .get("depends_on")
                .and_then(|v| v.as_str()),
            Some("code review")
        );
    }

    #[test]
    fn test_link_detection() {
        let engine = make_engine();

        let node = make_node("See [[Project Alpha]] for details");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::SuggestLink)));

        let node = make_node("CC @john about this");
        let intents = engine.analyze_node(&node);
        assert!(intents
            .iter()
            .any(|i| matches!(i.intent_type, IntentType::LinkToProject)));
    }

    #[test]
    fn test_tag_detection() {
        let engine = make_engine();

        let node = make_node("This is about #rust and #performance");
        let intents = engine.analyze_node(&node);
        assert_eq!(
            intents
                .iter()
                .filter(|i| matches!(i.intent_type, IntentType::SuggestTag))
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn test_extract_intents_deduplicates_existing_records() {
        let store = Arc::new(hx_storage::unified::UnifiedStore::in_memory(384).unwrap());
        let engine = IntentEngine::new(Arc::clone(&store));

        let node = make_node("TODO: Fix bug #urgent");
        store.nodes.insert(&node).await.unwrap();
        let first = engine.extract_intents_and_store(&node).await.unwrap();
        assert!(!first.is_empty());

        let second = engine.extract_intents_and_store(&node).await.unwrap();
        assert!(second.is_empty());

        let stored = store
            .nodes
            .list_intents(Some(node.id), Some(IntentStatus::Suggested), 50, 0)
            .await
            .unwrap();
        assert_eq!(stored.len(), first.len());
    }
}
