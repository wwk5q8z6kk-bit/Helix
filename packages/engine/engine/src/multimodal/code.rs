//! Source code processing: language detection, function/class extraction.
//!
//! Reads source code files directly and uses regex patterns to extract
//! structural elements (functions, classes, structs) per language.

use async_trait::async_trait;
use hx_core::{HxError, KnowledgeNode, MvResult};
use regex::Regex;
use std::path::Path;

use super::{check_file_size, ModalityProcessor, ModalityStatus, ProcessingResult};

/// Source code processor for multiple programming languages.
pub struct CodeProcessor;

impl CodeProcessor {
    pub fn new() -> Self {
        Self
    }
}

/// Detected language with associated regex patterns.
struct LanguagePatterns {
    name: &'static str,
    function_re: Regex,
    class_re: Option<Regex>,
}

/// Detect language from MIME type or file extension.
fn detect_language(file_path: &str, _mime: Option<&str>) -> &'static str {
    let ext = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "py" => "python",
        "rs" => "rust",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "mts" | "cts" => "typescript",
        "java" => "java",
        "c" | "h" => "c",
        "go" => "go",
        "rb" => "ruby",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        _ => "unknown",
    }
}

/// Get regex patterns for the detected language.
fn get_patterns(language: &str) -> LanguagePatterns {
    match language {
        "python" => LanguagePatterns {
            name: "Python",
            function_re: Regex::new(r"(?m)^\s*(?:async\s+)?def\s+(\w+)").expect("valid regex"),
            class_re: Some(Regex::new(r"(?m)^\s*class\s+(\w+)").expect("valid regex")),
        },
        "rust" => LanguagePatterns {
            name: "Rust",
            function_re: Regex::new(r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").expect("valid regex"),
            class_re: Some(
                Regex::new(r"(?m)^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(\w+)")
                    .expect("valid regex"),
            ),
        },
        "javascript" | "typescript" => LanguagePatterns {
            name: if language == "typescript" { "TypeScript" } else { "JavaScript" },
            function_re: Regex::new(
                r"(?m)(?:^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)|^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()"
            ).expect("valid regex"),
            class_re: Some(
                Regex::new(r"(?m)^\s*(?:export\s+)?class\s+(\w+)").expect("valid regex"),
            ),
        },
        "java" => LanguagePatterns {
            name: "Java",
            function_re: Regex::new(
                r"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\("
            ).expect("valid regex"),
            class_re: Some(
                Regex::new(r"(?m)^\s*(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)")
                    .expect("valid regex"),
            ),
        },
        "c" | "cpp" => LanguagePatterns {
            name: if language == "cpp" { "C++" } else { "C" },
            function_re: Regex::new(r"(?m)^\s*(?:\w+[\s*]+)+(\w+)\s*\([^)]*\)\s*\{").expect("valid regex"),
            class_re: if language == "cpp" {
                Some(Regex::new(r"(?m)^\s*(?:class|struct)\s+(\w+)").expect("valid regex"))
            } else {
                Some(Regex::new(r"(?m)^\s*(?:struct|typedef\s+struct)\s+(\w+)").expect("valid regex"))
            },
        },
        "go" => LanguagePatterns {
            name: "Go",
            function_re: Regex::new(r"(?m)^func\s+(?:\([^)]+\)\s+)?(\w+)").expect("valid regex"),
            class_re: Some(
                Regex::new(r"(?m)^type\s+(\w+)\s+(?:struct|interface)").expect("valid regex"),
            ),
        },
        "ruby" => LanguagePatterns {
            name: "Ruby",
            function_re: Regex::new(r"(?m)^\s*def\s+(\w+)").expect("valid regex"),
            class_re: Some(
                Regex::new(r"(?m)^\s*(?:class|module)\s+(\w+)").expect("valid regex"),
            ),
        },
        _ => LanguagePatterns {
            name: "Unknown",
            function_re: Regex::new(r"(?m)(?:function|def|fn)\s+(\w+)").expect("valid regex"),
            class_re: None,
        },
    }
}

/// Extract function/method names from source code.
fn extract_functions(content: &str, patterns: &LanguagePatterns) -> Vec<String> {
    patterns
        .function_re
        .captures_iter(content)
        .filter_map(|cap| {
            // Try group 1 first, then group 2 (for JS arrow functions)
            cap.get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str().to_string())
        })
        .collect()
}

/// Extract class/struct/trait names from source code.
fn extract_classes(content: &str, patterns: &LanguagePatterns) -> Vec<String> {
    patterns
        .class_re
        .as_ref()
        .map(|re| {
            re.captures_iter(content)
                .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
                .collect()
        })
        .unwrap_or_default()
}

#[async_trait]
impl ModalityProcessor for CodeProcessor {
    fn name(&self) -> &'static str {
        "code"
    }

    fn handles(&self) -> &[&str] {
        &[
            "text/x-python",
            "text/x-rust",
            "text/javascript",
            "text/x-java",
            "text/x-c",
            "text/x-go",
            "text/x-ruby",
            "text/x-typescript",
        ]
    }

    fn status(&self) -> ModalityStatus {
        ModalityStatus::new(self.name(), true, self.handles())
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing source code file");

        check_file_size(file_path).map_err(HxError::Storage)?;

        let content = std::fs::read_to_string(file_path)
            .map_err(|e| HxError::Storage(format!("failed to read source file: {e}")))?;

        let language = detect_language(file_path, None);
        let patterns = get_patterns(language);

        let functions = extract_functions(&content, &patterns);
        let classes = extract_classes(&content, &patterns);
        let line_count = content.lines().count();

        // Prefix content with language annotation for better embedding
        let annotated = format!("[Language: {}]\n\n{}", patterns.name, content);

        let summary = format!(
            "{} source: {} lines, {} functions, {} classes/structs",
            patterns.name,
            line_count,
            functions.len(),
            classes.len()
        );

        let mut result = ProcessingResult::new(annotated)
            .with_tag("code".to_string())
            .with_tag(language.to_string())
            .with_summary(summary);

        result
            .metadata
            .insert("language".into(), serde_json::json!(language));
        result
            .metadata
            .insert("line_count".into(), serde_json::json!(line_count));
        result
            .metadata
            .insert("function_count".into(), serde_json::json!(functions.len()));
        result
            .metadata
            .insert("class_count".into(), serde_json::json!(classes.len()));

        if !functions.is_empty() {
            result
                .metadata
                .insert("functions".into(), serde_json::json!(functions));
        }
        if !classes.is_empty() {
            result
                .metadata
                .insert("classes".into(), serde_json::json!(classes));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn name_returns_expected() {
        let processor = CodeProcessor::new();
        assert_eq!(processor.name(), "code");
    }

    #[test]
    fn handles_returns_correct_types() {
        let processor = CodeProcessor::new();
        let types = processor.handles();
        assert!(types.contains(&"text/x-python"));
        assert!(types.contains(&"text/x-rust"));
        assert!(types.contains(&"text/javascript"));
        assert!(types.contains(&"text/x-java"));
        assert!(types.contains(&"text/x-c"));
        assert!(types.contains(&"text/x-go"));
        assert!(types.contains(&"text/x-ruby"));
        assert!(types.contains(&"text/x-typescript"));
        assert_eq!(types.len(), 8);
    }

    #[tokio::test]
    async fn process_extracts_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.py");
        std::fs::write(
            &path,
            "def hello():\n    print('hello')\n\nclass MyClass:\n    pass\n",
        )
        .unwrap();

        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("[Language: Python]"));
        assert!(result.text_content.contains("def hello"));
        assert!(result.text_content.contains("class MyClass"));
    }

    #[tokio::test]
    async fn process_populates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(
            &path,
            "pub fn main() {}\nfn helper() {}\npub struct Config {}\n",
        )
        .unwrap();

        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert_eq!(result.metadata["language"], serde_json::json!("rust"));
        assert_eq!(result.metadata["line_count"], serde_json::json!(3));
        assert_eq!(result.metadata["function_count"], serde_json::json!(2));
        assert_eq!(result.metadata["class_count"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn process_adds_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.go");
        std::fs::write(&path, "package main\nfunc main() {}\n").unwrap();

        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.suggested_tags.contains(&"code".to_string()));
        assert!(result.suggested_tags.contains(&"go".to_string()));
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.py", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.py");
        std::fs::write(&path, "").unwrap();

        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("[Language: Python]"));
        assert_eq!(result.metadata["function_count"], serde_json::json!(0));
        assert_eq!(result.metadata["class_count"], serde_json::json!(0));
    }

    #[tokio::test]
    async fn process_handles_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.js");
        std::fs::write(&path, "{{{{ not valid js @@@ }}}}").unwrap();

        let processor = CodeProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Should not panic
        let result = processor.process(path.to_str().unwrap(), &node).await;
        assert!(result.is_ok());
    }

    #[test]
    fn status_reflects_availability() {
        let processor = CodeProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "code");
        assert!(status.available);
        assert_eq!(status.supported_types.len(), 8);
    }

    #[test]
    fn detect_language_from_extensions() {
        assert_eq!(detect_language("foo.py", None), "python");
        assert_eq!(detect_language("bar.rs", None), "rust");
        assert_eq!(detect_language("baz.js", None), "javascript");
        assert_eq!(detect_language("qux.ts", None), "typescript");
        assert_eq!(detect_language("Main.java", None), "java");
        assert_eq!(detect_language("main.c", None), "c");
        assert_eq!(detect_language("main.go", None), "go");
        assert_eq!(detect_language("app.rb", None), "ruby");
        assert_eq!(detect_language("no_ext", None), "unknown");
    }

    #[test]
    fn extract_python_functions_and_classes() {
        let code = "def foo():\n    pass\nasync def bar():\n    pass\nclass Baz:\n    pass\n";
        let patterns = get_patterns("python");
        let funcs = extract_functions(code, &patterns);
        let classes = extract_classes(code, &patterns);
        assert!(funcs.contains(&"foo".to_string()));
        assert!(funcs.contains(&"bar".to_string()));
        assert!(classes.contains(&"Baz".to_string()));
    }
}
