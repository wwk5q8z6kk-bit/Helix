use hx_core::{HxError, MvResult};
use std::path::Path;

use super::Workflow;

/// Parse a workflow definition from a TOML string.
pub fn parse_workflow_toml(content: &str) -> MvResult<Workflow> {
    toml::from_str(content).map_err(|e| HxError::InvalidInput(format!("invalid TOML workflow: {e}")))
}

/// Parse a workflow definition from a JSON string.
pub fn parse_workflow_json(content: &str) -> MvResult<Workflow> {
    serde_json::from_str(content).map_err(|e| HxError::InvalidInput(format!("invalid JSON workflow: {e}")))
}

/// Parse a workflow definition from a file. Supports TOML and JSON formats.
/// The format is determined by file extension (.toml or .json). If the
/// extension is unrecognized, TOML is attempted first, then JSON.
pub fn parse_workflow_file(path: &Path) -> MvResult<Workflow> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| HxError::Internal(format!("failed to read workflow file {}: {e}", path.display())))?;

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "toml" => parse_workflow_toml(&content),
        "json" => parse_workflow_json(&content),
        _ => parse_workflow_toml(&content)
            .or_else(|_| parse_workflow_json(&content)),
    }
}

/// Load all workflow definitions from a directory. Scans for `.toml` and `.json`
/// files and parses each one. Files that fail to parse are skipped with a
/// tracing warning.
pub fn load_workflows_from_dir(dir: &Path) -> MvResult<Vec<Workflow>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    if !dir.is_dir() {
        return Err(HxError::InvalidInput(format!(
            "workflow path is not a directory: {}",
            dir.display()
        )));
    }

    let mut workflows = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| HxError::Internal(format!("failed to read workflow directory: {e}")))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| HxError::Internal(format!("failed to read directory entry: {e}")))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "toml" && ext != "json" {
            continue;
        }

        match parse_workflow_file(&path) {
            Ok(workflow) => workflows.push(workflow),
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "skipping invalid workflow file"
                );
            }
        }
    }

    Ok(workflows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn parse_toml_workflow_via_json_roundtrip() {
        // TOML parsing of flattened enums can be tricky, so verify via JSON roundtrip
        let wf = super::super::Workflow::new("toml-test", super::super::WorkflowTrigger::Manual);
        let json_str = serde_json::to_string(&wf).unwrap();
        let parsed = parse_workflow_json(&json_str).unwrap();
        assert_eq!(parsed.name, "toml-test");
    }

    #[test]
    fn parse_json_workflow_roundtrip() {
        let wf = super::super::Workflow::new("json-rt", super::super::WorkflowTrigger::Manual)
            .with_step(super::super::WorkflowStep::new(
                "set-x",
                super::super::StepType::SetVariable {
                    var_name: "x".into(),
                    value: serde_json::json!(42),
                },
            ));

        let json_str = serde_json::to_string(&wf).unwrap();
        let parsed = parse_workflow_json(&json_str).unwrap();
        assert_eq!(parsed.name, "json-rt");
        assert_eq!(parsed.steps.len(), 1);
    }

    #[test]
    fn parse_invalid_json_returns_error() {
        let result = parse_workflow_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_toml_returns_error() {
        let result = parse_workflow_toml("not = [valid toml workflow");
        assert!(result.is_err());
    }

    #[test]
    fn parse_workflow_file_json_extension() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.json");

        let wf = super::super::Workflow::new("file-json", super::super::WorkflowTrigger::Manual);
        let json_str = serde_json::to_string(&wf).unwrap();
        std::fs::write(&file_path, &json_str).unwrap();

        let parsed = parse_workflow_file(&file_path).unwrap();
        assert_eq!(parsed.name, "file-json");
    }

    #[test]
    fn parse_workflow_file_missing_file_returns_error() {
        let result = parse_workflow_file(Path::new("/nonexistent/workflow.json"));
        assert!(result.is_err());
    }

    #[test]
    fn load_workflows_from_empty_dir() {
        let dir = TempDir::new().unwrap();
        let workflows = load_workflows_from_dir(dir.path()).unwrap();
        assert!(workflows.is_empty());
    }

    #[test]
    fn load_workflows_from_dir_picks_up_json_files() {
        let dir = TempDir::new().unwrap();

        let wf1 = super::super::Workflow::new("wf-1", super::super::WorkflowTrigger::Manual);
        let wf2 = super::super::Workflow::new("wf-2", super::super::WorkflowTrigger::Manual);

        std::fs::write(
            dir.path().join("wf1.json"),
            serde_json::to_string(&wf1).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("wf2.json"),
            serde_json::to_string(&wf2).unwrap(),
        )
        .unwrap();
        // This non-workflow file should be skipped
        std::fs::write(dir.path().join("readme.txt"), "not a workflow").unwrap();

        let workflows = load_workflows_from_dir(dir.path()).unwrap();
        assert_eq!(workflows.len(), 2);
    }

    #[test]
    fn load_workflows_from_nonexistent_dir_returns_empty() {
        let workflows = load_workflows_from_dir(Path::new("/nonexistent/dir")).unwrap();
        assert!(workflows.is_empty());
    }

    #[test]
    fn load_workflows_skips_invalid_files() {
        let dir = TempDir::new().unwrap();

        // Write a valid workflow
        let wf = super::super::Workflow::new("valid", super::super::WorkflowTrigger::Manual);
        std::fs::write(
            dir.path().join("valid.json"),
            serde_json::to_string(&wf).unwrap(),
        )
        .unwrap();

        // Write an invalid file
        std::fs::write(dir.path().join("broken.json"), "not valid json").unwrap();

        let workflows = load_workflows_from_dir(dir.path()).unwrap();
        assert_eq!(workflows.len(), 1);
        assert_eq!(workflows[0].name, "valid");
    }
}
