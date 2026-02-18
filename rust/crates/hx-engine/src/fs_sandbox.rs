//! Filesystem sandbox for secure path resolution and access control.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxLevel {
    ReadOnly,
    Supervised,
    Full,
}

impl SandboxLevel {
    fn rank(self) -> u8 {
        match self {
            Self::ReadOnly => 0,
            Self::Supervised => 1,
            Self::Full => 2,
        }
    }
}

impl std::str::FromStr for SandboxLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "read_only" => Ok(Self::ReadOnly),
            "supervised" => Ok(Self::Supervised),
            "full" => Ok(Self::Full),
            _ => Err(format!("unknown sandbox level: {s}")),
        }
    }
}

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("null byte in path")]
    NullByte,
    #[error("path escapes sandbox root: {0}")]
    JailEscape(PathBuf),
    #[error("symlink escapes sandbox: {0}")]
    SymlinkEscape(PathBuf),
    #[error("blocked path pattern: {0}")]
    BlockedPath(String),
    #[error("permission denied: {operation} requires {required:?}, sandbox is {current:?}")]
    PermissionDenied {
        operation: String,
        required: SandboxLevel,
        current: SandboxLevel,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

const DEFAULT_BLOCKED: &[&str] = &[
    ".env", ".git", ".ssh", ".gnupg", ".aws", "/etc", "/proc", "/sys", "/dev",
];

pub struct FsSandbox {
    root: PathBuf,
    level: SandboxLevel,
    blocked_patterns: Vec<String>,
    browser_domain_allowlist: Vec<String>,
}

impl FsSandbox {
    /// Create a new sandbox from the engine's SandboxConfig.
    pub fn new(
        root: PathBuf,
        level: SandboxLevel,
        blocked_patterns: Vec<String>,
        browser_domain_allowlist: Vec<String>,
    ) -> Self {
        let mut all_blocked: Vec<String> = DEFAULT_BLOCKED.iter().map(|s| s.to_string()).collect();
        all_blocked.extend(blocked_patterns);
        Self {
            root,
            level,
            blocked_patterns: all_blocked,
            browser_domain_allowlist,
        }
    }

    /// Create a sandbox from the engine config's sandbox section.
    pub fn from_config(config: &crate::config::SandboxConfig) -> Self {
        let level = config
            .level
            .parse::<SandboxLevel>()
            .unwrap_or(SandboxLevel::Supervised);
        let root = PathBuf::from(&config.root);
        Self::new(
            root,
            level,
            config.blocked_patterns.clone(),
            config.browser_domain_allowlist.clone(),
        )
    }

    /// Resolve and validate a path within the sandbox.
    pub fn resolve(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        // 1. Check for null bytes
        let path_str = path.to_string_lossy();
        if path_str.contains('\0') {
            return Err(SandboxError::NullByte);
        }

        // 2. Join with root if relative
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };

        // 3. Canonicalize (resolves symlinks) — fall back to logical normalization
        //    if the path doesn't exist yet.
        let canonical = match joined.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Path doesn't exist yet; do logical normalization.
                normalize_logical(&joined)
            }
            Err(e) => return Err(SandboxError::Io(e)),
        };

        // 4. Verify starts_with(root)
        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(_) => normalize_logical(&self.root),
        };
        if !canonical.starts_with(&canonical_root) {
            // Check if it's a symlink escape
            if joined.is_symlink() {
                return Err(SandboxError::SymlinkEscape(path.to_path_buf()));
            }
            return Err(SandboxError::JailEscape(path.to_path_buf()));
        }

        // 5. Check blocked patterns
        let canonical_str = canonical.to_string_lossy();
        for pattern in &self.blocked_patterns {
            if pattern.starts_with('/') {
                // Absolute path prefix block
                if canonical_str.starts_with(pattern) {
                    return Err(SandboxError::BlockedPath(pattern.clone()));
                }
            } else {
                // Component-based block (e.g., ".git", ".env")
                for component in canonical.components() {
                    if let std::path::Component::Normal(os) = component {
                        if os.to_string_lossy() == *pattern {
                            return Err(SandboxError::BlockedPath(pattern.clone()));
                        }
                    }
                }
            }
        }

        Ok(canonical)
    }

    /// Check if an operation is allowed at the current sandbox level.
    pub fn check_permission(
        &self,
        operation: &str,
        required: SandboxLevel,
    ) -> Result<(), SandboxError> {
        if self.level.rank() >= required.rank() {
            Ok(())
        } else {
            Err(SandboxError::PermissionDenied {
                operation: operation.to_string(),
                required,
                current: self.level,
            })
        }
    }

    /// Check if a browser domain is in the allowlist.
    pub fn check_browser_domain(&self, url: &str) -> Result<(), SandboxError> {
        // Extract domain from URL
        let domain = extract_domain(url);
        if self
            .browser_domain_allowlist
            .iter()
            .any(|allowed| domain == *allowed)
        {
            Ok(())
        } else {
            Err(SandboxError::BlockedPath(format!(
                "domain not in allowlist: {domain}"
            )))
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn level(&self) -> SandboxLevel {
        self.level
    }
}

/// Extract domain from a URL string (simple parsing without pulling in url crate).
fn extract_domain(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    without_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("")
        .to_string()
}

/// Logical path normalization without touching the filesystem.
fn normalize_logical(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !components.is_empty() {
                    components.pop();
                }
            }
            std::path::Component::CurDir => {}
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox_in(dir: &Path) -> FsSandbox {
        FsSandbox::new(
            dir.to_path_buf(),
            SandboxLevel::Supervised,
            vec![],
            vec!["example.com".to_string()],
        )
    }

    #[test]
    fn null_byte_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());
        let result = sandbox.resolve(Path::new("file\0name.txt"));
        assert!(matches!(result, Err(SandboxError::NullByte)));
    }

    #[test]
    fn traversal_detected() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());
        let result = sandbox.resolve(Path::new("../../etc/passwd"));
        assert!(
            matches!(result, Err(SandboxError::JailEscape(_))),
            "expected JailEscape, got: {:?}",
            result
        );
    }

    #[test]
    fn symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());

        // Create a symlink pointing outside the sandbox
        let link_path = dir.path().join("escape_link");
        #[cfg(unix)]
        std::os::unix::fs::symlink("/tmp", &link_path).unwrap();
        #[cfg(not(unix))]
        {
            // On non-unix, skip this test
            return;
        }

        let result = sandbox.resolve(Path::new("escape_link"));
        assert!(
            matches!(
                result,
                Err(SandboxError::JailEscape(_) | SandboxError::SymlinkEscape(_))
            ),
            "expected escape error, got: {:?}",
            result
        );
    }

    #[test]
    fn dotfile_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());

        // Create the .git directory so canonicalize works
        fs::create_dir_all(dir.path().join(".git")).unwrap();

        let result = sandbox.resolve(Path::new(".git"));
        assert!(
            matches!(result, Err(SandboxError::BlockedPath(_))),
            "expected BlockedPath, got: {:?}",
            result
        );
    }

    #[test]
    fn system_dir_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());
        let result = sandbox.resolve(Path::new("/etc/passwd"));
        // /etc is blocked and also escapes jail
        assert!(result.is_err());
    }

    #[test]
    fn valid_path_resolves() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());

        // Create a valid file
        let file_path = dir.path().join("notes.txt");
        fs::write(&file_path, "hello").unwrap();

        let result = sandbox.resolve(Path::new("notes.txt"));
        assert!(result.is_ok(), "expected Ok, got: {:?}", result);
        assert!(result.unwrap().ends_with("notes.txt"));
    }

    #[test]
    fn readonly_blocks_write() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = FsSandbox::new(
            dir.path().to_path_buf(),
            SandboxLevel::ReadOnly,
            vec![],
            vec![],
        );

        let result = sandbox.check_permission("write", SandboxLevel::Supervised);
        assert!(
            matches!(result, Err(SandboxError::PermissionDenied { .. })),
            "expected PermissionDenied, got: {:?}",
            result
        );
    }

    #[test]
    fn supervised_allows_read() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = FsSandbox::new(
            dir.path().to_path_buf(),
            SandboxLevel::Supervised,
            vec![],
            vec![],
        );

        let result = sandbox.check_permission("read", SandboxLevel::ReadOnly);
        assert!(result.is_ok());
    }

    #[test]
    fn browser_domain_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());
        let result = sandbox.check_browser_domain("https://example.com/page");
        assert!(result.is_ok());
    }

    #[test]
    fn browser_domain_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let sandbox = sandbox_in(dir.path());
        let result = sandbox.check_browser_domain("https://evil.com/hack");
        assert!(result.is_err());
    }
}
