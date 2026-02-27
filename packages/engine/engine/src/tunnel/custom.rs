//! Custom tunnel adapter — runs a user-specified command.
//!
//! Extracts the public URL from stdout using an optional regex pattern,
//! or uses a static URL from settings.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use tokio::io::AsyncBufReadExt;
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct CustomTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl CustomTunnel {
    pub fn new(config: TunnelConfig) -> MvResult<Self> {
        if config.get_setting("command").is_none() {
            return Err(HxError::Config(
                "Custom tunnel requires 'command' setting".into(),
            ));
        }
        Ok(Self {
            config,
            child: Arc::new(Mutex::new(None)),
            public_url: Arc::new(Mutex::new(None)),
            started_at: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
        })
    }
}

#[async_trait]
impl Tunnel for CustomTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Custom
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let command_str = self
            .config
            .get_setting("command")
            .ok_or_else(|| HxError::Config("missing 'command' setting".into()))?;

        // Split command into program and args (simple shell-like splitting)
        let parts: Vec<&str> = command_str.split_whitespace().collect();
        if parts.is_empty() {
            return Err(HxError::Config("empty command".into()));
        }

        let program = parts[0];
        let args = &parts[1..];

        // Replace {port} placeholder in args
        let port_str = self.config.local_port.to_string();
        let resolved_args: Vec<String> = args
            .iter()
            .map(|a| a.replace("{port}", &port_str))
            .collect();

        let mut cmd = tokio::process::Command::new(program);
        cmd.args(&resolved_args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn custom tunnel: {e}")))?;

        // Try to extract URL from stdout if url_regex is set
        let url = if let Some(regex_pattern) = self.config.get_setting("url_regex") {
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| HxError::Internal("no stdout from custom tunnel".into()))?;

            let url_re = Regex::new(regex_pattern)
                .map_err(|e| HxError::Config(format!("invalid url_regex: {e}")))?;

            let mut reader = tokio::io::BufReader::new(stdout).lines();
            let mut found = None;

            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
            loop {
                let line = tokio::time::timeout_at(deadline, reader.next_line()).await;
                match line {
                    Ok(Ok(Some(line))) => {
                        if let Some(m) = url_re.find(&line) {
                            found = Some(m.as_str().to_string());
                            break;
                        }
                    }
                    Ok(Ok(None)) => break,
                    Ok(Err(e)) => {
                        let _ = child.kill().await;
                        return Err(HxError::Internal(format!("stdout read error: {e}")));
                    }
                    Err(_) => {
                        let _ = child.kill().await;
                        return Err(HxError::Internal(
                            "timeout waiting for custom tunnel URL".into(),
                        ));
                    }
                }
            }

            found.ok_or_else(|| {
                HxError::Internal("custom tunnel exited without matching URL".into())
            })?
        } else if let Some(static_url) = self.config.get_setting("url") {
            // Use a static URL from settings
            static_url.to_string()
        } else {
            // No URL extraction method — just report the command is running
            format!("custom://localhost:{}", self.config.local_port)
        };

        *self.public_url.lock().await = Some(url.clone());
        *self.started_at.lock().await = Some(Utc::now());
        *self.error.lock().await = None;
        *guard = Some(child);

        Ok(url)
    }

    async fn stop(&self) -> MvResult<()> {
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        *self.public_url.lock().await = None;
        *self.started_at.lock().await = None;
        Ok(())
    }

    fn status(&self) -> TunnelStatus {
        let running = self
            .child
            .try_lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        let public_url = self.public_url.try_lock().ok().and_then(|g| g.clone());
        let started_at = self.started_at.try_lock().ok().and_then(|g| *g);
        let error = self.error.try_lock().ok().and_then(|g| g.clone());

        TunnelStatus {
            tunnel_type: TunnelType::Custom,
            name: self.config.name.clone(),
            running,
            public_url,
            error,
            started_at,
        }
    }

    async fn health_check(&self) -> MvResult<bool> {
        let guard = self.child.lock().await;
        match guard.as_ref() {
            Some(child) => Ok(child.id().is_some()),
            None => Ok(false),
        }
    }
}
