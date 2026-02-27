//! Cloudflare Tunnel adapter — spawns `cloudflared tunnel --url`.
//!
//! Requires `cloudflared` binary on PATH. Parses the assigned URL from
//! stderr output using a regex for `https://[a-z0-9-]+.trycloudflare.com`.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use tokio::io::AsyncBufReadExt;
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct CloudflareTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl CloudflareTunnel {
    pub fn new(config: TunnelConfig) -> Self {
        Self {
            config,
            child: Arc::new(Mutex::new(None)),
            public_url: Arc::new(Mutex::new(None)),
            started_at: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait]
impl Tunnel for CloudflareTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Cloudflare
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let mut cmd = tokio::process::Command::new("cloudflared");
        cmd.arg("tunnel")
            .arg("--url")
            .arg(format!("http://localhost:{}", self.config.local_port))
            .stderr(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn cloudflared: {e}")))?;

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| HxError::Internal("no stderr from cloudflared".into()))?;

        let url_re =
            Regex::new(r"https://[a-z0-9-]+\.trycloudflare\.com").expect("valid regex");

        let mut reader = tokio::io::BufReader::new(stderr).lines();
        let mut found_url = None;

        // Read stderr lines with a timeout to find the public URL
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let line = tokio::time::timeout_at(deadline, reader.next_line()).await;
            match line {
                Ok(Ok(Some(line))) => {
                    if let Some(m) = url_re.find(&line) {
                        found_url = Some(m.as_str().to_string());
                        break;
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(e)) => {
                    let _ = child.kill().await;
                    return Err(HxError::Internal(format!("stderr read error: {e}")));
                }
                Err(_) => {
                    let _ = child.kill().await;
                    return Err(HxError::Internal(
                        "timeout waiting for cloudflared URL".into(),
                    ));
                }
            }
        }

        let url = found_url.ok_or_else(|| {
            HxError::Internal("cloudflared exited without providing URL".into())
        })?;

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
        // Use try_lock to avoid blocking in a sync context
        let running = self
            .child
            .try_lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        let public_url = self.public_url.try_lock().ok().and_then(|g| g.clone());
        let started_at = self.started_at.try_lock().ok().and_then(|g| *g);
        let error = self.error.try_lock().ok().and_then(|g| g.clone());

        TunnelStatus {
            tunnel_type: TunnelType::Cloudflare,
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
            Some(child) => {
                // Check if process is still running (id returns None if exited)
                Ok(child.id().is_some())
            }
            None => Ok(false),
        }
    }
}
