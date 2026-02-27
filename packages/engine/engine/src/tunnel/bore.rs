//! Bore tunnel adapter — spawns `bore local` to expose a port via bore.pub.
//!
//! Requires `bore` binary on PATH. Parses the assigned remote port from
//! stdout output to construct the public URL.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use regex::Regex;
use tokio::io::AsyncBufReadExt;
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct BoreTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl BoreTunnel {
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
impl Tunnel for BoreTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Bore
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let server = self
            .config
            .get_setting("server")
            .unwrap_or("bore.pub");

        let mut cmd = tokio::process::Command::new("bore");
        cmd.arg("local")
            .arg(self.config.local_port.to_string())
            .arg("--to")
            .arg(server)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        // Add secret if configured
        if let Some(secret) = self.config.get_setting("secret") {
            cmd.arg("--secret").arg(secret);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn bore: {e}")))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| HxError::Internal("no stdout from bore".into()))?;

        // bore outputs something like "listening at bore.pub:12345"
        let port_re = Regex::new(r"(?:listening at |:)(\d{4,5})").expect("valid regex");

        let mut reader = tokio::io::BufReader::new(stdout).lines();
        let mut found_url = None;

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            let line = tokio::time::timeout_at(deadline, reader.next_line()).await;
            match line {
                Ok(Ok(Some(line))) => {
                    if let Some(caps) = port_re.captures(&line) {
                        if let Some(port) = caps.get(1) {
                            found_url =
                                Some(format!("{}:{}", server, port.as_str()));
                            break;
                        }
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
                        "timeout waiting for bore port assignment".into(),
                    ));
                }
            }
        }

        let url = found_url
            .ok_or_else(|| HxError::Internal("bore exited without port assignment".into()))?;

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
            tunnel_type: TunnelType::Bore,
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
