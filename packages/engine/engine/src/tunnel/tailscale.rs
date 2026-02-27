//! Tailscale Funnel adapter — exposes a local port via Tailscale Funnel.
//!
//! Requires `tailscale` binary on PATH and an active Tailscale connection.
//! Discovers the public URL by querying `tailscale status --json` for the hostname.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct TailscaleTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl TailscaleTunnel {
    pub fn new(config: TunnelConfig) -> Self {
        Self {
            config,
            child: Arc::new(Mutex::new(None)),
            public_url: Arc::new(Mutex::new(None)),
            started_at: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Get the Tailscale hostname by running `tailscale status --json`.
    async fn get_hostname(&self) -> MvResult<String> {
        let output = tokio::process::Command::new("tailscale")
            .args(["status", "--json"])
            .output()
            .await
            .map_err(|e| HxError::Internal(format!("failed to run tailscale status: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HxError::Internal(format!(
                "tailscale status failed: {stderr}"
            )));
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| HxError::Internal(format!("failed to parse tailscale status: {e}")))?;

        let dns_name = json
            .get("Self")
            .and_then(|s| s.get("DNSName"))
            .and_then(|d| d.as_str())
            .ok_or_else(|| HxError::Internal("no DNSName in tailscale status".into()))?;

        // DNSName typically ends with a trailing dot — strip it
        let hostname = dns_name.trim_end_matches('.');
        Ok(format!("https://{hostname}"))
    }
}

#[async_trait]
impl Tunnel for TailscaleTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Tailscale
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let child = tokio::process::Command::new("tailscale")
            .arg("funnel")
            .arg(self.config.local_port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn tailscale funnel: {e}")))?;

        *guard = Some(child);

        // Allow tailscale funnel to register
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let url = self.get_hostname().await?;

        *self.public_url.lock().await = Some(url.clone());
        *self.started_at.lock().await = Some(Utc::now());
        *self.error.lock().await = None;

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
            tunnel_type: TunnelType::Tailscale,
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
