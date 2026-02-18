//! Ngrok tunnel adapter — spawns `ngrok http` and queries the local API.
//!
//! Requires `ngrok` binary on PATH. Discovers the public URL by querying
//! the ngrok local API at `http://127.0.0.1:4040/api/tunnels`.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct NgrokTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl NgrokTunnel {
    pub fn new(config: TunnelConfig) -> Self {
        Self {
            config,
            child: Arc::new(Mutex::new(None)),
            public_url: Arc::new(Mutex::new(None)),
            started_at: Arc::new(Mutex::new(None)),
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Query the ngrok local API for the tunnel's public URL.
    async fn fetch_public_url(&self) -> MvResult<String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| HxError::Internal(e.to_string()))?;

        let resp = client
            .get("http://127.0.0.1:4040/api/tunnels")
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("ngrok API query failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("ngrok API parse failed: {e}")))?;

        body.get("tunnels")
            .and_then(|t| t.as_array())
            .and_then(|arr| arr.first())
            .and_then(|t| t.get("public_url"))
            .and_then(|u| u.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| HxError::Internal("no tunnels found in ngrok API".into()))
    }
}

#[async_trait]
impl Tunnel for NgrokTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Ngrok
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let mut cmd = tokio::process::Command::new("ngrok");
        cmd.arg("http")
            .arg(self.config.local_port.to_string())
            .arg("--log=stdout")
            .arg("--log-format=json")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        // Add authtoken if configured
        if let Some(token) = self.config.get_setting("authtoken") {
            cmd.arg("--authtoken").arg(token);
        }

        let child = cmd
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn ngrok: {e}")))?;

        *guard = Some(child);

        // Wait for ngrok to start, then query the API
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let mut url = None;
        for _ in 0..10 {
            match self.fetch_public_url().await {
                Ok(u) => {
                    url = Some(u);
                    break;
                }
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
        }

        let url = match url {
            Some(u) => u,
            None => {
                // Cleanup on failure
                if let Some(mut child) = guard.take() {
                    let _ = child.kill().await;
                }
                return Err(HxError::Internal(
                    "timeout waiting for ngrok public URL".into(),
                ));
            }
        };

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
            tunnel_type: TunnelType::Ngrok,
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
