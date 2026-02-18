//! SSH reverse tunnel adapter — spawns `ssh -R` to forward a remote port.
//!
//! Requires `ssh` binary on PATH and appropriate key-based authentication.
//! The public URL is constructed from the configured host and remote port settings.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::process::Child;
use tokio::sync::Mutex;

use hx_core::{HxError, MvResult};

use super::{Tunnel, TunnelConfig, TunnelStatus, TunnelType};

pub struct SshTunnel {
    config: TunnelConfig,
    child: Arc<Mutex<Option<Child>>>,
    public_url: Arc<Mutex<Option<String>>>,
    started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl SshTunnel {
    pub fn new(config: TunnelConfig) -> MvResult<Self> {
        if config.get_setting("host").is_none() {
            return Err(HxError::Config(
                "SSH tunnel requires 'host' setting".into(),
            ));
        }
        if config.get_setting("user").is_none() {
            return Err(HxError::Config(
                "SSH tunnel requires 'user' setting".into(),
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
impl Tunnel for SshTunnel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn tunnel_type(&self) -> TunnelType {
        TunnelType::Ssh
    }

    async fn start(&self) -> MvResult<String> {
        let mut guard = self.child.lock().await;
        if guard.is_some() {
            return Err(HxError::InvalidInput("tunnel already running".into()));
        }

        let host = self
            .config
            .get_setting("host")
            .ok_or_else(|| HxError::Config("missing 'host' setting".into()))?;
        let user = self
            .config
            .get_setting("user")
            .ok_or_else(|| HxError::Config("missing 'user' setting".into()))?;
        let remote_port = self
            .config
            .get_setting("remote_port")
            .unwrap_or("0");

        // ssh -R <remote_port>:localhost:<local_port> user@host -N
        let forward_spec = format!("{}:localhost:{}", remote_port, self.config.local_port);

        let mut cmd = tokio::process::Command::new("ssh");
        cmd.arg("-R")
            .arg(&forward_spec)
            .arg(format!("{user}@{host}"))
            .arg("-N")
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("ExitOnForwardFailure=yes")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        // Add identity file if configured
        if let Some(key) = self.config.get_setting("identity_file") {
            cmd.arg("-i").arg(key);
        }

        let child = cmd
            .spawn()
            .map_err(|e| HxError::Internal(format!("failed to spawn ssh: {e}")))?;

        // Construct URL from settings
        let url = if remote_port == "0" {
            format!("{host}:<dynamic>")
        } else {
            format!("{host}:{remote_port}")
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
            tunnel_type: TunnelType::Ssh,
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
