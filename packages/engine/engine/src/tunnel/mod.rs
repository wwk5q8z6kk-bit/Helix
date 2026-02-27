//! Tunnel system for exposing local Helix server externally.
//!
//! Supports Cloudflare, Ngrok, Tailscale, Bore, SSH, and custom tunnels.
//! All implementations spawn child processes via `tokio::process::Command`.

pub mod bore;
pub mod cloudflare;
pub mod custom;
pub mod ngrok;
pub mod ssh;
pub mod tailscale;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use hx_core::MvResult;

// ---------------------------------------------------------------------------
// Tunnel Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TunnelType {
    Cloudflare,
    Ngrok,
    Tailscale,
    Bore,
    Ssh,
    Custom,
}

impl TunnelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cloudflare => "cloudflare",
            Self::Ngrok => "ngrok",
            Self::Tailscale => "tailscale",
            Self::Bore => "bore",
            Self::Ssh => "ssh",
            Self::Custom => "custom",
        }
    }
}

impl std::str::FromStr for TunnelType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cloudflare" => Ok(Self::Cloudflare),
            "ngrok" => Ok(Self::Ngrok),
            "tailscale" => Ok(Self::Tailscale),
            "bore" => Ok(Self::Bore),
            "ssh" => Ok(Self::Ssh),
            "custom" => Ok(Self::Custom),
            _ => Err(format!("unknown tunnel type: {s}")),
        }
    }
}

impl std::fmt::Display for TunnelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Tunnel Configuration & Status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelConfig {
    pub id: Uuid,
    pub tunnel_type: TunnelType,
    pub name: String,
    pub local_port: u16,
    pub settings: HashMap<String, String>,
}

impl TunnelConfig {
    pub fn new(tunnel_type: TunnelType, name: impl Into<String>, local_port: u16) -> Self {
        Self {
            id: Uuid::now_v7(),
            tunnel_type,
            name: name.into(),
            local_port,
            settings: HashMap::new(),
        }
    }

    pub fn with_setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(key.into(), value.into());
        self
    }

    pub fn get_setting(&self, key: &str) -> Option<&str> {
        self.settings.get(key).map(|s| s.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelStatus {
    pub tunnel_type: TunnelType,
    pub name: String,
    pub running: bool,
    pub public_url: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Tunnel Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Tunnel: Send + Sync {
    /// Human-readable name for this tunnel instance.
    fn name(&self) -> &str;

    /// The type of tunnel.
    fn tunnel_type(&self) -> TunnelType;

    /// Start the tunnel, returning the public URL.
    async fn start(&self) -> MvResult<String>;

    /// Stop the tunnel.
    async fn stop(&self) -> MvResult<()>;

    /// Get the current status of the tunnel.
    fn status(&self) -> TunnelStatus;

    /// Check if the tunnel process is still running.
    async fn health_check(&self) -> MvResult<bool>;
}

// ---------------------------------------------------------------------------
// Tunnel Registry
// ---------------------------------------------------------------------------

pub struct TunnelRegistry {
    tunnels: RwLock<HashMap<Uuid, Arc<dyn Tunnel>>>,
    configs: RwLock<Vec<TunnelConfig>>,
}

impl TunnelRegistry {
    pub fn new() -> Self {
        Self {
            tunnels: RwLock::new(HashMap::new()),
            configs: RwLock::new(Vec::new()),
        }
    }

    pub async fn register(&self, config: TunnelConfig, tunnel: Arc<dyn Tunnel>) {
        let id = config.id;
        self.configs.write().await.push(config);
        self.tunnels.write().await.insert(id, tunnel);
    }

    pub async fn remove(&self, id: Uuid) -> bool {
        let removed = self.tunnels.write().await.remove(&id).is_some();
        if removed {
            self.configs.write().await.retain(|c| c.id != id);
        }
        removed
    }

    pub async fn get(&self, id: Uuid) -> Option<Arc<dyn Tunnel>> {
        self.tunnels.read().await.get(&id).cloned()
    }

    pub async fn list_configs(&self) -> Vec<TunnelConfig> {
        self.configs.read().await.clone()
    }

    pub async fn list_statuses(&self) -> Vec<TunnelStatus> {
        let tunnels = self.tunnels.read().await;
        tunnels.values().map(|t| t.status()).collect()
    }
}

pub async fn bootstrap_tunnels(_engine: Arc<crate::engine::HelixEngine>) -> MvResult<()> {
    // TODO: Implement tunnel bootstrapping from config
    Ok(())
}

impl Default for TunnelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct MockTunnel {
        name: String,
        tunnel_type: TunnelType,
        running: Arc<Mutex<bool>>,
        url: Arc<Mutex<Option<String>>>,
        started_at: Arc<Mutex<Option<DateTime<Utc>>>>,
    }

    impl MockTunnel {
        fn new(name: &str, tt: TunnelType) -> Self {
            Self {
                name: name.to_string(),
                tunnel_type: tt,
                running: Arc::new(Mutex::new(false)),
                url: Arc::new(Mutex::new(None)),
                started_at: Arc::new(Mutex::new(None)),
            }
        }
    }

    #[async_trait]
    impl Tunnel for MockTunnel {
        fn name(&self) -> &str {
            &self.name
        }

        fn tunnel_type(&self) -> TunnelType {
            self.tunnel_type
        }

        async fn start(&self) -> MvResult<String> {
            let url = "https://mock.trycloudflare.com".to_string();
            *self.running.lock().unwrap() = true;
            *self.url.lock().unwrap() = Some(url.clone());
            *self.started_at.lock().unwrap() = Some(Utc::now());
            Ok(url)
        }

        async fn stop(&self) -> MvResult<()> {
            *self.running.lock().unwrap() = false;
            *self.url.lock().unwrap() = None;
            *self.started_at.lock().unwrap() = None;
            Ok(())
        }

        fn status(&self) -> TunnelStatus {
            TunnelStatus {
                tunnel_type: self.tunnel_type,
                name: self.name.clone(),
                running: *self.running.lock().unwrap(),
                public_url: self.url.lock().unwrap().clone(),
                error: None,
                started_at: *self.started_at.lock().unwrap(),
            }
        }

        async fn health_check(&self) -> MvResult<bool> {
            Ok(*self.running.lock().unwrap())
        }
    }

    // --- TunnelConfig tests ---

    #[test]
    fn tunnel_config_new_sets_defaults() {
        let config = TunnelConfig::new(TunnelType::Cloudflare, "cf-test", 9470);
        assert_eq!(config.tunnel_type, TunnelType::Cloudflare);
        assert_eq!(config.name, "cf-test");
        assert_eq!(config.local_port, 9470);
        assert!(config.settings.is_empty());
    }

    #[test]
    fn tunnel_config_with_setting_and_get() {
        let config = TunnelConfig::new(TunnelType::Ngrok, "ngrok-test", 8080)
            .with_setting("authtoken", "tok_123")
            .with_setting("region", "us");
        assert_eq!(config.get_setting("authtoken"), Some("tok_123"));
        assert_eq!(config.get_setting("region"), Some("us"));
        assert_eq!(config.get_setting("nonexistent"), None);
    }

    // --- TunnelType tests ---

    #[test]
    fn tunnel_type_display_and_from_str() {
        assert_eq!(TunnelType::Cloudflare.to_string(), "cloudflare");
        assert_eq!(TunnelType::Ngrok.to_string(), "ngrok");
        assert_eq!(TunnelType::Tailscale.to_string(), "tailscale");
        assert_eq!(TunnelType::Bore.to_string(), "bore");
        assert_eq!(TunnelType::Ssh.to_string(), "ssh");
        assert_eq!(TunnelType::Custom.to_string(), "custom");

        assert_eq!("cloudflare".parse::<TunnelType>().unwrap(), TunnelType::Cloudflare);
        assert_eq!("ngrok".parse::<TunnelType>().unwrap(), TunnelType::Ngrok);
        assert_eq!("ssh".parse::<TunnelType>().unwrap(), TunnelType::Ssh);
    }

    #[test]
    fn tunnel_type_from_str_rejects_unknown() {
        assert!("webhook".parse::<TunnelType>().is_err());
        assert!("Cloudflare".parse::<TunnelType>().is_err());
        assert!("".parse::<TunnelType>().is_err());
    }

    #[test]
    fn tunnel_type_as_str_matches_display() {
        for tt in [
            TunnelType::Cloudflare,
            TunnelType::Ngrok,
            TunnelType::Tailscale,
            TunnelType::Bore,
            TunnelType::Ssh,
            TunnelType::Custom,
        ] {
            assert_eq!(tt.as_str(), tt.to_string());
        }
    }

    #[test]
    fn tunnel_status_serialization() {
        let status = TunnelStatus {
            tunnel_type: TunnelType::Ngrok,
            name: "test-ngrok".into(),
            running: true,
            public_url: Some("https://abc.ngrok.io".into()),
            error: None,
            started_at: Some(Utc::now()),
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["tunnel_type"], "ngrok");
        assert_eq!(json["running"], true);
        assert_eq!(json["public_url"], "https://abc.ngrok.io");
    }

    // --- Registry tests ---

    #[tokio::test]
    async fn registry_register_and_get() {
        let registry = TunnelRegistry::new();
        let tunnel = Arc::new(MockTunnel::new("cf", TunnelType::Cloudflare));
        let config = TunnelConfig::new(TunnelType::Cloudflare, "cf", 9470);
        let id = config.id;

        registry.register(config, tunnel).await;
        assert!(registry.get(id).await.is_some());
    }

    #[tokio::test]
    async fn registry_remove() {
        let registry = TunnelRegistry::new();
        let tunnel = Arc::new(MockTunnel::new("rm", TunnelType::Bore));
        let config = TunnelConfig::new(TunnelType::Bore, "rm", 8080);
        let id = config.id;

        registry.register(config, tunnel).await;
        assert!(registry.remove(id).await);
        assert!(registry.get(id).await.is_none());
        assert!(!registry.remove(id).await);
    }

    #[tokio::test]
    async fn registry_list_configs_and_statuses() {
        let registry = TunnelRegistry::new();
        assert!(registry.list_configs().await.is_empty());
        assert!(registry.list_statuses().await.is_empty());

        let t1 = Arc::new(MockTunnel::new("t1", TunnelType::Cloudflare));
        let t2 = Arc::new(MockTunnel::new("t2", TunnelType::Ngrok));

        registry
            .register(TunnelConfig::new(TunnelType::Cloudflare, "t1", 9470), t1)
            .await;
        registry
            .register(TunnelConfig::new(TunnelType::Ngrok, "t2", 8080), t2)
            .await;

        assert_eq!(registry.list_configs().await.len(), 2);
        assert_eq!(registry.list_statuses().await.len(), 2);
    }

    #[tokio::test]
    async fn mock_tunnel_start_stop_lifecycle() {
        let tunnel = MockTunnel::new("lifecycle", TunnelType::Cloudflare);

        assert!(!tunnel.status().running);
        assert!(tunnel.status().public_url.is_none());

        let url = tunnel.start().await.unwrap();
        assert_eq!(url, "https://mock.trycloudflare.com");
        assert!(tunnel.status().running);
        assert!(tunnel.status().public_url.is_some());
        assert!(tunnel.health_check().await.unwrap());

        tunnel.stop().await.unwrap();
        assert!(!tunnel.status().running);
        assert!(tunnel.status().public_url.is_none());
        assert!(!tunnel.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn registry_get_nonexistent_returns_none() {
        let registry = TunnelRegistry::new();
        assert!(registry.get(Uuid::now_v7()).await.is_none());
    }

    #[tokio::test]
    async fn default_creates_empty_registry() {
        let registry = TunnelRegistry::default();
        assert!(registry.list_configs().await.is_empty());
        assert!(registry.list_statuses().await.is_empty());
    }
}
