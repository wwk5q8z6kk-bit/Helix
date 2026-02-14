use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum HxError {
    #[error("node not found: {0}")]
    NodeNotFound(Uuid),

    #[error("relationship not found: {0}")]
    RelationshipNotFound(Uuid),

    #[error("duplicate node: {0}")]
    DuplicateNode(Uuid),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("graph error: {0}")]
    Graph(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("migration error: {0}")]
    Migration(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("auth error: {0}")]
    Auth(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("keychain error: {0}")]
    Keychain(String),

    #[error("keychain: not found: {0}")]
    KeychainNotFound(String),

    #[error("keychain: not initialized")]
    KeychainNotInitialized,

    #[error("keychain: invalid password")]
    KeychainInvalidPassword,

    #[error("keychain: already initialized")]
    KeychainAlreadyInitialized,

    #[error("vault sealed: unlock the vault before accessing credentials")]
    VaultSealed,

    #[error("consumer error: {0}")]
    Consumer(String),

    #[error("access denied: {0}")]
    AccessDenied(String),

    #[error("proxy error: {0}")]
    Proxy(String),

    #[error("federation error: {0}")]
    Federation(String),
}

pub type MvResult<T> = Result<T, HxError>;
