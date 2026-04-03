//! # gRPC Authentication for Regime Bridge
//!
//! Provides bearer-token and optional mTLS authentication for the
//! `RegimeBridgeService` gRPC server.
//!
//! ## Bearer Token Authentication
//!
//! When `REGIME_GRPC_AUTH_TOKEN` is set, every incoming RPC must include
//! an `authorization` metadata header with the value `Bearer <token>`.
//! Requests without a valid token receive `UNAUTHENTICATED`.
//!
//! When the env var is unset the interceptor is a no-op pass-through,
//! preserving backward compatibility for development / trusted-network
//! deployments.
//!
//! ## Multi-Token Support & Token Rotation
//!
//! `REGIME_GRPC_AUTH_TOKEN` accepts a **comma-separated** list of tokens:
//!
//! ```text
//! REGIME_GRPC_AUTH_TOKEN=old-token,new-token
//! ```
//!
//! This enables zero-downtime token rotation:
//!   1. Add the new token to the env var alongside the old one.
//!   2. Restart the server (or call `add_token()` / `replace_tokens()` at runtime).
//!   3. Migrate clients to the new token.
//!   4. Remove the old token.
//!
//! Tokens can also be added/revoked/replaced at runtime without restart
//! via `AuthInterceptor::add_token()`, `revoke_token()`, and
//! `replace_tokens()`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::regime_bridge_auth::{AuthConfig, AuthInterceptor};
//! use tonic::transport::Server;
//!
//! let config = AuthConfig::from_env();
//! let interceptor = AuthInterceptor::new(config.clone());
//!
//! // Wrap your service with the interceptor
//! let svc = RegimeBridgeServiceServer::with_interceptor(server, interceptor);
//!
//! Server::builder()
//!     .add_service(svc)
//!     .serve(addr)
//!     .await?;
//! ```
//!
//! ## Runtime Token Rotation
//!
//! ```rust,ignore
//! // Add a new token (old tokens remain valid)
//! interceptor.add_token("new-secret-42");
//!
//! // Revoke an old token
//! interceptor.revoke_token("old-secret-41");
//!
//! // Atomically replace all tokens
//! interceptor.replace_tokens(vec!["fresh-token-1".into(), "fresh-token-2".into()]);
//! ```
//!
//! ## mTLS (Optional — requires `tls` feature)
//!
//! For mTLS, enable the `tls` crate feature and pass `tls_cert_path` /
//! `tls_key_path` / `tls_ca_cert_path` in [`AuthConfig`]. The caller
//! (e.g. `start_authenticated_grpc`) will use these to configure
//! `tonic::transport::ServerTlsConfig`.
//!
//! ## Environment Variables
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `REGIME_GRPC_AUTH_TOKEN` | *(unset — auth disabled)* | Shared secret token(s). Comma-separated for multi-token. Clients must send `Bearer <token>` in the `authorization` metadata header. |
//! | `REGIME_GRPC_TLS_CERT`  | *(unset)* | Path to PEM-encoded server certificate for mTLS (requires `tls` feature). |
//! | `REGIME_GRPC_TLS_KEY`   | *(unset)* | Path to PEM-encoded server private key for mTLS (requires `tls` feature). |
//! | `REGIME_GRPC_TLS_CA`    | *(unset)* | Path to PEM-encoded CA certificate for client verification (mTLS, requires `tls` feature). |

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Status};
use tracing::{debug, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Authentication configuration for the regime bridge gRPC server.
#[derive(Debug, Clone, Default)]
pub struct AuthConfig {
    /// Bearer token(s) for request authentication.
    /// When empty, token auth is disabled (all requests pass through).
    /// Multiple tokens are supported for zero-downtime rotation.
    pub bearer_tokens: Vec<String>,

    /// Path to PEM-encoded server certificate (for mTLS).
    /// Only used when the `tls` crate feature is enabled.
    pub tls_cert_path: Option<String>,

    /// Path to PEM-encoded server private key (for mTLS).
    /// Only used when the `tls` crate feature is enabled.
    pub tls_key_path: Option<String>,

    /// Path to PEM-encoded CA certificate for client verification (mTLS).
    /// Only used when the `tls` crate feature is enabled.
    pub tls_ca_cert_path: Option<String>,
}

impl AuthConfig {
    /// Load auth configuration from environment variables.
    ///
    /// - `REGIME_GRPC_AUTH_TOKEN` → `bearer_tokens` (comma-separated for multiple)
    /// - `REGIME_GRPC_TLS_CERT`  → `tls_cert_path`
    /// - `REGIME_GRPC_TLS_KEY`   → `tls_key_path`
    /// - `REGIME_GRPC_TLS_CA`    → `tls_ca_cert_path`
    ///
    /// ## Multi-Token Example
    ///
    /// ```bash
    /// REGIME_GRPC_AUTH_TOKEN=token-alpha,token-beta
    /// ```
    pub fn from_env() -> Self {
        let bearer_tokens: Vec<String> = std::env::var("REGIME_GRPC_AUTH_TOKEN")
            .ok()
            .map(|val| {
                val.split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let tls_cert_path = std::env::var("REGIME_GRPC_TLS_CERT").ok();
        let tls_key_path = std::env::var("REGIME_GRPC_TLS_KEY").ok();
        let tls_ca_cert_path = std::env::var("REGIME_GRPC_TLS_CA").ok();

        if !bearer_tokens.is_empty() {
            info!(
                "🔐 gRPC bearer-token authentication ENABLED ({} token{})",
                bearer_tokens.len(),
                if bearer_tokens.len() == 1 { "" } else { "s" }
            );
        } else {
            info!(
                "🔓 gRPC bearer-token authentication DISABLED (set REGIME_GRPC_AUTH_TOKEN to enable)"
            );
        }

        if tls_cert_path.is_some() && tls_key_path.is_some() {
            #[cfg(feature = "tls")]
            info!(
                "🔒 gRPC TLS ENABLED (cert={}, key={}{})",
                tls_cert_path.as_deref().unwrap_or("?"),
                tls_key_path.as_deref().unwrap_or("?"),
                if tls_ca_cert_path.is_some() {
                    ", mTLS with client CA verification"
                } else {
                    ", server-only TLS (no client cert required)"
                }
            );
            #[cfg(not(feature = "tls"))]
            warn!(
                "⚠️ TLS cert/key paths are set but the `tls` feature is not enabled — TLS will be ignored. \
                 Rebuild with `--features tls` to enable TLS support."
            );
        }

        Self {
            bearer_tokens,
            tls_cert_path,
            tls_key_path,
            tls_ca_cert_path,
        }
    }

    /// Create an auth config with a single bearer token (no TLS).
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            bearer_tokens: vec![token.into()],
            ..Default::default()
        }
    }

    /// Create an auth config with multiple bearer tokens (no TLS).
    pub fn with_tokens(tokens: Vec<String>) -> Self {
        Self {
            bearer_tokens: tokens,
            ..Default::default()
        }
    }

    /// Create an auth config with TLS paths.
    pub fn with_tls(
        cert_path: impl Into<String>,
        key_path: impl Into<String>,
        ca_cert_path: Option<String>,
    ) -> Self {
        Self {
            bearer_tokens: Vec::new(),
            tls_cert_path: Some(cert_path.into()),
            tls_key_path: Some(key_path.into()),
            tls_ca_cert_path: ca_cert_path,
        }
    }

    /// Whether bearer-token authentication is enabled.
    pub fn is_token_auth_enabled(&self) -> bool {
        !self.bearer_tokens.is_empty()
    }

    /// Whether TLS is configured (at minimum cert + key) **and** the `tls`
    /// feature is compiled in.
    pub fn is_tls_enabled(&self) -> bool {
        cfg!(feature = "tls") && self.tls_cert_path.is_some() && self.tls_key_path.is_some()
    }

    /// Whether mutual TLS (client certificate verification) is configured
    /// **and** the `tls` feature is compiled in.
    pub fn is_mtls_enabled(&self) -> bool {
        self.is_tls_enabled() && self.tls_ca_cert_path.is_some()
    }

    /// Backward-compatible getter: returns the first token (if any).
    ///
    /// Prefer `bearer_tokens` directly when working with multi-token logic.
    pub fn bearer_token(&self) -> Option<&str> {
        self.bearer_tokens.first().map(|s| s.as_str())
    }
}

// ============================================================================
// Bearer Token Interceptor (multi-token, hot-swappable)
// ============================================================================

/// gRPC interceptor that validates bearer tokens in the `authorization`
/// metadata header.
///
/// Implements `tonic::service::Interceptor` so it can be used with
/// `RegimeBridgeServiceServer::with_interceptor()`.
///
/// When no tokens are configured, all requests pass through (no-op mode).
///
/// ## Multi-Token & Runtime Rotation
///
/// The interceptor holds tokens behind an `Arc<RwLock<HashSet<String>>>`,
/// which allows:
/// - Multiple valid tokens simultaneously (for rotation windows)
/// - Atomic add/revoke/replace at runtime without server restart
/// - Thread-safe access from concurrent gRPC handler tasks
#[derive(Clone)]
pub struct AuthInterceptor {
    inner: Arc<InterceptorInner>,
}

struct InterceptorInner {
    /// Set of valid tokens. Empty = auth disabled (pass-through).
    /// Protected by `RwLock` for runtime hot-swap without restart.
    tokens: RwLock<HashSet<String>>,

    /// Whether auth was enabled at construction time.
    /// Used to distinguish "no tokens configured" from "all tokens revoked".
    auth_enabled: std::sync::atomic::AtomicBool,

    /// Counter for rejected requests (for observability).
    rejected_count: std::sync::atomic::AtomicU64,

    /// Counter for accepted requests.
    accepted_count: std::sync::atomic::AtomicU64,

    /// Counter for token rotation operations.
    rotation_count: std::sync::atomic::AtomicU64,
}

// Manual Debug to avoid leaking the tokens
impl std::fmt::Debug for AuthInterceptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // We can't async-block in Debug, so we check whether auth is enabled
        // via the atomic flag instead of locking the RwLock.
        f.debug_struct("AuthInterceptor")
            .field(
                "auth_enabled",
                &self
                    .inner
                    .auth_enabled
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "rejected",
                &self
                    .inner
                    .rejected_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "accepted",
                &self
                    .inner
                    .accepted_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "rotations",
                &self
                    .inner
                    .rotation_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl AuthInterceptor {
    /// Create a new interceptor from an [`AuthConfig`].
    ///
    /// If `config.bearer_tokens` is empty, the interceptor passes all
    /// requests through without checking.
    pub fn new(config: AuthConfig) -> Self {
        let enabled = !config.bearer_tokens.is_empty();
        let token_set: HashSet<String> = config.bearer_tokens.into_iter().collect();

        Self {
            inner: Arc::new(InterceptorInner {
                tokens: RwLock::new(token_set),
                auth_enabled: std::sync::atomic::AtomicBool::new(enabled),
                rejected_count: std::sync::atomic::AtomicU64::new(0),
                accepted_count: std::sync::atomic::AtomicU64::new(0),
                rotation_count: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Create a no-op interceptor that allows all requests.
    pub fn allow_all() -> Self {
        Self {
            inner: Arc::new(InterceptorInner {
                tokens: RwLock::new(HashSet::new()),
                auth_enabled: std::sync::atomic::AtomicBool::new(false),
                rejected_count: std::sync::atomic::AtomicU64::new(0),
                accepted_count: std::sync::atomic::AtomicU64::new(0),
                rotation_count: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Create an interceptor that requires a specific token.
    pub fn with_token(token: impl Into<String>) -> Self {
        let mut set = HashSet::new();
        set.insert(token.into());

        Self {
            inner: Arc::new(InterceptorInner {
                tokens: RwLock::new(set),
                auth_enabled: std::sync::atomic::AtomicBool::new(true),
                rejected_count: std::sync::atomic::AtomicU64::new(0),
                accepted_count: std::sync::atomic::AtomicU64::new(0),
                rotation_count: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Create an interceptor with multiple valid tokens.
    pub fn with_tokens(tokens: Vec<String>) -> Self {
        let enabled = !tokens.is_empty();
        let set: HashSet<String> = tokens.into_iter().collect();

        Self {
            inner: Arc::new(InterceptorInner {
                tokens: RwLock::new(set),
                auth_enabled: std::sync::atomic::AtomicBool::new(enabled),
                rejected_count: std::sync::atomic::AtomicU64::new(0),
                accepted_count: std::sync::atomic::AtomicU64::new(0),
                rotation_count: std::sync::atomic::AtomicU64::new(0),
            }),
        }
    }

    /// Add a new valid token at runtime (for token rotation).
    ///
    /// Returns `true` if the token was newly inserted, `false` if it already existed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // During token rotation: add the new token before revoking the old
    /// interceptor.add_token("new-secret-v2").await;
    /// // ... migrate clients ...
    /// interceptor.revoke_token("old-secret-v1").await;
    /// ```
    pub async fn add_token(&self, token: impl Into<String>) -> bool {
        let token = token.into();
        let mut tokens = self.inner.tokens.write().await;
        let inserted = tokens.insert(token);
        if inserted {
            self.inner
                .auth_enabled
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.inner
                .rotation_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info!(
                "🔑 Token added — {} active token{}",
                tokens.len(),
                if tokens.len() == 1 { "" } else { "s" }
            );
        }
        inserted
    }

    /// Revoke a token at runtime.
    ///
    /// Returns `true` if the token was found and removed, `false` if it
    /// was not present.
    ///
    /// If all tokens are revoked, auth becomes a pass-through (disabled).
    pub async fn revoke_token(&self, token: &str) -> bool {
        let mut tokens = self.inner.tokens.write().await;
        let removed = tokens.remove(token);
        if removed {
            self.inner
                .rotation_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if tokens.is_empty() {
                self.inner
                    .auth_enabled
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                warn!("⚠️ All tokens revoked — auth is now DISABLED (all requests will pass)");
            } else {
                info!(
                    "🔑 Token revoked — {} active token{} remaining",
                    tokens.len(),
                    if tokens.len() == 1 { "" } else { "s" }
                );
            }
        }
        removed
    }

    /// Atomically replace all tokens with a new set.
    ///
    /// This is the safest rotation method — clients using the old token
    /// will be rejected immediately after this call.
    ///
    /// If `new_tokens` is empty, auth is disabled.
    pub async fn replace_tokens(&self, new_tokens: Vec<String>) {
        let enabled = !new_tokens.is_empty();
        let count = new_tokens.len();
        let new_set: HashSet<String> = new_tokens.into_iter().collect();

        let mut tokens = self.inner.tokens.write().await;
        *tokens = new_set;
        self.inner
            .auth_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self.inner
            .rotation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if enabled {
            info!(
                "🔑 Tokens replaced — {} active token{}",
                count,
                if count == 1 { "" } else { "s" }
            );
        } else {
            warn!("⚠️ Tokens replaced with empty set — auth is now DISABLED");
        }
    }

    /// Number of currently active tokens.
    ///
    /// Note: this acquires a read lock. In hot paths, prefer checking
    /// `is_enabled()` which uses an atomic.
    pub async fn active_token_count(&self) -> usize {
        self.inner.tokens.read().await.len()
    }

    /// Number of requests rejected due to auth failure.
    pub fn rejected_count(&self) -> u64 {
        self.inner
            .rejected_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of requests that passed auth.
    pub fn accepted_count(&self) -> u64 {
        self.inner
            .accepted_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of token rotation operations (add/revoke/replace).
    pub fn rotation_count(&self) -> u64 {
        self.inner
            .rotation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether token auth is active.
    ///
    /// Uses an atomic flag — no lock contention. Returns `false` if
    /// no tokens are configured or all have been revoked.
    pub fn is_enabled(&self) -> bool {
        self.inner
            .auth_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Synchronous token validation used by the `Interceptor` impl.
    ///
    /// Because `tonic::service::Interceptor::call` is synchronous, we
    /// use `try_read()` to avoid blocking. If the lock is contended
    /// (extremely unlikely — only during active rotation), we fall back
    /// to rejecting with `UNAVAILABLE` so the client retries.
    fn validate_token_sync(&self, provided: &str) -> Result<(), Status> {
        let tokens = match self.inner.tokens.try_read() {
            Ok(guard) => guard,
            Err(_) => {
                // Lock is held by a writer (token rotation in progress).
                // This is a transient condition — tell the client to retry.
                return Err(Status::unavailable(
                    "Auth token store is being updated — retry shortly",
                ));
            }
        };

        if tokens.is_empty() {
            // Auth was enabled but all tokens were revoked at runtime.
            // Pass through (same as disabled).
            self.inner
                .accepted_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }

        // Check the provided token against all valid tokens using
        // constant-time comparison to prevent timing attacks.
        //
        // We iterate ALL tokens regardless of match to keep timing uniform.
        let mut matched = false;
        for expected in tokens.iter() {
            if constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                matched = true;
                // Don't break — keep iterating for constant-time behavior
            }
        }

        if matched {
            debug!("✅ gRPC auth: token validated");
            self.inner
                .accepted_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        } else {
            warn!("🚫 gRPC auth: invalid token");
            self.inner
                .rejected_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(Status::unauthenticated("Invalid authentication token"))
        }
    }
}

impl tonic::service::Interceptor for AuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        // Fast path: auth disabled (no lock needed)
        if !self
            .inner
            .auth_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.inner
                .accepted_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(request);
        }

        // Extract the `authorization` metadata header
        let auth_header = request.metadata().get("authorization");

        match auth_header {
            Some(value) => {
                let value_str = value.to_str().map_err(|_| {
                    warn!("🚫 gRPC auth: non-ASCII authorization header");
                    self.inner
                        .rejected_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Status::unauthenticated("Invalid authorization header encoding")
                })?;

                // Expect "Bearer <token>" format (case-insensitive prefix)
                let provided_token = if let Some(token) = value_str.strip_prefix("Bearer ") {
                    token
                } else if let Some(token) = value_str.strip_prefix("bearer ") {
                    token
                } else {
                    warn!("🚫 gRPC auth: authorization header missing Bearer prefix");
                    self.inner
                        .rejected_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(Status::unauthenticated(
                        "Authorization header must use Bearer scheme",
                    ));
                };

                // Validate against all configured tokens
                self.validate_token_sync(provided_token)?;
                Ok(request)
            }
            None => {
                warn!("🚫 gRPC auth: missing authorization header");
                self.inner
                    .rejected_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(Status::unauthenticated(
                    "Missing authorization header — expected: Bearer <token>",
                ))
            }
        }
    }
}

// ============================================================================
// TLS Configuration Builder (requires `tls` feature)
// ============================================================================

/// Build a `tonic::transport::ServerTlsConfig` from the [`AuthConfig`].
///
/// Returns `None` if TLS is not configured (no cert/key paths) or if the
/// `tls` feature is not enabled.
///
/// # Errors
///
/// Returns an error if the certificate or key files cannot be read.
#[cfg(feature = "tls")]
pub async fn build_tls_config(
    config: &AuthConfig,
) -> Result<Option<tonic::transport::ServerTlsConfig>, anyhow::Error> {
    let (cert_path, key_path) = match (&config.tls_cert_path, &config.tls_key_path) {
        (Some(cert), Some(key)) => (cert, key),
        _ => return Ok(None),
    };

    let cert_pem = tokio::fs::read(cert_path).await.map_err(|e| {
        anyhow::anyhow!("Failed to read TLS certificate from '{}': {}", cert_path, e)
    })?;

    let key_pem = tokio::fs::read(key_path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read TLS key from '{}': {}", key_path, e))?;

    let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);

    let mut tls_config = tonic::transport::ServerTlsConfig::new().identity(identity);

    // If a CA certificate is provided, enable client certificate verification (mTLS)
    if let Some(ca_path) = &config.tls_ca_cert_path {
        let ca_pem = tokio::fs::read(ca_path).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to read TLS CA certificate from '{}': {}",
                ca_path,
                e
            )
        })?;

        let ca_cert = tonic::transport::Certificate::from_pem(ca_pem);
        tls_config = tls_config.client_ca_root(ca_cert);

        info!(
            "🔒 mTLS configured: clients must present a certificate signed by CA at '{}'",
            ca_path
        );
    }

    Ok(Some(tls_config))
}

/// Stub for when the `tls` feature is not enabled — always returns `Ok(None)`.
#[cfg(not(feature = "tls"))]
pub async fn build_tls_config(_config: &AuthConfig) -> Result<Option<()>, anyhow::Error> {
    Ok(None)
}

// ============================================================================
// Client-side auth helpers
// ============================================================================

/// Build a `tonic::transport::Channel` with optional TLS and bearer token.
///
/// This is a convenience helper for consumers / neuromorphic clients
/// that need to connect to an authenticated `RegimeBridgeService`.
///
/// When the `tls` feature is not enabled, TLS-related parameters are
/// accepted but ignored (plain TCP connection is used).
#[cfg(feature = "tls")]
pub async fn build_authenticated_channel(
    endpoint: &str,
    _bearer_token: Option<&str>,
    tls_ca_cert_path: Option<&str>,
    tls_client_cert_path: Option<&str>,
    tls_client_key_path: Option<&str>,
) -> Result<tonic::transport::Channel, anyhow::Error> {
    let mut ep = tonic::transport::Channel::from_shared(endpoint.to_string())
        .map_err(|e| anyhow::anyhow!("Invalid endpoint '{}': {}", endpoint, e))?;

    // Configure client-side TLS if CA cert provided
    if let Some(ca_path) = tls_ca_cert_path {
        let ca_pem = tokio::fs::read(ca_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read CA cert from '{}': {}", ca_path, e))?;

        let mut tls_config = tonic::transport::ClientTlsConfig::new()
            .ca_certificate(tonic::transport::Certificate::from_pem(ca_pem));

        // If client cert + key provided, configure mTLS client identity
        if let (Some(cert_path), Some(key_path)) = (tls_client_cert_path, tls_client_key_path) {
            let cert_pem = tokio::fs::read(cert_path).await?;
            let key_pem = tokio::fs::read(key_path).await?;
            let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);
            tls_config = tls_config.identity(identity);
        }

        ep = ep.tls_config(tls_config)?;
    }

    let channel = ep.connect().await?;

    // Note: bearer token injection is done per-request via metadata, not on
    // the channel itself. Use `inject_auth_metadata()` when making requests.
    if _bearer_token.is_some() {
        info!("🔑 gRPC client: bearer token configured (will be injected per-request)");
    }

    Ok(channel)
}

/// Build a plain (non-TLS) channel when the `tls` feature is not enabled.
///
/// TLS-related parameters are accepted but ignored.
#[cfg(not(feature = "tls"))]
pub async fn build_authenticated_channel(
    endpoint: &str,
    _bearer_token: Option<&str>,
    _tls_ca_cert_path: Option<&str>,
    _tls_client_cert_path: Option<&str>,
    _tls_client_key_path: Option<&str>,
) -> Result<tonic::transport::Channel, anyhow::Error> {
    if _tls_ca_cert_path.is_some() {
        warn!(
            "⚠️ TLS CA cert path provided but `tls` feature is not enabled — connecting without TLS"
        );
    }

    let ep = tonic::transport::Channel::from_shared(endpoint.to_string())
        .map_err(|e| anyhow::anyhow!("Invalid endpoint '{}': {}", endpoint, e))?;

    let channel = ep.connect().await?;

    if _bearer_token.is_some() {
        info!("🔑 gRPC client: bearer token configured (will be injected per-request)");
    }

    Ok(channel)
}

/// Inject `authorization: Bearer <token>` into a tonic `Request`'s metadata.
///
/// Call this on each outbound request before sending.
///
/// ```rust,ignore
/// let mut request = tonic::Request::new(payload);
/// inject_auth_metadata(&mut request, "my-secret-token")?;
/// let response = client.push_regime_state(request).await?;
/// ```
pub fn inject_auth_metadata<T>(request: &mut Request<T>, token: &str) -> Result<(), anyhow::Error> {
    let header_value = format!("Bearer {}", token);
    let metadata_value: tonic::metadata::MetadataValue<tonic::metadata::Ascii> = header_value
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid token for metadata header: {}", e))?;

    request
        .metadata_mut()
        .insert("authorization", metadata_value);
    Ok(())
}

// ============================================================================
// Utilities
// ============================================================================

/// Constant-time byte comparison to prevent timing side-channel attacks on
/// token validation.
///
/// Returns `true` if both slices are equal in length and content.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mutex to serialize tests that manipulate environment variables.
    /// `std::env::set_var` / `remove_var` affect the whole process, so
    /// parallel tests that touch the same env vars will race.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // ── AuthConfig tests ────────────────────────────────────────────────

    #[test]
    fn test_auth_config_default_is_disabled() {
        let config = AuthConfig::default();
        assert!(!config.is_token_auth_enabled());
        assert!(!config.is_tls_enabled());
        assert!(!config.is_mtls_enabled());
        assert!(config.bearer_tokens.is_empty());
        assert!(config.bearer_token().is_none());
    }

    #[test]
    fn test_auth_config_with_token() {
        let config = AuthConfig::with_token("secret-42");
        assert!(config.is_token_auth_enabled());
        assert_eq!(config.bearer_token(), Some("secret-42"));
        assert_eq!(config.bearer_tokens.len(), 1);
        assert!(!config.is_tls_enabled());
    }

    #[test]
    fn test_auth_config_with_tokens_multiple() {
        let config = AuthConfig::with_tokens(vec![
            "token-alpha".into(),
            "token-beta".into(),
            "token-gamma".into(),
        ]);
        assert!(config.is_token_auth_enabled());
        assert_eq!(config.bearer_tokens.len(), 3);
        // bearer_token() returns the first one
        assert_eq!(config.bearer_token(), Some("token-alpha"));
    }

    #[test]
    fn test_auth_config_with_tokens_empty() {
        let config = AuthConfig::with_tokens(vec![]);
        assert!(!config.is_token_auth_enabled());
        assert!(config.bearer_token().is_none());
    }

    #[test]
    fn test_auth_config_with_tls_no_ca() {
        let config = AuthConfig::with_tls("/tmp/cert.pem", "/tmp/key.pem", None);
        assert!(!config.is_token_auth_enabled());
        // is_tls_enabled depends on feature flag
        if cfg!(feature = "tls") {
            assert!(config.is_tls_enabled());
            assert!(!config.is_mtls_enabled());
        } else {
            assert!(!config.is_tls_enabled());
        }
    }

    #[test]
    fn test_auth_config_with_mtls() {
        let config = AuthConfig::with_tls(
            "/tmp/cert.pem",
            "/tmp/key.pem",
            Some("/tmp/ca.pem".to_string()),
        );
        if cfg!(feature = "tls") {
            assert!(config.is_tls_enabled());
            assert!(config.is_mtls_enabled());
        } else {
            assert!(!config.is_tls_enabled());
            assert!(!config.is_mtls_enabled());
        }
    }

    #[test]
    fn test_auth_config_from_env_defaults() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Clear any env vars that might be set (serialized by ENV_MUTEX)
        unsafe { std::env::remove_var("REGIME_GRPC_AUTH_TOKEN") };
        unsafe { std::env::remove_var("REGIME_GRPC_TLS_CERT") };
        unsafe { std::env::remove_var("REGIME_GRPC_TLS_KEY") };
        unsafe { std::env::remove_var("REGIME_GRPC_TLS_CA") };

        let config = AuthConfig::from_env();
        assert!(!config.is_token_auth_enabled());
        assert!(!config.is_tls_enabled());
    }

    #[test]
    fn test_auth_config_from_env_single_token() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe { std::env::set_var("REGIME_GRPC_AUTH_TOKEN", "single-secret") };
        let config = AuthConfig::from_env();
        assert!(config.is_token_auth_enabled());
        assert_eq!(config.bearer_tokens.len(), 1);
        assert!(config.bearer_tokens.contains(&"single-secret".to_string()));
        unsafe { std::env::remove_var("REGIME_GRPC_AUTH_TOKEN") };
    }

    #[test]
    fn test_auth_config_from_env_comma_separated_tokens() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe { std::env::set_var("REGIME_GRPC_AUTH_TOKEN", "token-a, token-b ,token-c") };
        let config = AuthConfig::from_env();
        assert!(config.is_token_auth_enabled());
        assert_eq!(config.bearer_tokens.len(), 3);
        assert!(config.bearer_tokens.contains(&"token-a".to_string()));
        assert!(config.bearer_tokens.contains(&"token-b".to_string()));
        assert!(config.bearer_tokens.contains(&"token-c".to_string()));
        unsafe { std::env::remove_var("REGIME_GRPC_AUTH_TOKEN") };
    }

    #[test]
    fn test_auth_config_from_env_ignores_empty_segments() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe { std::env::set_var("REGIME_GRPC_AUTH_TOKEN", "token-a,,, ,token-b,") };
        let config = AuthConfig::from_env();
        assert_eq!(config.bearer_tokens.len(), 2);
        assert!(config.bearer_tokens.contains(&"token-a".to_string()));
        assert!(config.bearer_tokens.contains(&"token-b".to_string()));
        unsafe { std::env::remove_var("REGIME_GRPC_AUTH_TOKEN") };
    }

    // ── AuthInterceptor tests ───────────────────────────────────────────

    #[test]
    fn test_interceptor_allow_all_passes_everything() {
        let mut interceptor = AuthInterceptor::allow_all();
        assert!(!interceptor.is_enabled());

        let request = Request::new(());
        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_ok());
        assert_eq!(interceptor.accepted_count(), 1);
        assert_eq!(interceptor.rejected_count(), 0);
    }

    #[test]
    fn test_interceptor_rejects_missing_header() {
        let mut interceptor = AuthInterceptor::with_token("my-secret");
        assert!(interceptor.is_enabled());

        let request = Request::new(());
        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert!(status.message().contains("Missing authorization header"));
        assert_eq!(interceptor.rejected_count(), 1);
        assert_eq!(interceptor.accepted_count(), 0);
    }

    #[test]
    fn test_interceptor_accepts_valid_token() {
        let mut interceptor = AuthInterceptor::with_token("correct-token");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", "Bearer correct-token".parse().unwrap());

        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_ok());
        assert_eq!(interceptor.accepted_count(), 1);
        assert_eq!(interceptor.rejected_count(), 0);
    }

    #[test]
    fn test_interceptor_rejects_wrong_token() {
        let mut interceptor = AuthInterceptor::with_token("correct-token");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", "Bearer wrong-token".parse().unwrap());

        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert!(status.message().contains("Invalid authentication token"));
        assert_eq!(interceptor.rejected_count(), 1);
    }

    #[test]
    fn test_interceptor_rejects_missing_bearer_prefix() {
        let mut interceptor = AuthInterceptor::with_token("my-secret");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", "Token my-secret".parse().unwrap());

        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert!(status.message().contains("Bearer scheme"));
    }

    #[test]
    fn test_interceptor_accepts_lowercase_bearer() {
        let mut interceptor = AuthInterceptor::with_token("my-secret");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", "bearer my-secret".parse().unwrap());

        let result = tonic::service::Interceptor::call(&mut interceptor, request);
        assert!(result.is_ok());
        assert_eq!(interceptor.accepted_count(), 1);
    }

    #[test]
    fn test_interceptor_counts_accumulate() {
        let mut interceptor = AuthInterceptor::with_token("token123");

        // 3 valid requests
        for _ in 0..3 {
            let mut request = Request::new(());
            request
                .metadata_mut()
                .insert("authorization", "Bearer token123".parse().unwrap());
            let _ = tonic::service::Interceptor::call(&mut interceptor, request);
        }

        // 2 invalid requests
        for _ in 0..2 {
            let request = Request::new(());
            let _ = tonic::service::Interceptor::call(&mut interceptor, request);
        }

        assert_eq!(interceptor.accepted_count(), 3);
        assert_eq!(interceptor.rejected_count(), 2);
    }

    #[test]
    fn test_interceptor_is_clone() {
        let interceptor = AuthInterceptor::with_token("abc");
        let cloned = interceptor.clone();

        // Clones share the same inner Arc — counters are shared
        assert!(cloned.is_enabled());
    }

    #[test]
    fn test_interceptor_debug_does_not_leak_token() {
        let interceptor = AuthInterceptor::with_token("super-secret-value");
        let debug_str = format!("{:?}", interceptor);

        assert!(
            !debug_str.contains("super-secret-value"),
            "Debug output must not contain the token! Got: {}",
            debug_str
        );
        assert!(
            debug_str.contains("auth_enabled: true"),
            "Debug should indicate auth is enabled. Got: {}",
            debug_str
        );
    }

    #[test]
    fn test_interceptor_debug_shows_rotation_count() {
        let interceptor = AuthInterceptor::with_token("abc");
        let debug_str = format!("{:?}", interceptor);
        assert!(
            debug_str.contains("rotations: 0"),
            "Debug should show rotation count. Got: {}",
            debug_str
        );
    }

    // ── Multi-token interceptor tests ───────────────────────────────────

    #[test]
    fn test_interceptor_multi_token_accepts_any_valid() {
        let mut interceptor =
            AuthInterceptor::with_tokens(vec!["token-a".into(), "token-b".into()]);

        // Token A should work
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer token-a".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut interceptor, req).is_ok());

        // Token B should also work
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer token-b".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut interceptor, req).is_ok());

        assert_eq!(interceptor.accepted_count(), 2);
    }

    #[test]
    fn test_interceptor_multi_token_rejects_invalid() {
        let mut interceptor =
            AuthInterceptor::with_tokens(vec!["token-a".into(), "token-b".into()]);

        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer token-c".parse().unwrap());
        let result = tonic::service::Interceptor::call(&mut interceptor, req);
        assert!(result.is_err());
        assert_eq!(interceptor.rejected_count(), 1);
    }

    // ── Runtime token rotation tests ────────────────────────────────────

    #[tokio::test]
    async fn test_add_token_enables_auth() {
        let interceptor = AuthInterceptor::allow_all();
        assert!(!interceptor.is_enabled());

        let inserted = interceptor.add_token("new-secret").await;
        assert!(inserted);
        assert!(interceptor.is_enabled());
        assert_eq!(interceptor.active_token_count().await, 1);
        assert_eq!(interceptor.rotation_count(), 1);
    }

    #[tokio::test]
    async fn test_add_token_duplicate_returns_false() {
        let interceptor = AuthInterceptor::with_token("existing");
        let inserted = interceptor.add_token("existing").await;
        assert!(!inserted);
        assert_eq!(interceptor.active_token_count().await, 1);
        assert_eq!(interceptor.rotation_count(), 0);
    }

    #[tokio::test]
    async fn test_revoke_token_removes_it() {
        let interceptor = AuthInterceptor::with_tokens(vec!["token-a".into(), "token-b".into()]);

        let removed = interceptor.revoke_token("token-a").await;
        assert!(removed);
        assert!(interceptor.is_enabled()); // token-b still active
        assert_eq!(interceptor.active_token_count().await, 1);
        assert_eq!(interceptor.rotation_count(), 1);

        // Verify token-a no longer works
        let mut int_clone = interceptor.clone();
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer token-a".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_clone, req).is_err());

        // Verify token-b still works
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer token-b".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_clone, req).is_ok());
    }

    #[tokio::test]
    async fn test_revoke_all_disables_auth() {
        let interceptor = AuthInterceptor::with_token("only-token");
        assert!(interceptor.is_enabled());

        interceptor.revoke_token("only-token").await;
        assert!(!interceptor.is_enabled());
        assert_eq!(interceptor.active_token_count().await, 0);
    }

    #[tokio::test]
    async fn test_revoke_nonexistent_returns_false() {
        let interceptor = AuthInterceptor::with_token("real-token");
        let removed = interceptor.revoke_token("fake-token").await;
        assert!(!removed);
        assert_eq!(interceptor.rotation_count(), 0);
    }

    #[tokio::test]
    async fn test_replace_tokens_atomically() {
        let interceptor = AuthInterceptor::with_tokens(vec!["old-a".into(), "old-b".into()]);

        interceptor
            .replace_tokens(vec!["new-x".into(), "new-y".into()])
            .await;

        assert!(interceptor.is_enabled());
        assert_eq!(interceptor.active_token_count().await, 2);
        assert_eq!(interceptor.rotation_count(), 1);

        // Old tokens should be rejected
        let mut int_clone = interceptor.clone();
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer old-a".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_clone, req).is_err());

        // New tokens should be accepted
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer new-x".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_clone, req).is_ok());
    }

    #[tokio::test]
    async fn test_replace_tokens_with_empty_disables() {
        let interceptor = AuthInterceptor::with_token("token");
        interceptor.replace_tokens(vec![]).await;
        assert!(!interceptor.is_enabled());
    }

    #[tokio::test]
    async fn test_rotation_workflow_zero_downtime() {
        // Simulate a zero-downtime token rotation:
        // 1. Start with old token
        // 2. Add new token (both valid)
        // 3. Revoke old token (only new valid)

        let interceptor = AuthInterceptor::with_token("v1-secret");

        // Step 1: v1 works
        let mut int_c = interceptor.clone();
        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer v1-secret".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_c, req).is_ok());

        // Step 2: Add v2 — both work
        interceptor.add_token("v2-secret").await;

        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer v1-secret".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_c, req).is_ok());

        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer v2-secret".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_c, req).is_ok());

        // Step 3: Revoke v1 — only v2 works
        interceptor.revoke_token("v1-secret").await;

        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer v1-secret".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_c, req).is_err());

        let mut req = Request::new(());
        req.metadata_mut()
            .insert("authorization", "Bearer v2-secret".parse().unwrap());
        assert!(tonic::service::Interceptor::call(&mut int_c, req).is_ok());

        // Verify counts
        assert_eq!(interceptor.rotation_count(), 2); // 1 add + 1 revoke
    }

    // ── Constant-time comparison tests ──────────────────────────────────

    #[test]
    fn test_constant_time_eq_equal() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"a", b"a"));
    }

    #[test]
    fn test_constant_time_eq_not_equal() {
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(b"ab", b"abc"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }

    // ── inject_auth_metadata tests ──────────────────────────────────────

    #[test]
    fn test_inject_auth_metadata_adds_header() {
        let mut request = Request::new(());
        inject_auth_metadata(&mut request, "my-token-42").unwrap();

        let header = request.metadata().get("authorization").unwrap();
        assert_eq!(header.to_str().unwrap(), "Bearer my-token-42");
    }

    #[test]
    fn test_inject_auth_metadata_overwrites_existing() {
        let mut request = Request::new(());
        inject_auth_metadata(&mut request, "first-token").unwrap();
        inject_auth_metadata(&mut request, "second-token").unwrap();

        let header = request.metadata().get("authorization").unwrap();
        assert_eq!(header.to_str().unwrap(), "Bearer second-token");
    }

    // ── build_tls_config tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_build_tls_config_returns_none_when_not_configured() {
        let config = AuthConfig::default();
        let result = build_tls_config(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_build_tls_config_returns_none_when_partial() {
        // Only cert, no key
        let config = AuthConfig {
            tls_cert_path: Some("/tmp/cert.pem".to_string()),
            ..Default::default()
        };
        let result = build_tls_config(&config).await.unwrap();
        assert!(result.is_none());
    }

    #[cfg(feature = "tls")]
    #[tokio::test]
    async fn test_build_tls_config_errors_on_missing_files() {
        let config = AuthConfig::with_tls("/nonexistent/cert.pem", "/nonexistent/key.pem", None);
        let result = build_tls_config(&config).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to read TLS certificate")
        );
    }
}
