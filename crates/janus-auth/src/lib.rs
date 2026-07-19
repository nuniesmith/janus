//! Bearer-token auth for janus HTTP surfaces.
//!
//! janus exposes two HTTP surfaces on the tailnet (`tailscale serve`): the
//! forward REST server (risk config, signal generation, brain kill-switch …)
//! and the janus-api gateway (service start/stop, config PUT, position ingress
//! …). Reads are a deliberately open posture — every device on Jordan's tailnet
//! may GET signals, health, metrics and WS streams. **Mutations were not.**
//! Until this crate, `PUT /api/v1/risk/config` — the knob that loosens the
//! armed bot's risk limits — was writable by any tailnet device with zero auth.
//!
//! This middleware closes that hole at a single, unit-tested seam: every
//! mutating request (any method that is not GET / HEAD / OPTIONS) must present
//! `Authorization: Bearer <JANUS_API_TOKEN>`. GET / HEAD / OPTIONS pass
//! untouched, which keeps `/health`, `/metrics`, all read endpoints and WS
//! upgrades (an HTTP GET) open by construction — no path allowlist needed.
//!
//! ## Fail posture when `JANUS_API_TOKEN` is unset
//!
//! **Fail closed.** With no token configured the server cannot authenticate
//! anyone, so mutating requests are denied `503 Service Unavailable` and a
//! prominent warning is logged once at startup. This follows the
//! user-management design's governing principle — *"Fail loud, not open:
//! absence of configuration must never be a silent bypass"* (design
//! `§4.0`) — and is the security-correct default for this crate specifically:
//! the whole point of the token is to close a live write hole, so the failure
//! mode of a *misconfigured* deploy must not be to leave that hole open.
//! Reads stay open in every posture, so health/metrics/observability and the
//! Gate-A read paths never depend on the token being set.
//!
//! The token is compared in constant time (SHA-256 digests + [`subtle`]) and is
//! never logged.

use axum::{
    extract::{Request, State},
    http::{Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

const TOKEN_ENV: &str = "JANUS_API_TOKEN";

/// Auth posture, resolved once from the environment at server-construction time.
///
/// Cloned into each request via axum's middleware state; `Enforcing` holds only
/// the SHA-256 digest of the token, never the token bytes themselves.
#[derive(Clone)]
pub enum Posture {
    /// `JANUS_API_TOKEN` is set: enforce bearer on every mutating request.
    Enforcing { token_digest: [u8; 32] },
    /// `JANUS_API_TOKEN` is unset/empty: deny mutating requests (fail closed).
    Unconfigured,
}

/// Per-request decision produced by [`decide`]. Pure and exhaustively tested;
/// the axum layer is a thin translation of this into a [`Response`].
#[derive(Debug, PartialEq, Eq)]
pub enum Decision {
    /// Non-mutating method, or a valid bearer: let the request through.
    Allow,
    /// Mutating request with a missing/invalid bearer while enforcing → 401.
    Unauthorized,
    /// Mutating request while the server has no token configured → 503.
    Misconfigured,
}

impl Posture {
    /// Resolve the posture from `JANUS_API_TOKEN`. An empty or whitespace-only
    /// value is treated as unset (fail closed). Does not log — call
    /// [`Posture::log_startup`] once after construction.
    pub fn from_env() -> Self {
        match std::env::var(TOKEN_ENV) {
            Ok(tok) if !tok.trim().is_empty() => Self::from_token(tok.as_bytes()),
            _ => Posture::Unconfigured,
        }
    }

    /// Build an `Enforcing` posture from a raw token (test/constructor helper).
    pub fn from_token(token: &[u8]) -> Self {
        Posture::Enforcing {
            token_digest: sha256(token),
        }
    }

    /// Emit a one-line startup log describing the posture for `surface`
    /// (e.g. `"forward-rest"`). Never logs the token. The `Unconfigured` case
    /// logs at WARN so a misconfigured deploy is loud in the container log.
    pub fn log_startup(&self, surface: &str) {
        match self {
            Posture::Enforcing { .. } => tracing::info!(
                surface,
                "janus-auth: bearer enforcement ENABLED on mutating routes ({TOKEN_ENV} set)"
            ),
            Posture::Unconfigured => tracing::warn!(
                surface,
                "janus-auth: {TOKEN_ENV} is UNSET — mutating (non-GET) requests will be DENIED \
                 (503, fail-closed). GET/HEAD/OPTIONS stay open. Set {TOKEN_ENV} to enable writes."
            ),
        }
    }

    fn matches(&self, provided_token: &[u8]) -> bool {
        match self {
            Posture::Enforcing { token_digest } => {
                // Hash the provided token to a fixed 32 bytes before comparing,
                // so the constant-time compare never short-circuits on a length
                // difference (which would leak the token length).
                let provided_digest = sha256(provided_token);
                provided_digest.ct_eq(token_digest).into()
            }
            Posture::Unconfigured => false,
        }
    }
}

/// True for methods that mutate state and therefore require a bearer.
/// GET, HEAD and OPTIONS (CORS preflight, WS upgrade) are non-mutating.
fn is_mutating(method: &Method) -> bool {
    !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS)
}

/// Extract the token from an `Authorization: Bearer <token>` header value.
/// The scheme match is ASCII-case-insensitive; the token is returned verbatim.
fn bearer_token(auth_header: Option<&str>) -> Option<&str> {
    let raw = auth_header?;
    let (scheme, rest) = raw.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        let tok = rest.trim();
        if tok.is_empty() { None } else { Some(tok) }
    } else {
        None
    }
}

/// Pure authorization decision. `auth_header` is the raw `Authorization` value.
pub fn decide(posture: &Posture, method: &Method, auth_header: Option<&str>) -> Decision {
    if !is_mutating(method) {
        return Decision::Allow;
    }
    match posture {
        Posture::Unconfigured => Decision::Misconfigured,
        Posture::Enforcing { .. } => match bearer_token(auth_header) {
            Some(tok) if posture.matches(tok.as_bytes()) => Decision::Allow,
            _ => Decision::Unauthorized,
        },
    }
}

/// axum middleware: enforce [`decide`] on every request. Attach with
/// `axum::middleware::from_fn_with_state(posture, janus_auth::enforce)`.
pub async fn enforce(State(posture): State<Posture>, req: Request, next: Next) -> Response {
    let auth = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match decide(&posture, req.method(), auth) {
        Decision::Allow => next.run(req).await,
        Decision::Unauthorized => (
            StatusCode::UNAUTHORIZED,
            "unauthorized: valid 'Authorization: Bearer <JANUS_API_TOKEN>' required for mutating requests\n",
        )
            .into_response(),
        Decision::Misconfigured => (
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable: JANUS_API_TOKEN is not configured; mutating requests are denied (fail-closed)\n",
        )
            .into_response(),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request as HttpRequest, StatusCode},
        routing::{get, post},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt; // oneshot

    const TOKEN: &str = "s3cr3t-token-value";

    fn enforcing() -> Posture {
        Posture::from_token(TOKEN.as_bytes())
    }

    // ---- pure decision matrix -------------------------------------------

    #[test]
    fn get_is_always_allowed_even_without_token() {
        for p in [enforcing(), Posture::Unconfigured] {
            assert_eq!(decide(&p, &Method::GET, None), Decision::Allow);
            assert_eq!(decide(&p, &Method::HEAD, None), Decision::Allow);
            assert_eq!(decide(&p, &Method::OPTIONS, None), Decision::Allow);
        }
    }

    #[test]
    fn mutating_without_header_is_denied_when_enforcing() {
        let p = enforcing();
        for m in [Method::POST, Method::PUT, Method::DELETE, Method::PATCH] {
            assert_eq!(decide(&p, &m, None), Decision::Unauthorized);
        }
    }

    #[test]
    fn mutating_with_wrong_token_is_denied() {
        let p = enforcing();
        assert_eq!(
            decide(&p, &Method::PUT, Some("Bearer not-the-token")),
            Decision::Unauthorized
        );
        // right length shape, wrong value
        assert_eq!(
            decide(&p, &Method::PUT, Some("Bearer s3cr3t-token-valuX")),
            Decision::Unauthorized
        );
    }

    #[test]
    fn mutating_with_correct_token_is_allowed() {
        let p = enforcing();
        for m in [Method::POST, Method::PUT, Method::DELETE, Method::PATCH] {
            assert_eq!(
                decide(&p, &m, Some(&format!("Bearer {TOKEN}"))),
                Decision::Allow
            );
        }
    }

    #[test]
    fn bearer_scheme_is_case_insensitive_token_is_not() {
        let p = enforcing();
        assert_eq!(
            decide(&p, &Method::POST, Some(&format!("bearer {TOKEN}"))),
            Decision::Allow
        );
        assert_eq!(
            decide(&p, &Method::POST, Some(&format!("BEARER {TOKEN}"))),
            Decision::Allow
        );
        // token itself must match exactly (case-sensitive)
        assert_eq!(
            decide(&p, &Method::POST, Some("Bearer S3CR3T-TOKEN-VALUE")),
            Decision::Unauthorized
        );
    }

    #[test]
    fn malformed_authorization_headers_are_denied() {
        let p = enforcing();
        for h in [
            "",
            "Bearer",
            "Bearer ",
            TOKEN,                // no scheme
            "Basic dXNlcjpwdw==", // wrong scheme
            "Token abc",
        ] {
            assert_eq!(
                decide(&p, &Method::POST, Some(h)),
                Decision::Unauthorized,
                "header {h:?} should be unauthorized"
            );
        }
    }

    #[test]
    fn unconfigured_denies_mutations_but_allows_reads() {
        let p = Posture::Unconfigured;
        // even a bearer header cannot help — there is nothing to match against
        assert_eq!(
            decide(&p, &Method::POST, Some(&format!("Bearer {TOKEN}"))),
            Decision::Misconfigured
        );
        assert_eq!(decide(&p, &Method::GET, None), Decision::Allow);
    }

    #[test]
    fn from_env_treats_empty_and_whitespace_as_unset() {
        // exercised without touching the process env: whitespace-only token
        // via from_token would be Enforcing, but from_env's guard rejects it.
        // Verify the guard logic directly.
        assert!(matches!(posture_for("   "), Posture::Unconfigured));
        assert!(matches!(posture_for(""), Posture::Unconfigured));
        assert!(matches!(posture_for("abc"), Posture::Enforcing { .. }));
    }

    // mirror of from_env's guard, kept env-free for deterministic tests
    fn posture_for(tok: &str) -> Posture {
        if tok.trim().is_empty() {
            Posture::Unconfigured
        } else {
            Posture::from_token(tok.as_bytes())
        }
    }

    #[test]
    fn constant_time_compare_is_length_independent() {
        // matches() must return a bool for tokens of any length without panic
        // (the SHA-256 pre-hash guarantees equal-length constant-time compare).
        let p = enforcing();
        assert!(p.matches(TOKEN.as_bytes()));
        assert!(!p.matches(b"short"));
        assert!(!p.matches(b"a-much-longer-token-than-the-real-one-aaaaaaaaaaaa"));
        assert!(!p.matches(b""));
    }

    // ---- axum layer end-to-end matrix -----------------------------------

    fn app(posture: Posture) -> Router {
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/read", get(|| async { "ok" }))
            .route("/mutate", post(|| async { "done" }))
            .layer(axum::middleware::from_fn_with_state(posture, enforce))
    }

    async fn status(app: Router, method: Method, path: &str, auth: Option<&str>) -> StatusCode {
        let mut req = HttpRequest::builder().method(method).uri(path);
        if let Some(a) = auth {
            req = req.header(header::AUTHORIZATION, a);
        }
        let resp = app.oneshot(req.body(Body::empty()).unwrap()).await.unwrap();
        resp.status()
    }

    #[tokio::test]
    async fn layer_allows_get_and_gates_post() {
        // enforcing
        assert_eq!(
            status(app(enforcing()), Method::GET, "/health", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(app(enforcing()), Method::GET, "/read", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(app(enforcing()), Method::POST, "/mutate", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(
                app(enforcing()),
                Method::POST,
                "/mutate",
                Some("Bearer wrong")
            )
            .await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(
                app(enforcing()),
                Method::POST,
                "/mutate",
                Some(&format!("Bearer {TOKEN}"))
            )
            .await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn layer_unconfigured_fails_closed_on_post_open_on_get() {
        assert_eq!(
            status(app(Posture::Unconfigured), Method::GET, "/read", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(
                app(Posture::Unconfigured),
                Method::POST,
                "/mutate",
                Some(&format!("Bearer {TOKEN}"))
            )
            .await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn denied_response_never_echoes_the_token() {
        let resp = app(enforcing())
            .oneshot(
                HttpRequest::builder()
                    .method(Method::POST)
                    .uri("/mutate")
                    .header(header::AUTHORIZATION, format!("Bearer {TOKEN}-wrong"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body);
        assert!(
            !text.contains(TOKEN),
            "response body must not leak the token"
        );
    }
}
