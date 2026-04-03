//! JWT Authentication Middleware
//!
//! Provides JWT-based authentication for API endpoints.
//!
//! ## Usage
//!
//! ### Protecting routes:
//! ```rust
//! use crate::api::auth::{AuthConfig, require_auth};
//!
//! let protected_router = Router::new()
//!     .route("/protected", get(handler))
//!     .layer(middleware::from_fn_with_state(auth_config.clone(), require_auth));
//! ```
//!
//! ### Generating tokens:
//! ```rust
//! let token = generate_jwt(&auth_config, "user_id", 86400)?;
//! ```

use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    /// Subject (user ID or service name)
    pub sub: String,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Optional: Issuer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,

    /// Optional: Audience
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
}

/// Authentication configuration
#[derive(Clone)]
pub struct AuthConfig {
    /// JWT secret for encoding/decoding
    pub secret: String,

    /// Token expiry duration in seconds
    #[allow(dead_code)]
    pub expiry_seconds: i64,

    /// Optional issuer name
    pub issuer: Option<String>,

    /// Optional audience
    pub audience: Option<String>,

    /// Whether authentication is enabled
    pub enabled: bool,
}

impl AuthConfig {
    /// Create a new auth configuration
    #[allow(dead_code)]
    pub fn new(secret: String, expiry_seconds: i64) -> Self {
        Self {
            secret,
            expiry_seconds,
            issuer: Some("fks-data-service".to_string()),
            audience: None,
            enabled: true,
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self, String> {
        let secret = std::env::var("DATA_SERVICE_JWT_SECRET")
            .map_err(|_| "DATA_SERVICE_JWT_SECRET not set")?;

        let expiry_seconds = std::env::var("DATA_SERVICE_JWT_EXPIRY")
            .unwrap_or_else(|_| "86400".to_string())
            .parse::<i64>()
            .map_err(|_| "Invalid DATA_SERVICE_JWT_EXPIRY")?;

        let enabled = std::env::var("DATA_SERVICE_JWT_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);

        Ok(Self {
            secret,
            expiry_seconds,
            issuer: Some("fks-data-service".to_string()),
            audience: None,
            enabled,
        })
    }

    /// Create a disabled auth config (for development)
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        Self {
            secret: "disabled".to_string(),
            expiry_seconds: 86400,
            issuer: None,
            audience: None,
            enabled: false,
        }
    }
}

/// Generate a JWT token
#[allow(dead_code)]
pub fn generate_jwt(
    config: &AuthConfig,
    subject: &str,
    custom_expiry: Option<i64>,
) -> Result<String, jsonwebtoken::errors::Error> {
    let now = chrono::Utc::now().timestamp();
    let exp = now + custom_expiry.unwrap_or(config.expiry_seconds);

    let claims = Claims {
        sub: subject.to_string(),
        iat: now,
        exp,
        iss: config.issuer.clone(),
        aud: config.audience.clone(),
    };

    let encoding_key = EncodingKey::from_secret(config.secret.as_bytes());
    encode(&Header::default(), &claims, &encoding_key)
}

/// Validate a JWT token
pub fn validate_jwt(
    config: &AuthConfig,
    token: &str,
) -> Result<Claims, jsonwebtoken::errors::Error> {
    let decoding_key = DecodingKey::from_secret(config.secret.as_bytes());
    let mut validation = Validation::default();

    if let Some(ref issuer) = config.issuer {
        validation.set_issuer(&[issuer]);
    }

    if let Some(ref audience) = config.audience {
        validation.set_audience(&[audience]);
    }

    let token_data = decode::<Claims>(token, &decoding_key, &validation)?;
    Ok(token_data.claims)
}

/// Extract JWT token from request
fn extract_token(req: &Request) -> Option<String> {
    // Try Authorization header first (Bearer token)
    if let Some(auth_header) = req.headers().get(header::AUTHORIZATION)
        && let Ok(auth_str) = auth_header.to_str()
        && let Some(token) = auth_str.strip_prefix("Bearer ")
    {
        return Some(token.to_string());
    }

    // Try X-API-Key header as fallback
    if let Some(api_key) = req.headers().get("X-API-Key")
        && let Ok(key_str) = api_key.to_str()
    {
        return Some(key_str.to_string());
    }

    None
}

/// Authentication middleware
pub async fn require_auth(
    State(config): State<Arc<AuthConfig>>,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    // If auth is disabled, pass through
    if !config.enabled {
        debug!("JWT authentication is disabled, allowing request");
        return Ok(next.run(req).await);
    }

    // Extract token
    let token = match extract_token(&req) {
        Some(t) => t,
        None => {
            warn!("No authentication token provided");
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "Missing authentication token",
                    "message": "Provide a valid JWT token in the Authorization header (Bearer <token>) or X-API-Key header"
                })),
            ).into_response());
        }
    };

    // Validate token
    match validate_jwt(&config, &token) {
        Ok(claims) => {
            debug!("JWT validated for subject: {}", claims.sub);
            // Token is valid, continue with request
            Ok(next.run(req).await)
        }
        Err(err) => {
            warn!("JWT validation failed: {:?}", err);
            Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({
                    "error": "Invalid authentication token",
                    "message": format!("Token validation failed: {}", err)
                })),
            )
                .into_response())
        }
    }
}

/// Generate token endpoint handler
#[allow(dead_code)]
#[derive(Deserialize)]
pub struct GenerateTokenRequest {
    /// Subject (user/service identifier)
    pub subject: String,

    /// Optional custom expiry in seconds
    pub expiry_seconds: Option<i64>,
}

#[allow(dead_code)]
#[derive(Serialize)]
pub struct GenerateTokenResponse {
    /// Generated JWT token
    pub token: String,

    /// Token expiry timestamp
    pub expires_at: i64,
}

/// Handler to generate a new JWT token
///
/// This endpoint itself should be protected by a master key or other authentication
#[allow(dead_code)]
pub async fn generate_token_handler(
    State(config): State<Arc<AuthConfig>>,
    Json(payload): Json<GenerateTokenRequest>,
) -> Result<Json<GenerateTokenResponse>, (StatusCode, Json<serde_json::Value>)> {
    if !config.enabled {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "JWT authentication is disabled"
            })),
        ));
    }

    let token = generate_jwt(&config, &payload.subject, payload.expiry_seconds).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "Failed to generate token",
                "message": e.to_string()
            })),
        )
    })?;

    let exp =
        chrono::Utc::now().timestamp() + payload.expiry_seconds.unwrap_or(config.expiry_seconds);

    Ok(Json(GenerateTokenResponse {
        token,
        expires_at: exp,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_validate_jwt() {
        let config = AuthConfig::new("test_secret_key_12345".to_string(), 3600);

        // Generate token
        let token = generate_jwt(&config, "test_user", None).unwrap();
        assert!(!token.is_empty());

        // Validate token
        let claims = validate_jwt(&config, &token).unwrap();
        assert_eq!(claims.sub, "test_user");
        assert!(claims.exp > claims.iat);
    }

    #[test]
    fn test_invalid_token() {
        let config = AuthConfig::new("test_secret_key_12345".to_string(), 3600);

        // Try to validate an invalid token
        let result = validate_jwt(&config, "invalid.token.here");
        assert!(result.is_err());
    }

    #[test]
    fn test_expired_token() {
        // Use -120 seconds to ensure token is well past the default JWT leeway (60s)
        let config = AuthConfig::new("test_secret_key_12345".to_string(), -120);

        let token = generate_jwt(&config, "test_user", Some(-120)).unwrap();

        // Validation should fail due to expiration
        let result = validate_jwt(&config, &token);
        assert!(result.is_err());
    }
}
