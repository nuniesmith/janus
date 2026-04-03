//! Setup API routes for JANUS Rust Gateway.
//!
//! Provides endpoints for initial system setup, including Authelia admin user creation
//! and other first-time configuration tasks.

use argon2::{
    Argon2, Params, Version,
    password_hash::{PasswordHasher, SaltString, rand_core::OsRng},
};
use axum::{
    Json, Router,
    extract::{FromRequest, Request, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::AsyncWriteExt;
#[allow(unused_imports)]
use tracing::{debug, error, info, warn};

use crate::state::AppState;

/// Custom JSON extractor that returns proper error responses.
///
/// This wraps axum's Json extractor to provide user-friendly error messages
/// when JSON parsing fails, instead of axum's default empty 400 response.
pub struct JsonBody<T>(pub T);

impl<S, T> FromRequest<S> for JsonBody<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    // Note: Can't use `async fn` here due to axum 0.8's FromRequest trait requirements
    #[allow(clippy::manual_async_fn)]
    fn from_request(
        req: Request,
        state: &S,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        async move {
            match Json::<T>::from_request(req, state).await {
                Ok(Json(value)) => Ok(JsonBody(value)),
                Err(rejection) => {
                    let error_message = match &rejection {
                        JsonRejection::JsonDataError(e) => {
                            format!("Invalid JSON data: {}", e)
                        }
                        JsonRejection::JsonSyntaxError(e) => {
                            format!("JSON syntax error: {}", e)
                        }
                        JsonRejection::MissingJsonContentType(e) => {
                            format!("Missing Content-Type header: {}", e)
                        }
                        JsonRejection::BytesRejection(e) => {
                            format!("Failed to read request body: {}", e)
                        }
                        _ => format!("JSON parsing error: {}", rejection),
                    };

                    warn!("JSON extraction failed: {}", error_message);

                    let error_response = SetupError {
                        success: false,
                        message: error_message,
                        details: None,
                    };

                    Err((StatusCode::BAD_REQUEST, Json(error_response)).into_response())
                }
            }
        }
    }
}

/// Path to Authelia users database file (can be overridden via AUTHELIA_USERS_DB_PATH env var)
const AUTHELIA_USERS_DB_PATH_DEFAULT: &str = "/config/authelia/users_database.yml";
/// Alternative path for local development
const AUTHELIA_USERS_DB_PATH_DEV: &str = "./infrastructure/config/authelia/users_database.yml";

/// Setup completion request.
#[derive(Debug, Deserialize)]
pub struct SetupRequest {
    username: String,
    #[serde(default)]
    email: Option<String>,
    password: String,
}

/// Setup completion response.
#[derive(Debug, Serialize)]
pub struct SetupResponse {
    success: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    username: Option<String>,
}

/// Setup status response.
#[derive(Debug, Serialize)]
pub struct SetupStatus {
    setup_complete: bool,
    message: String,
    required_steps: Vec<String>,
}

/// Error response for setup failures.
#[derive(Debug, Serialize)]
pub struct SetupError {
    success: bool,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<String>,
}

/// Check if setup is complete.
///
/// Returns the current setup status and any required steps.
async fn get_setup_status(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<SetupStatus>, StatusCode> {
    info!("Setup status check requested");

    // Check if Authelia users database has non-default users
    let setup_complete = check_setup_complete().await;

    let mut required_steps = Vec::new();

    if !setup_complete {
        required_steps.push("Create admin user".to_string());
    }

    let response = SetupStatus {
        setup_complete,
        message: if setup_complete {
            "Setup complete".to_string()
        } else {
            "Setup wizard available".to_string()
        },
        required_steps,
    };

    Ok(Json(response))
}

/// Check if setup has been completed by looking for non-default admin users
async fn check_setup_complete() -> bool {
    let users_db_path = get_users_db_path();

    match fs::read_to_string(&users_db_path).await {
        Ok(content) => {
            // Check if there's a custom admin user (not the default placeholder)
            // The default has password hash starting with the known default
            let has_custom_admin = content.contains("groups:")
                && content.contains("- admins")
                && !content.contains("# Default password: ChangeMe123!");

            // Also check if setup_complete marker exists
            let has_marker = content.contains("# SETUP_COMPLETE: true");

            has_custom_admin || has_marker
        }
        Err(_) => false,
    }
}

/// Get the path to the Authelia users database file
fn get_users_db_path() -> PathBuf {
    // Check for environment variable override first
    if let Ok(env_path) = std::env::var("AUTHELIA_USERS_DB_PATH") {
        let path = PathBuf::from(&env_path);
        info!("Using AUTHELIA_USERS_DB_PATH from environment: {:?}", path);
        return path;
    }

    // Try production path
    let prod_path = PathBuf::from(AUTHELIA_USERS_DB_PATH_DEFAULT);
    if prod_path.exists() {
        return prod_path;
    }

    // Fall back to development path
    let dev_path = PathBuf::from(AUTHELIA_USERS_DB_PATH_DEV);
    if dev_path.exists() {
        return dev_path;
    }

    // Default to production path (will be created if needed)
    prod_path
}

/// Hash password using Argon2id (Authelia compatible)
fn hash_password_argon2(password: &str) -> Result<String, String> {
    // Authelia uses Argon2id with specific parameters
    // Default: m=65536 (64MB), t=3, p=4
    let params =
        Params::new(65536, 3, 4, Some(32)).map_err(|e| format!("Argon2 params error: {}", e))?;

    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);

    let salt = SaltString::generate(&mut OsRng);

    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| format!("Password hashing error: {}", e))?;

    Ok(password_hash.to_string())
}

/// Complete the initial setup.
///
/// This endpoint:
/// 1. Validates the provided credentials
/// 2. Hashes the password with Argon2id
/// 3. Creates the Authelia admin user in users_database.yml
/// 4. Marks setup as complete
async fn complete_setup(
    State(state): State<Arc<AppState>>,
    JsonBody(request): JsonBody<SetupRequest>,
) -> Result<Json<SetupResponse>, (StatusCode, Json<SetupError>)> {
    info!("Setup completion requested for user: {}", request.username);

    // Check if setup is already complete
    if check_setup_complete().await {
        warn!("Setup already completed, rejecting new setup request");
        return Err((
            StatusCode::CONFLICT,
            Json(SetupError {
                success: false,
                message: "Setup has already been completed".to_string(),
                details: Some(
                    "An admin user already exists. Use the password reset function if needed."
                        .to_string(),
                ),
            }),
        ));
    }

    // Validate username
    if request.username.len() < 3 {
        warn!("Setup failed: username too short");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(SetupError {
                success: false,
                message: "Username must be at least 3 characters".to_string(),
                details: None,
            }),
        ));
    }

    if request.username.contains(char::is_whitespace) {
        warn!("Setup failed: username contains whitespace");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(SetupError {
                success: false,
                message: "Username cannot contain spaces".to_string(),
                details: None,
            }),
        ));
    }

    // Validate username characters (alphanumeric, underscore, hyphen only)
    if !request
        .username
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        warn!("Setup failed: username contains invalid characters");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(SetupError {
                success: false,
                message: "Username can only contain letters, numbers, underscores, and hyphens"
                    .to_string(),
                details: None,
            }),
        ));
    }

    // Validate password - Authelia requires strong passwords
    if request.password.len() < 12 {
        warn!("Setup failed: password too short");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(SetupError {
                success: false,
                message: "Password must be at least 12 characters".to_string(),
                details: None,
            }),
        ));
    }

    // Check password complexity
    let has_upper = request.password.chars().any(|c| c.is_uppercase());
    let has_lower = request.password.chars().any(|c| c.is_lowercase());
    let has_digit = request.password.chars().any(|c| c.is_ascii_digit());
    let has_special = request.password.chars().any(|c| !c.is_alphanumeric());

    if !has_upper || !has_lower || !has_digit || !has_special {
        warn!("Setup failed: password doesn't meet complexity requirements");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(SetupError {
                success: false,
                message:
                    "Password must contain uppercase, lowercase, number, and special character"
                        .to_string(),
                details: None,
            }),
        ));
    }

    // Hash the password with Argon2id
    let password_hash = match hash_password_argon2(&request.password) {
        Ok(hash) => hash,
        Err(e) => {
            error!("Failed to hash password: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SetupError {
                    success: false,
                    message: "Failed to process password".to_string(),
                    details: Some(e),
                }),
            ));
        }
    };

    // Auto-generate a placeholder email if none was provided.
    // Authelia's file provider expects an email field in users_database.yml.
    let email = request
        .email
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| format!("{}@fkstrading.local", request.username));

    info!(
        "Creating Authelia admin user: {} ({})",
        request.username, email
    );

    // Create the Authelia user entry
    match create_authelia_user(&request.username, &email, &password_hash).await {
        Ok(()) => {
            info!(
                "Successfully created Authelia admin user: {}",
                request.username
            );

            // Record metrics
            state.metrics.setup_completions.inc();

            Ok(Json(SetupResponse {
                success: true,
                message: "Admin user created successfully. You can now log in at the authentication portal.".to_string(),
                username: Some(request.username),
            }))
        }
        Err(e) => {
            error!("Failed to create Authelia user: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SetupError {
                    success: false,
                    message: "Failed to create admin user".to_string(),
                    details: Some(e),
                }),
            ))
        }
    }
}

/// Create or update the Authelia users database with the new admin user
async fn create_authelia_user(
    username: &str,
    email: &str,
    password_hash: &str,
) -> Result<(), String> {
    let users_db_path = get_users_db_path();

    // Create the new users database content
    let users_yaml = format!(
        r#"##
## Authelia Users Database
## FKS Trading Platform
## https://www.authelia.com/configuration/authentication/file/
##
## Generated by FKS Setup Wizard
## SETUP_COMPLETE: true
##

users:
    # Admin user created during initial setup
    {username}:
        disabled: false
        displayname: "Administrator"
        password: "{password_hash}"
        email: {email}
        groups:
            - admins
            - dev
            - traders
            - users

##
## Groups Reference:
## -----------------
## admins   - Full administrative access, all services, two-factor required for sensitive ops
## dev      - Developer access to dev tools, debugging endpoints, code deployment
## traders  - Access to trading execution service, order management, two-factor required
## users    - Basic access to web application, dashboards, read-only metrics
##
"#,
        username = username,
        password_hash = password_hash,
        email = email,
    );

    // Ensure parent directory exists
    if let Some(parent) = users_db_path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    // Write the users database file
    let mut file = fs::File::create(&users_db_path)
        .await
        .map_err(|e| format!("Failed to create users database file: {}", e))?;

    file.write_all(users_yaml.as_bytes())
        .await
        .map_err(|e| format!("Failed to write users database: {}", e))?;

    file.sync_all()
        .await
        .map_err(|e| format!("Failed to sync users database: {}", e))?;

    info!("Authelia users database written to: {:?}", users_db_path);

    // Set file permissions (readable by authelia user)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&users_db_path, perms)
            .map_err(|e| format!("Failed to set file permissions: {}", e))?;
    }

    Ok(())
}

/// Reset setup status (admin only - for testing).
///
/// This endpoint allows administrators to reset the setup state for testing.
/// In production, this should be protected by admin authentication.
async fn reset_setup(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SetupResponse>, (StatusCode, Json<SetupError>)> {
    // `state` is used in debug builds (reset path) but not in release builds (early return).
    #[cfg(not(debug_assertions))]
    let _ = &state;
    warn!("Setup reset requested");

    // Only allow in debug/development mode
    #[cfg(not(debug_assertions))]
    {
        return Err((
            StatusCode::FORBIDDEN,
            Json(SetupError {
                success: false,
                message: "Setup reset is not allowed in production".to_string(),
                details: None,
            }),
        ));
    }

    #[cfg(debug_assertions)]
    {
        let users_db_path = get_users_db_path();

        // Create a default/empty users database
        let default_yaml = r#"##
## Authelia Users Database
## FKS Trading Platform
## https://www.authelia.com/configuration/authentication/file/
##
## Password Generation:
##   docker run --rm authelia/authelia:latest authelia crypto hash generate argon2 --password 'yourpassword'
##
## IMPORTANT: Run the setup wizard to create your admin account!
##

users:
    # Default placeholder - will be replaced by setup wizard
    # Default password: ChangeMe123! (DO NOT USE IN PRODUCTION)
    admin:
        disabled: true
        displayname: "Placeholder Admin"
        password: "$argon2id$v=19$m=65536,t=3,p=4$a/m4b4gP2WqgrDutHRSCBA$q5KpsHI7AfrDNcykEhEFpKCDVMjX8/PuPVmVRi4xBKY"
        email: admin@example.com
        groups:
            - admins
"#;

        if let Err(e) = fs::write(&users_db_path, default_yaml).await {
            error!("Failed to reset users database: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SetupError {
                    success: false,
                    message: "Failed to reset setup".to_string(),
                    details: Some(e.to_string()),
                }),
            ));
        }

        state.metrics.setup_resets.inc();

        Ok(Json(SetupResponse {
            success: true,
            message: "Setup has been reset. You can run the setup wizard again.".to_string(),
            username: None,
        }))
    }
}

/// Build and return the setup routes.
pub fn setup_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/setup/status", get(get_setup_status))
        .route("/api/setup/complete", post(complete_setup))
        .route("/api/setup/reset", post(reset_setup))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;
    use crate::redis_dispatcher::SignalDispatcher;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn test_password_hashing() {
        let password = "SecurePassword123!";
        let hash = hash_password_argon2(password).unwrap();

        // Verify hash format (Argon2id PHC string format)
        assert!(hash.starts_with("$argon2id$"));
        assert!(hash.contains("$m=65536"));
        assert!(hash.contains(",t=3"));
        assert!(hash.contains(",p=4"));
    }

    #[tokio::test]
    async fn test_setup_status() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = setup_routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/setup/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_setup_validation_username_too_short() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = setup_routes().with_state(state);

        let payload = r#"{"username":"ab","password":"SecurePassword123!"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/setup/complete")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_setup_validation_username_single_char() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = setup_routes().with_state(state);

        // With email removed as a required field, test that missing username
        // still fails validation even when email is omitted.
        let payload = r#"{"username":"a","password":"SecurePassword123!"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/setup/complete")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_setup_validation_password_too_short() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = setup_routes().with_state(state);

        let payload = r#"{"username":"admin","password":"Short1!"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/setup/complete")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_setup_validation_password_no_special() {
        let settings = Settings::from_env();
        let dispatcher = Arc::new(SignalDispatcher::new("redis://localhost:6379/0"));
        let state = Arc::new(AppState::new(settings, dispatcher));
        let app = setup_routes().with_state(state);

        let payload = r#"{"username":"admin","password":"SecurePassword123"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/setup/complete")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
