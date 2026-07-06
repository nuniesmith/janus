//! Janus runtime-config endpoints for the FKS WebUI settings panel.
//!
//! `GET /api/config` returns the *effective* Janus/optimizer configuration:
//! the env-var defaults the modules actually read at boot, overlaid with any
//! operator overrides stored in the Redis hash [`OVERRIDES_KEY`]. `PUT
//! /api/config` validates and stores those overrides, then publishes a
//! [`ParamNotification::ConfigUpdate`] on the optimizer's
//! `fks:{instance}:param_updates` channel so running subscribers (janus-api
//! and the forward service) log the change.
//!
//! # Honesty about effect-at-runtime
//!
//! Every field served here is **boot-time configuration** — the consuming
//! code reads the env var once and never re-reads it:
//!
//! | field                      | env var                  | read by / when |
//! |----------------------------|--------------------------|----------------|
//! | `optimize_assets`          | `OPTIMIZE_ASSETS`        | `Config::load` at process boot; standalone optimizer CLI at its boot |
//! | `optimize_interval`        | `OPTIMIZE_INTERVAL`      | standalone optimizer runner at its boot (not read by the unified janus binary) |
//! | `optimize_trials`          | `OPTIMIZE_TRIALS`        | standalone optimizer runner at its boot |
//! | `optimize_historical_days` | `OPTIMIZE_HISTORICAL_DAYS` | standalone optimizer runner at its boot |
//! | `data_kline_intervals`     | `DATA_KLINE_INTERVALS`   | data module when live ingestion starts |
//! | `janus_auto_start`         | `JANUS_AUTO_START`       | `bin/janus` `main()` at process boot |
//! | `janus_bootstrap_days`     | `JANUS_BOOTSTRAP_DAYS`   | forward module brain boot (affinity bootstrap) |
//!
//! Accordingly `requires_restart` lists *all* fields: a stored override never
//! reconfigures a running module. The overrides hash is the WebUI's source of
//! truth for the *desired* config; wiring boot-time consumption of the hash
//! (so a restart picks overrides up without editing `.env`) is tracked as
//! follow-up work in the PR that introduced this module.
//!
//! Redis resilience: `GET` degrades to env defaults with
//! `redis_available: false` when Redis is unreachable (never an error);
//! `PUT` returns an honest `503` because the overrides cannot be persisted.

use axum::{
    Extension, Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use janus_core::{JanusState, ParamManager, optimized_params::ParamNotification};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};

/// Redis hash holding operator overrides (field name → string value).
pub const OVERRIDES_KEY: &str = "janus:config:overrides";

/// Upper bound for any single Redis roundtrip made by these handlers. The
/// WebUI settings page loads this once per visit; a slow/unreachable Redis
/// must degrade fast rather than hang the panel.
const REDIS_TIMEOUT: Duration = Duration::from_millis(2_000);

/// Intervals the optimizer accepts for `OPTIMIZE_INTERVAL`. The optimizer has
/// no fixed whitelist — it parses any `<n><s|m|h|d>` token
/// (`services/optimizer::config::parse_interval`) and uses the string as a
/// data-file suffix — so this is the documented, UI-facing set: the Kraken
/// OHLC timeframes the collector stores (1m/5m/15m/1h/4h/1d) plus the
/// deployed default `6h` and its neighbours. All parse with the optimizer's
/// rule.
pub const VALID_INTERVALS: &[&str] = &["1m", "5m", "15m", "30m", "1h", "4h", "6h", "12h", "1d"];

/// Kline stream intervals accepted for `DATA_KLINE_INTERVALS`. The data
/// module forwards these verbatim to the exchange WebSocket subscription
/// (Binance by default), so validate against Binance's kline interval set.
const VALID_KLINE_INTERVALS: &[&str] = &[
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w",
    "1M",
];

/// Every config field is read once at boot (see module docs), so an override
/// only takes effect after the owning process/module restarts.
pub const REQUIRES_RESTART: &[&str] = &[
    "optimize_assets",
    "optimize_interval",
    "optimize_trials",
    "optimize_historical_days",
    "data_kline_intervals",
    "janus_auto_start",
    "janus_bootstrap_days",
];

/// The seven settings-panel fields, in the WebUI's `JanusConfig` shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JanusConfigValues {
    pub optimize_assets: String,
    pub optimize_interval: String,
    pub optimize_trials: u32,
    pub optimize_historical_days: u32,
    pub data_kline_intervals: String,
    pub janus_auto_start: bool,
    pub janus_bootstrap_days: u32,
}

/// `GET /api/config` response — the WebUI's `JanusConfigResponse` plus the
/// `requires_restart` honesty marker.
#[derive(Debug, Serialize)]
pub struct JanusConfigResponse {
    pub config: JanusConfigValues,
    /// `optimize_assets` split into individual symbols.
    pub assets_list: Vec<String>,
    /// `"redis"` when at least one stored override is applied, else
    /// `"env_defaults"`.
    pub source: &'static str,
    pub valid_intervals: Vec<&'static str>,
    /// Result of the same live Redis roundtrip `/health` reports under
    /// `components.redis` (here it doubles as the overrides read).
    pub redis_available: bool,
    /// Fields that only take effect after the owning process restarts —
    /// currently all of them (boot-time env config; see module docs).
    pub requires_restart: Vec<&'static str>,
}

/// `PUT /api/config` body. All fields optional so partial updates validate
/// only what they touch; the WebUI always sends all seven.
#[derive(Debug, Default, Deserialize)]
pub struct JanusConfigUpdate {
    pub optimize_assets: Option<String>,
    pub optimize_interval: Option<String>,
    pub optimize_trials: Option<u32>,
    pub optimize_historical_days: Option<u32>,
    pub data_kline_intervals: Option<String>,
    pub janus_auto_start: Option<bool>,
    pub janus_bootstrap_days: Option<u32>,
}

// ─── Env defaults ────────────────────────────────────────────────────────────

/// Effective defaults from the same env vars the consuming code reads
/// (fallbacks mirror each consumer's own hardcoded default).
pub(crate) fn env_defaults() -> JanusConfigValues {
    defaults_from(|k| std::env::var(k).ok())
}

/// Testable core of [`env_defaults`]: build the defaults from any lookup.
fn defaults_from(get: impl Fn(&str) -> Option<String>) -> JanusConfigValues {
    let get_str = |k: &str, d: &str| get(k).filter(|v| !v.is_empty()).unwrap_or_else(|| d.into());
    let get_u32 = |k: &str, d: u32| get(k).and_then(|v| v.trim().parse().ok()).unwrap_or(d);
    JanusConfigValues {
        optimize_assets: get_str("OPTIMIZE_ASSETS", "BTC,ETH,SOL"),
        optimize_interval: get_str("OPTIMIZE_INTERVAL", "15m"),
        optimize_trials: get_u32("OPTIMIZE_TRIALS", 100),
        optimize_historical_days: get_u32("OPTIMIZE_HISTORICAL_DAYS", 30),
        data_kline_intervals: get_str("DATA_KLINE_INTERVALS", "1m,5m"),
        janus_auto_start: get("JANUS_AUTO_START")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(false),
        janus_bootstrap_days: get_u32("JANUS_BOOTSTRAP_DAYS", 30),
    }
}

// ─── Pure helpers (unit-tested without Redis) ────────────────────────────────

/// Overlay stored overrides onto the env defaults. Returns the names of the
/// fields an override was applied to (unknown hash fields and values that
/// fail to parse are ignored — the env default stands).
fn apply_overrides(
    config: &mut JanusConfigValues,
    overrides: &HashMap<String, String>,
) -> Vec<String> {
    let mut applied = Vec::new();
    for (field, value) in overrides {
        let ok = match field.as_str() {
            "optimize_assets" => {
                config.optimize_assets = value.clone();
                true
            }
            "optimize_interval" => {
                config.optimize_interval = value.clone();
                true
            }
            "optimize_trials" => match value.parse() {
                Ok(v) => {
                    config.optimize_trials = v;
                    true
                }
                Err(_) => false,
            },
            "optimize_historical_days" => match value.parse() {
                Ok(v) => {
                    config.optimize_historical_days = v;
                    true
                }
                Err(_) => false,
            },
            "data_kline_intervals" => {
                config.data_kline_intervals = value.clone();
                true
            }
            "janus_auto_start" => match value.parse() {
                Ok(v) => {
                    config.janus_auto_start = v;
                    true
                }
                Err(_) => false,
            },
            "janus_bootstrap_days" => match value.parse() {
                Ok(v) => {
                    config.janus_bootstrap_days = v;
                    true
                }
                Err(_) => false,
            },
            _ => false,
        };
        if ok {
            applied.push(field.clone());
        }
    }
    applied.sort();
    applied
}

/// Split a comma-separated asset string into its symbols.
fn split_assets(assets: &str) -> Vec<String> {
    assets
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Validate a `PUT /api/config` payload and normalize it into
/// `(field, stored-string-value)` pairs for the overrides hash.
///
/// Bounds mirror the settings form (trials 1–1000, historical days 1–365)
/// plus `janus_bootstrap_days` 0–365 (`0` = skip bootstrap, per the forward
/// module's contract).
fn validate_update(update: &JanusConfigUpdate) -> Result<Vec<(&'static str, String)>, String> {
    let mut fields = Vec::new();

    if let Some(assets) = &update.optimize_assets {
        let list = split_assets(assets);
        if list.is_empty() {
            return Err("optimize_assets must contain at least one symbol".into());
        }
        for sym in &list {
            if sym.len() > 12 || !sym.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(format!(
                    "optimize_assets: '{sym}' is not a valid symbol (ascii alphanumeric, max 12 chars)"
                ));
            }
        }
        fields.push(("optimize_assets", list.join(",")));
    }

    if let Some(interval) = &update.optimize_interval {
        let interval = interval.trim();
        if !VALID_INTERVALS.contains(&interval) {
            return Err(format!(
                "optimize_interval '{interval}' is not in the valid set {VALID_INTERVALS:?}"
            ));
        }
        fields.push(("optimize_interval", interval.to_string()));
    }

    if let Some(trials) = update.optimize_trials {
        if !(1..=1_000).contains(&trials) {
            return Err("optimize_trials must be between 1 and 1000".into());
        }
        fields.push(("optimize_trials", trials.to_string()));
    }

    if let Some(days) = update.optimize_historical_days {
        if !(1..=365).contains(&days) {
            return Err("optimize_historical_days must be between 1 and 365".into());
        }
        fields.push(("optimize_historical_days", days.to_string()));
    }

    if let Some(intervals) = &update.data_kline_intervals {
        let list: Vec<&str> = intervals
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if list.is_empty() {
            return Err("data_kline_intervals must contain at least one interval".into());
        }
        for iv in &list {
            if !VALID_KLINE_INTERVALS.contains(iv) {
                return Err(format!(
                    "data_kline_intervals: '{iv}' is not a Binance kline interval ({VALID_KLINE_INTERVALS:?})"
                ));
            }
        }
        fields.push(("data_kline_intervals", list.join(",")));
    }

    if let Some(auto) = update.janus_auto_start {
        fields.push(("janus_auto_start", auto.to_string()));
    }

    if let Some(days) = update.janus_bootstrap_days {
        if days > 365 {
            return Err("janus_bootstrap_days must be between 0 and 365".into());
        }
        fields.push(("janus_bootstrap_days", days.to_string()));
    }

    if fields.is_empty() {
        return Err("no recognized config fields in payload".into());
    }
    Ok(fields)
}

// ─── Redis I/O ───────────────────────────────────────────────────────────────

/// Read the overrides hash. `Err(())` means Redis is unavailable (any
/// connect/command/timeout failure); `Ok` proves a live roundtrip, so the
/// caller can report `redis_available: true` — the same liveness signal
/// `/health` derives from its PING.
async fn read_overrides(state: &Arc<JanusState>) -> Result<HashMap<String, String>, ()> {
    use redis::AsyncCommands;
    let client = state.redis_client().await.map_err(|_| ())?;
    let fut = async {
        let mut conn = client.get_multiplexed_async_connection().await.ok()?;
        conn.hgetall::<_, HashMap<String, String>>(OVERRIDES_KEY)
            .await
            .ok()
    };
    match tokio::time::timeout(REDIS_TIMEOUT, fut).await {
        Ok(Some(map)) => Ok(map),
        _ => Err(()),
    }
}

/// Write validated overrides and publish a `ConfigUpdate` notification on the
/// optimizer's param-updates channel (one pipelined roundtrip).
async fn write_overrides(
    state: &Arc<JanusState>,
    channel: &str,
    fields: &[(&'static str, String)],
) -> Result<(), String> {
    let client = state
        .redis_client()
        .await
        .map_err(|e| format!("redis client: {e}"))?;
    let notification = ParamNotification::ConfigUpdate {
        timestamp: chrono::Utc::now().to_rfc3339(),
        changed: fields.iter().map(|(f, _)| f.to_string()).collect(),
    };
    let payload = serde_json::to_string(&notification).map_err(|e| e.to_string())?;

    let mut pipe = redis::pipe();
    for (field, value) in fields {
        pipe.hset(OVERRIDES_KEY, *field, value).ignore();
    }
    pipe.publish(channel, payload).ignore();

    let fut = async {
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| e.to_string())?;
        pipe.query_async::<()>(&mut conn)
            .await
            .map_err(|e| e.to_string())
    };
    match tokio::time::timeout(REDIS_TIMEOUT, fut).await {
        Ok(res) => res,
        Err(_) => Err("redis write timed out".into()),
    }
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `GET /api/config` — effective config (env defaults ⊕ Redis overrides).
/// Redis down → env defaults with `redis_available: false`, never an error.
#[tracing::instrument(skip_all)]
pub(crate) async fn get_config_handler(State(state): State<Arc<JanusState>>) -> impl IntoResponse {
    let mut config = env_defaults();
    let (redis_available, applied) = match read_overrides(&state).await {
        Ok(overrides) => (true, apply_overrides(&mut config, &overrides)),
        Err(()) => (false, Vec::new()),
    };
    let assets_list = split_assets(&config.optimize_assets);
    Json(JanusConfigResponse {
        assets_list,
        source: if applied.is_empty() {
            "env_defaults"
        } else {
            "redis"
        },
        valid_intervals: VALID_INTERVALS.to_vec(),
        redis_available,
        requires_restart: REQUIRES_RESTART.to_vec(),
        config,
    })
}

/// `PUT /api/config` — validate and persist overrides, then notify the
/// param-updates channel. Redis down → honest `503` (the write cannot be
/// persisted anywhere).
#[tracing::instrument(skip_all)]
pub(crate) async fn put_config_handler(
    State(state): State<Arc<JanusState>>,
    Extension(param_manager): Extension<Arc<ParamManager>>,
    Json(update): Json<JanusConfigUpdate>,
) -> Response {
    let fields = match validate_update(&update) {
        Ok(fields) => fields,
        Err(reason) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": reason })),
            )
                .into_response();
        }
    };

    let channel = param_manager.updates_channel();
    match write_overrides(&state, &channel, &fields).await {
        Ok(()) => {
            let saved: Vec<&str> = fields.iter().map(|(f, _)| *f).collect();
            tracing::info!(?saved, "Janus config overrides saved to Redis");
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ok": true,
                    "saved": saved,
                    "source": "redis",
                    "requires_restart": REQUIRES_RESTART,
                    "message": "Overrides saved. These are boot-time settings — they take effect after the owning janus/optimizer process restarts.",
                })),
            )
                .into_response()
        }
        Err(reason) => {
            tracing::warn!(error = %reason, "Janus config PUT failed: Redis unavailable");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "ok": false,
                    "error": "redis_unavailable",
                    "message": format!("Cannot persist config overrides: {reason}"),
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup<'a>(map: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            map.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn defaults_mirror_consumer_fallbacks_when_env_is_empty() {
        let d = defaults_from(|_| None);
        assert_eq!(d.optimize_assets, "BTC,ETH,SOL");
        assert_eq!(d.optimize_interval, "15m");
        assert_eq!(d.optimize_trials, 100);
        assert_eq!(d.optimize_historical_days, 30);
        assert_eq!(d.data_kline_intervals, "1m,5m");
        assert!(!d.janus_auto_start);
        assert_eq!(d.janus_bootstrap_days, 30);
    }

    #[test]
    fn defaults_read_the_documented_env_vars() {
        let env = [
            ("OPTIMIZE_ASSETS", "BTC,DOGE"),
            ("OPTIMIZE_INTERVAL", "6h"),
            ("OPTIMIZE_TRIALS", "250"),
            ("OPTIMIZE_HISTORICAL_DAYS", "60"),
            ("DATA_KLINE_INTERVALS", "1m,5m,15m"),
            ("JANUS_AUTO_START", "true"),
            ("JANUS_BOOTSTRAP_DAYS", "14"),
        ];
        let d = defaults_from(lookup(&env));
        assert_eq!(d.optimize_assets, "BTC,DOGE");
        assert_eq!(d.optimize_interval, "6h");
        assert_eq!(d.optimize_trials, 250);
        assert_eq!(d.optimize_historical_days, 60);
        assert_eq!(d.data_kline_intervals, "1m,5m,15m");
        assert!(d.janus_auto_start);
        assert_eq!(d.janus_bootstrap_days, 14);
    }

    #[test]
    fn deployed_default_interval_is_in_the_valid_set() {
        // The fks stack deploys OPTIMIZE_INTERVAL=6h; the UI select must be
        // able to render the effective value.
        assert!(VALID_INTERVALS.contains(&"6h"));
        // And the optimizer runner's own default:
        assert!(VALID_INTERVALS.contains(&"15m"));
    }

    #[test]
    fn apply_overrides_overlays_and_reports_applied_fields() {
        let mut config = defaults_from(|_| None);
        let overrides: HashMap<String, String> = [
            ("optimize_trials".to_string(), "500".to_string()),
            ("janus_auto_start".to_string(), "true".to_string()),
            ("bogus_field".to_string(), "x".to_string()),
            (
                "optimize_historical_days".to_string(),
                "not-a-number".to_string(),
            ),
        ]
        .into();
        let applied = apply_overrides(&mut config, &overrides);
        assert_eq!(applied, vec!["janus_auto_start", "optimize_trials"]);
        assert_eq!(config.optimize_trials, 500);
        assert!(config.janus_auto_start);
        // Unparseable / unknown overrides leave the defaults intact.
        assert_eq!(config.optimize_historical_days, 30);
        assert_eq!(config.optimize_assets, "BTC,ETH,SOL");
    }

    #[test]
    fn validate_normalizes_assets_and_kline_intervals() {
        let update = JanusConfigUpdate {
            optimize_assets: Some(" btc, eth ,sol ".into()),
            data_kline_intervals: Some("1m , 5m".into()),
            ..Default::default()
        };
        let fields = validate_update(&update).unwrap();
        assert!(fields.contains(&("optimize_assets", "BTC,ETH,SOL".to_string())));
        assert!(fields.contains(&("data_kline_intervals", "1m,5m".to_string())));
    }

    #[test]
    fn validate_rejects_bad_values() {
        let bad_interval = JanusConfigUpdate {
            optimize_interval: Some("7q".into()),
            ..Default::default()
        };
        assert!(
            validate_update(&bad_interval)
                .unwrap_err()
                .contains("optimize_interval")
        );

        let bad_trials = JanusConfigUpdate {
            optimize_trials: Some(0),
            ..Default::default()
        };
        assert!(
            validate_update(&bad_trials)
                .unwrap_err()
                .contains("optimize_trials")
        );

        let bad_days = JanusConfigUpdate {
            optimize_historical_days: Some(9_999),
            ..Default::default()
        };
        assert!(
            validate_update(&bad_days)
                .unwrap_err()
                .contains("optimize_historical_days")
        );

        let bad_kline = JanusConfigUpdate {
            data_kline_intervals: Some("1m,17m".into()),
            ..Default::default()
        };
        assert!(
            validate_update(&bad_kline)
                .unwrap_err()
                .contains("data_kline_intervals")
        );

        let bad_symbol = JanusConfigUpdate {
            optimize_assets: Some("BTC,ET;H".into()),
            ..Default::default()
        };
        assert!(
            validate_update(&bad_symbol)
                .unwrap_err()
                .contains("optimize_assets")
        );

        assert!(
            validate_update(&JanusConfigUpdate::default())
                .unwrap_err()
                .contains("no recognized config fields")
        );
    }

    #[test]
    fn bootstrap_days_zero_means_skip_and_is_allowed() {
        let update = JanusConfigUpdate {
            janus_bootstrap_days: Some(0),
            ..Default::default()
        };
        let fields = validate_update(&update).unwrap();
        assert_eq!(fields, vec![("janus_bootstrap_days", "0".to_string())]);
    }
}
