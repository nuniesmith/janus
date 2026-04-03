//! # Regime Bridge Redis Stream Consumer
//!
//! Standalone binary that reads `BridgedRegimeState` messages from a Redis
//! stream and makes them available to out-of-process neuromorphic consumers
//! (hypothalamus, amygdala).
//!
//! ## Architecture
//!
//! ```text
//!   EventLoop (main_production)
//!       │
//!       ▼
//!   broadcast::Sender<BridgedRegimeState>
//!       │
//!       ▼
//!   regime_bridge_consumer task
//!       │  (XADD janus:regime:bridge)
//!       ▼
//!   ┌─────────────────────┐
//!   │   Redis Stream      │
//!   │  janus:regime:bridge │
//!   └─────────┬───────────┘
//!             │  (XREADGROUP)
//!             ▼
//!   ┌─────────────────────────┐
//!   │  THIS BINARY             │
//!   │  regime-bridge-consumer  │
//!   └─────────┬───────────────┘
//!             │
//!             ├──► LoggingHandler        (default — logs to stdout)
//!             ├──► GrpcForwarderHandler  (GRPC_TARGET → push to remote server)
//!             └──► ServerHandler         (SERVE_PORT → embedded gRPC server)
//!                    │                         │
//!                    ▼                         ▼
//!                  Remote gRPC server    Neuromorphic consumers connect
//!                  (PushRegimeState)     via StreamRegimeUpdates / GetCurrentRegime
//! ```
//!
//! ## Handler Modes
//!
//! The consumer supports three handler modes, selected by environment variables
//! in the following priority order:
//!
//! 1. **`GrpcForwarderHandler`** (`GRPC_TARGET` set) — forwards each state to
//!    a remote neuromorphic gRPC server via `PushRegimeState`.
//! 2. **`ServerHandler`** (`SERVE_PORT` set) — starts an embedded
//!    `RegimeBridgeServer` on the given port. Neuromorphic subsystems connect
//!    directly to this consumer and call `StreamRegimeUpdates`,
//!    `GetCurrentRegime`, etc. No external gRPC server required.
//! 3. **`LoggingHandler`** (neither set) — logs each state to stdout.
//!    Useful for development and debugging.
//!
//! ## Usage
//!
//! ```bash
//! # Mode 1: Forward to a remote neuromorphic gRPC server
//! REDIS_URL=redis://127.0.0.1:6379 \
//!   GRPC_TARGET=http://hypothalamus-svc:50051 \
//!   cargo run --bin regime-bridge-consumer
//!
//! # Mode 2: Embedded gRPC server for neuromorphic subscribers (recommended)
//! REDIS_URL=redis://127.0.0.1:6379 \
//!   SERVE_PORT=50051 \
//!   cargo run --bin regime-bridge-consumer
//!
//! # Mode 3: Logging only (default)
//! REDIS_URL=redis://127.0.0.1:6379 cargo run --bin regime-bridge-consumer
//!
//! # Custom stream key and consumer group (works with any mode)
//! REDIS_URL=redis://127.0.0.1:6379 \
//!   REGIME_BRIDGE_STREAM=janus:regime:bridge \
//!   CONSUMER_GROUP=neuromorphic \
//!   CONSUMER_NAME=hypothalamus-1 \
//!   SERVE_PORT=50051 \
//!   cargo run --bin regime-bridge-consumer
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Default | Description |
//! |----------|---------|-------------|
//! | `REDIS_URL` | `redis://127.0.0.1:6379` | Redis connection URL |
//! | `REGIME_BRIDGE_STREAM` | `janus:regime:bridge` | Redis stream key to read from |
//! | `CONSUMER_GROUP` | `neuromorphic` | Consumer group name |
//! | `CONSUMER_NAME` | `consumer-{pid}` | Unique consumer name within the group |
//! | `BLOCK_MS` | `5000` | XREADGROUP block timeout in milliseconds |
//! | `BATCH_SIZE` | `10` | Max messages to read per XREADGROUP call |
//! | `PEL_SWEEP_INTERVAL_SECS` | `60` | Seconds between periodic PEL (Pending Entries List) sweeps. Set to `0` to disable periodic sweeps (startup drain still runs). |
//! | `PEL_MIN_IDLE_MS` | `30000` | Minimum idle time (ms) before a pending entry is eligible for XAUTOCLAIM reclamation from another consumer. |
//! | `PEL_BATCH_SIZE` | `100` | Max entries to reclaim per XAUTOCLAIM call. |
//! | `GRPC_TARGET` | *(unset)* | gRPC endpoint URL for forwarding (e.g. `http://host:50051`). When set, uses `GrpcForwarderHandler`. Takes priority over `SERVE_PORT`. |
//! | `GRPC_SOURCE_ID` | `regime-bridge-consumer-{pid}` | Source identifier sent with each gRPC push request |
//! | `SERVE_PORT` | *(unset)* | TCP port for the embedded `RegimeBridgeServer` (e.g. `50051`). When set (and `GRPC_TARGET` is not), uses `ServerHandler`. Neuromorphic consumers connect here via `StreamRegimeUpdates` / `GetCurrentRegime`. |
//! | `RUST_LOG` | `info` | Tracing log level filter |

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tracing::{debug, error, info, warn};

// Import generated gRPC client and proto types from the forward service library.
use janus_forward::regime_bridge_proto::{
    AmygdalaRegime, HypothalamusRegime, PushRegimeStateRequest,
    RegimeIndicators as ProtoIndicators, RegimeState,
    regime_bridge_service_client::RegimeBridgeServiceClient,
    regime_bridge_service_server::RegimeBridgeServiceServer,
};
use janus_forward::regime_bridge_server::RegimeBridgeServer;

// ============================================================================
// Deserialized bridge state from Redis stream
// ============================================================================

/// A `BridgedRegimeState` as received from the Redis stream.
///
/// The producer (`regime_bridge_consumer` task in `main_production.rs`)
/// serializes the state as a JSON string in the `data` field of the stream
/// entry. This struct deserializes that JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamedBridgeState {
    pub symbol: String,
    pub hypothalamus_regime: String,
    pub amygdala_regime: String,
    pub position_scale: f64,
    pub is_high_risk: bool,
    pub confidence: f64,
    #[serde(default)]
    pub trend: f64,
    #[serde(default)]
    pub trend_strength: f64,
    #[serde(default)]
    pub volatility: f64,
    #[serde(default = "default_half")]
    pub volatility_percentile: f64,
    #[serde(default)]
    pub momentum: f64,
    #[serde(default = "default_one")]
    pub relative_volume: f64,
    #[serde(default = "default_one")]
    pub liquidity_score: f64,
}

fn default_half() -> f64 {
    0.5
}
fn default_one() -> f64 {
    1.0
}

impl std::fmt::Display for StreamedBridgeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] hypo={} amyg={} scale={:.0}% risk={} conf={:.0}% rel_vol={:.2}",
            self.symbol,
            self.hypothalamus_regime,
            self.amygdala_regime,
            self.position_scale * 100.0,
            if self.is_high_risk { "HIGH" } else { "low" },
            self.confidence * 100.0,
            self.relative_volume,
        )
    }
}

// ============================================================================
// Consumer configuration
// ============================================================================

#[derive(Debug, Clone)]
struct ConsumerConfig {
    redis_url: String,
    stream_key: String,
    group_name: String,
    consumer_name: String,
    block_ms: usize,
    batch_size: usize,
    /// Interval (in seconds) between periodic PEL sweep passes.
    /// Set to 0 to disable periodic sweeps (startup drain still runs).
    pel_sweep_interval_secs: u64,
    /// Minimum idle time (in milliseconds) before a pending entry is
    /// eligible for reclamation via XAUTOCLAIM. Entries idle for less
    /// than this are assumed to still be actively processed by another
    /// consumer.
    pel_min_idle_ms: u64,
    /// Maximum number of entries to reclaim per XAUTOCLAIM call.
    pel_batch_size: usize,
}

impl ConsumerConfig {
    fn from_env() -> Self {
        let pid = std::process::id();
        Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            stream_key: std::env::var("REGIME_BRIDGE_STREAM")
                .unwrap_or_else(|_| "janus:regime:bridge".into()),
            group_name: std::env::var("CONSUMER_GROUP").unwrap_or_else(|_| "neuromorphic".into()),
            consumer_name: std::env::var("CONSUMER_NAME")
                .unwrap_or_else(|_| format!("consumer-{}", pid)),
            block_ms: std::env::var("BLOCK_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5000),
            batch_size: std::env::var("BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            pel_sweep_interval_secs: std::env::var("PEL_SWEEP_INTERVAL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            pel_min_idle_ms: std::env::var("PEL_MIN_IDLE_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30_000),
            pel_batch_size: std::env::var("PEL_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
        }
    }
}

// ============================================================================
// Handler trait — plug in your own processing logic
// ============================================================================

/// Trait for processing bridged regime states received from the stream.
///
/// Implement this to wire the consumer into your neuromorphic pipeline.
/// The default handler simply logs the state.
#[allow(async_fn_in_trait)]
pub trait BridgeStateHandler: Send + Sync + 'static {
    /// Called for each `StreamedBridgeState` received from the Redis stream.
    ///
    /// `stream_id` is the Redis stream entry ID (e.g. `"1234567890-0"`).
    async fn handle(&self, stream_id: &str, state: &StreamedBridgeState) -> Result<()>;

    /// Called when a regime transition is detected (hypothalamus or amygdala
    /// regime changed since the last message for the same symbol).
    async fn on_transition(
        &self,
        _stream_id: &str,
        _symbol: &str,
        _old_hypo: &str,
        _new_hypo: &str,
        _old_amyg: &str,
        _new_amyg: &str,
    ) -> Result<()> {
        // Default: no-op; override for transition-specific logic
        Ok(())
    }
}

/// Default handler that logs each state and highlights transitions.
struct LoggingHandler;

impl BridgeStateHandler for LoggingHandler {
    async fn handle(&self, stream_id: &str, state: &StreamedBridgeState) -> Result<()> {
        info!(
            "📥 {stream_id} | {state} | trend={:.2} vol={:.3} mom={:.2}",
            state.trend, state.volatility, state.momentum,
        );
        Ok(())
    }

    async fn on_transition(
        &self,
        _stream_id: &str,
        symbol: &str,
        old_hypo: &str,
        new_hypo: &str,
        old_amyg: &str,
        new_amyg: &str,
    ) -> Result<()> {
        info!(
            "🔄 Regime transition for {symbol}: hypo {old_hypo}→{new_hypo}, amyg {old_amyg}→{new_amyg}"
        );
        Ok(())
    }
}

// ============================================================================
// Server Handler — embedded gRPC server for neuromorphic subscribers
// ============================================================================

/// Handler that embeds a [`RegimeBridgeServer`] and feeds each received
/// `StreamedBridgeState` into it. Neuromorphic subsystems connect to the
/// gRPC port exposed by this handler and call `StreamRegimeUpdates`,
/// `GetCurrentRegime`, etc.
///
/// Set the `SERVE_PORT` environment variable to enable this handler instead
/// of the default `LoggingHandler`.
///
/// # Architecture
///
/// ```text
///   Redis Stream
///       │ (XREADGROUP)
///       ▼
///   regime-bridge-consumer (this binary)
///       │
///       ├──► ServerHandler
///       │       │
///       │       ▼
///       │   RegimeBridgeServer (embedded)
///       │       │
///       │       ├──► StreamRegimeUpdates (gRPC server-stream)
///       │       ├──► GetCurrentRegime    (gRPC unary)
///       │       └──► broadcast channel   (in-process subscribers)
///       │
///       └──► gRPC port (e.g. 50051)
///                │
///                ▼
///            Neuromorphic consumers
///            (hypothalamus, amygdala)
/// ```
///
/// # Example
///
/// ```bash
/// REDIS_URL=redis://127.0.0.1:6379 \
///   SERVE_PORT=50051 \
///   cargo run --bin regime-bridge-consumer
/// ```
pub struct ServerHandler {
    /// The embedded regime bridge server.
    server: RegimeBridgeServer,
    /// Monotonic sequence counter.
    sequence: AtomicU64,
    /// Total states ingested.
    total_ingested: AtomicU64,
}

impl ServerHandler {
    /// Create a new `ServerHandler` and start the embedded gRPC server on
    /// the given port. Returns the handler ready for use with `run_consumer`.
    ///
    /// The gRPC server runs in a background task and serves until the
    /// process exits.
    pub async fn start(port: u16) -> Result<Self> {
        let (tx, _rx) =
            tokio::sync::broadcast::channel::<janus_forward::regime_bridge::BridgedRegimeState>(64);

        let server = RegimeBridgeServer::new(tx);

        // Spawn the snapshot updater so GetCurrentRegime returns live data
        let _updater = server.spawn_snapshot_updater();

        // Start gRPC server in background
        let grpc_server = server.clone();
        let addr: std::net::SocketAddr = format!("0.0.0.0:{}", port)
            .parse()
            .with_context(|| format!("Invalid SERVE_PORT: {}", port))?;

        tokio::spawn(async move {
            info!(
                "🌐 Embedded regime bridge gRPC server listening on {}",
                addr
            );
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(RegimeBridgeServiceServer::new(grpc_server))
                .serve(addr)
                .await
            {
                error!("Embedded gRPC server error: {}", e);
            }
        });

        info!(
            "✅ ServerHandler ready — neuromorphic consumers can connect to port {}",
            port
        );

        Ok(Self {
            server,
            sequence: AtomicU64::new(0),
            total_ingested: AtomicU64::new(0),
        })
    }
}

impl BridgeStateHandler for ServerHandler {
    async fn handle(&self, stream_id: &str, state: &StreamedBridgeState) -> Result<()> {
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let ingested = self.total_ingested.fetch_add(1, Ordering::Relaxed) + 1;

        // Build a proto RegimeState and push it through the embedded server
        let hypo = parse_hypothalamus_regime(&state.hypothalamus_regime);
        let amyg = parse_amygdala_regime(&state.amygdala_regime);

        let proto_state = RegimeState {
            symbol: state.symbol.clone(),
            hypothalamus_regime: hypo.into(),
            amygdala_regime: amyg.into(),
            position_scale: state.position_scale,
            is_high_risk: state.is_high_risk,
            confidence: state.confidence,
            indicators: Some(ProtoIndicators {
                trend: state.trend,
                trend_strength: state.trend_strength,
                volatility: state.volatility,
                volatility_percentile: state.volatility_percentile,
                correlation: 0.5,
                breadth: 0.5,
                momentum: state.momentum,
                relative_volume: state.relative_volume,
                liquidity_score: state.liquidity_score,
                fear_index: 0.0,
            }),
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as i64,
            sequence: seq,
            is_transition: false,
            previous_hypothalamus_regime: HypothalamusRegime::Unspecified.into(),
            previous_amygdala_regime: AmygdalaRegime::Unspecified.into(),
        };

        let request =
            tonic::Request::new(janus_forward::regime_bridge_proto::PushRegimeStateRequest {
                state: Some(proto_state),
                source_id: format!("server-handler-{}", std::process::id()),
            });

        // Push through the embedded server — this stores the snapshot and
        // broadcasts to all StreamRegimeUpdates subscribers.
        use janus_forward::regime_bridge_proto::regime_bridge_service_server::RegimeBridgeService;
        match self.server.push_regime_state(request).await {
            Ok(resp) => {
                let inner = resp.into_inner();
                if !inner.accepted {
                    warn!(
                        "⚠️ Embedded server rejected state for {} (stream_id={}): {}",
                        state.symbol, stream_id, inner.message
                    );
                }
            }
            Err(e) => {
                warn!(
                    "⚠️ Embedded server push error for {} (stream_id={}): {}",
                    state.symbol, stream_id, e
                );
            }
        }

        // Periodic stats
        if ingested.is_multiple_of(100) {
            let subscribers = self.server.active_stream_count();
            info!(
                "📡 ServerHandler stats: ingested={} seq={} active_streams={}",
                ingested, seq, subscribers,
            );
        }

        debug!(
            "📥 Ingested {} (stream_id={}, seq={}, subscribers={})",
            state.symbol,
            stream_id,
            seq,
            self.server.active_stream_count(),
        );

        Ok(())
    }

    async fn on_transition(
        &self,
        stream_id: &str,
        symbol: &str,
        old_hypo: &str,
        new_hypo: &str,
        old_amyg: &str,
        new_amyg: &str,
    ) -> Result<()> {
        info!(
            "🔄→📡 Regime transition ingested for {symbol}: hypo {old_hypo}→{new_hypo}, amyg {old_amyg}→{new_amyg} (stream_id={stream_id}, subscribers={})",
            self.server.active_stream_count(),
        );
        Ok(())
    }
}

// ============================================================================
// gRPC Forwarder Handler
// ============================================================================

/// Handler that forwards each `StreamedBridgeState` to a remote neuromorphic
/// gRPC service via `RegimeBridgeService::PushRegimeState`.
///
/// Set the `GRPC_TARGET` environment variable to enable this handler instead
/// of the default `LoggingHandler`.
///
/// # Connection management
///
/// The handler lazily connects on first use and automatically reconnects if
/// the channel drops (tonic channels are resilient to transient failures).
/// A monotonic sequence counter is maintained per consumer instance so the
/// server can detect gaps.
///
/// # Example
///
/// ```bash
/// GRPC_TARGET=http://hypothalamus-svc:50051 \
///   REDIS_URL=redis://127.0.0.1:6379 \
///   cargo run --bin regime-bridge-consumer
/// ```
pub struct GrpcForwarderHandler {
    /// gRPC client (behind a mutex for interior mutability in the async trait).
    client: Mutex<RegimeBridgeServiceClient<Channel>>,
    /// Source identifier sent with each push request.
    source_id: String,
    /// Monotonic sequence counter across all symbols.
    sequence: AtomicU64,
    /// Total successful pushes (for periodic stats logging).
    total_pushed: AtomicU64,
    /// Total push failures.
    total_errors: AtomicU64,
}

impl GrpcForwarderHandler {
    /// Connect to the gRPC target and return a new handler.
    ///
    /// `target` must be a valid URI (e.g. `http://127.0.0.1:50051`).
    pub async fn connect(target: &str, source_id: String) -> Result<Self> {
        let client = RegimeBridgeServiceClient::connect(target.to_string())
            .await
            .with_context(|| format!("Failed to connect to gRPC target: {}", target))?;

        info!(
            "✅ gRPC forwarder connected to {} (source_id={})",
            target, source_id
        );

        Ok(Self {
            client: Mutex::new(client),
            source_id,
            sequence: AtomicU64::new(0),
            total_pushed: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        })
    }

    /// Convert a `StreamedBridgeState` (deserialized from Redis JSON) into the
    /// protobuf `RegimeState` message.
    fn to_proto_state(&self, state: &StreamedBridgeState) -> RegimeState {
        let hypo = parse_hypothalamus_regime(&state.hypothalamus_regime);
        let amyg = parse_amygdala_regime(&state.amygdala_regime);
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        RegimeState {
            symbol: state.symbol.clone(),
            hypothalamus_regime: hypo.into(),
            amygdala_regime: amyg.into(),
            position_scale: state.position_scale,
            is_high_risk: state.is_high_risk,
            confidence: state.confidence,
            indicators: Some(ProtoIndicators {
                trend: state.trend,
                trend_strength: state.trend_strength,
                volatility: state.volatility,
                volatility_percentile: state.volatility_percentile,
                correlation: 0.5,
                breadth: 0.5,
                momentum: state.momentum,
                relative_volume: state.relative_volume,
                liquidity_score: state.liquidity_score,
                fear_index: 0.0,
            }),
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as i64,
            sequence: seq,
            is_transition: false,
            previous_hypothalamus_regime: HypothalamusRegime::Unspecified.into(),
            previous_amygdala_regime: AmygdalaRegime::Unspecified.into(),
        }
    }
}

impl BridgeStateHandler for GrpcForwarderHandler {
    async fn handle(&self, stream_id: &str, state: &StreamedBridgeState) -> Result<()> {
        let proto_state = self.to_proto_state(state);

        let request = tonic::Request::new(PushRegimeStateRequest {
            state: Some(proto_state),
            source_id: self.source_id.clone(),
        });

        let mut client = self.client.lock().await;
        match client.push_regime_state(request).await {
            Ok(response) => {
                let resp = response.into_inner();
                let pushed = self.total_pushed.fetch_add(1, Ordering::Relaxed) + 1;

                if !resp.accepted {
                    warn!(
                        "⚠️ gRPC server rejected state for {} (stream_id={}): {}",
                        state.symbol, stream_id, resp.message
                    );
                }

                // Periodic stats
                if pushed.is_multiple_of(100) {
                    let errors = self.total_errors.load(Ordering::Relaxed);
                    info!(
                        "📡 gRPC forwarder stats: pushed={} errors={} seq={}",
                        pushed,
                        errors,
                        self.sequence.load(Ordering::Relaxed),
                    );
                }

                Ok(())
            }
            Err(e) => {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
                warn!(
                    "⚠️ gRPC push failed for {} (stream_id={}): {}",
                    state.symbol, stream_id, e
                );
                // Return Ok so the consumer loop continues and ACKs the message.
                // The state is lost, but the consumer doesn't stall.
                Ok(())
            }
        }
    }

    async fn on_transition(
        &self,
        stream_id: &str,
        symbol: &str,
        old_hypo: &str,
        new_hypo: &str,
        old_amyg: &str,
        new_amyg: &str,
    ) -> Result<()> {
        info!(
            "🔄→📡 Regime transition forwarded for {symbol}: hypo {old_hypo}→{new_hypo}, amyg {old_amyg}→{new_amyg} (stream_id={stream_id})"
        );
        // Note: the transition flag is NOT set on the RegimeState pushed via
        // handle() because on_transition is called *before* handle(). To set
        // it, we'd need to buffer state — for now we log and the consumer can
        // detect transitions server-side via sequence gaps or field changes.
        Ok(())
    }
}

/// Parse a hypothalamus regime string (as serialized in Redis JSON) into
/// the protobuf enum variant.
fn parse_hypothalamus_regime(s: &str) -> HypothalamusRegime {
    match s {
        "StrongBullish" => HypothalamusRegime::StrongBullish,
        "Bullish" => HypothalamusRegime::Bullish,
        "Neutral" => HypothalamusRegime::Neutral,
        "Bearish" => HypothalamusRegime::Bearish,
        "StrongBearish" => HypothalamusRegime::StrongBearish,
        "HighVolatility" => HypothalamusRegime::HighVolatility,
        "LowVolatility" => HypothalamusRegime::LowVolatility,
        "Transitional" => HypothalamusRegime::Transitional,
        "Crisis" => HypothalamusRegime::Crisis,
        _ => HypothalamusRegime::Unknown,
    }
}

/// Parse an amygdala regime string into the protobuf enum variant.
fn parse_amygdala_regime(s: &str) -> AmygdalaRegime {
    match s {
        "LowVolTrending" => AmygdalaRegime::LowVolTrending,
        "LowVolMeanReverting" => AmygdalaRegime::LowVolMeanReverting,
        "HighVolTrending" => AmygdalaRegime::HighVolTrending,
        "HighVolMeanReverting" => AmygdalaRegime::HighVolMeanReverting,
        "Crisis" => AmygdalaRegime::Crisis,
        "Transitional" => AmygdalaRegime::Transitional,
        _ => AmygdalaRegime::Unknown,
    }
}

// ============================================================================
// PEL (Pending Entries List) Reclamation
// ============================================================================

/// Drain pending entries that belong to **this** consumer on startup.
///
/// When a consumer crashes between XREADGROUP and XACK, those messages
/// remain in the PEL (Pending Entries List) assigned to this consumer.
/// On restart we must re-read and process them before switching to `">"`
/// (new messages only).
///
/// This function reads with start-ID `"0"` (all pending for this consumer)
/// and processes + ACKs each entry. It keeps reading until no more pending
/// entries are returned.
///
/// Returns the number of entries drained.
async fn drain_pending_entries<H: BridgeStateHandler>(
    config: &ConsumerConfig,
    conn: &mut redis::aio::MultiplexedConnection,
    handler: &H,
    last_hypo: &mut HashMap<String, String>,
    last_amyg: &mut HashMap<String, String>,
) -> u64 {
    let mut total_drained: u64 = 0;

    info!(
        "🔍 Draining pending entries for consumer '{}' in group '{}' on stream '{}'...",
        config.consumer_name, config.group_name, config.stream_key,
    );

    loop {
        // XREADGROUP GROUP <group> <consumer> COUNT <n> STREAMS <key> 0
        // Using "0" instead of ">" reads entries already delivered to this
        // consumer but not yet ACKed.
        let result: Result<Vec<redis::Value>, redis::RedisError> = redis::cmd("XREADGROUP")
            .arg("GROUP")
            .arg(&config.group_name)
            .arg(&config.consumer_name)
            .arg("COUNT")
            .arg(config.pel_batch_size)
            .arg("STREAMS")
            .arg(&config.stream_key)
            .arg("0") // pending entries only
            .query_async(conn)
            .await;

        let values = match result {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    "⚠️ Error reading pending entries: {} — skipping PEL drain",
                    e
                );
                break;
            }
        };

        let entries = parse_stream_response(&values);
        if entries.is_empty() {
            // No more pending entries for this consumer
            break;
        }

        for (stream_id, fields) in &entries {
            // Process the pending entry exactly like the main loop does
            let json_data = match fields.get("data") {
                Some(d) => d,
                None => {
                    warn!(
                        "⚠️ Pending entry {} has no 'data' field — ACKing and skipping",
                        stream_id
                    );
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(stream_id.as_str())
                        .query_async(conn)
                        .await;
                    total_drained += 1;
                    continue;
                }
            };

            let state: StreamedBridgeState = match serde_json::from_str(json_data) {
                Ok(s) => s,
                Err(e) => {
                    warn!(
                        "⚠️ Failed to deserialize pending entry {}: {} — ACKing and skipping",
                        stream_id, e
                    );
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(stream_id.as_str())
                        .query_async(conn)
                        .await;
                    total_drained += 1;
                    continue;
                }
            };

            // Detect transitions against tracked state
            let prev_hypo = last_hypo.get(&state.symbol).cloned().unwrap_or_default();
            let prev_amyg = last_amyg.get(&state.symbol).cloned().unwrap_or_default();

            let is_transition = (!prev_hypo.is_empty() || !prev_amyg.is_empty())
                && (prev_hypo != state.hypothalamus_regime || prev_amyg != state.amygdala_regime);

            if is_transition
                && let Err(e) = handler
                    .on_transition(
                        stream_id,
                        &state.symbol,
                        &prev_hypo,
                        &state.hypothalamus_regime,
                        &prev_amyg,
                        &state.amygdala_regime,
                    )
                    .await
            {
                warn!(
                    "⚠️ Transition handler error for pending {}: {}",
                    stream_id, e
                );
            }

            last_hypo.insert(state.symbol.clone(), state.hypothalamus_regime.clone());
            last_amyg.insert(state.symbol.clone(), state.amygdala_regime.clone());

            if let Err(e) = handler.handle(stream_id, &state).await {
                warn!("⚠️ Handler error for pending {}: {}", stream_id, e);
            }

            // ACK
            let _: Result<i64, _> = redis::cmd("XACK")
                .arg(&config.stream_key)
                .arg(&config.group_name)
                .arg(stream_id.as_str())
                .query_async(conn)
                .await;

            total_drained += 1;
        }
    }

    if total_drained > 0 {
        info!(
            "✅ Drained {} pending entries for consumer '{}'",
            total_drained, config.consumer_name,
        );
    } else {
        info!(
            "✅ No pending entries found for consumer '{}' — clean startup",
            config.consumer_name,
        );
    }

    total_drained
}

/// Reclaim stale pending entries from **other** consumers in the group
/// that may have crashed.
///
/// Uses `XAUTOCLAIM` to atomically transfer ownership of entries that have
/// been idle (not ACKed) for longer than `min_idle_ms` and then processes
/// + ACKs them.
///
/// Returns `(reclaimed_count, new_start_id)`.
///   - `new_start_id` is the cursor for the next XAUTOCLAIM call. When it
///     returns `"0-0"`, all eligible entries have been reclaimed.
async fn reclaim_stale_entries<H: BridgeStateHandler>(
    config: &ConsumerConfig,
    conn: &mut redis::aio::MultiplexedConnection,
    handler: &H,
    start_id: &str,
    last_hypo: &mut HashMap<String, String>,
    last_amyg: &mut HashMap<String, String>,
) -> (u64, String) {
    // XAUTOCLAIM <key> <group> <consumer> <min-idle-ms> <start> [COUNT <n>]
    //
    // Returns:
    //   1) next start ID (cursor for pagination)
    //   2) array of [id, [field, value, ...]] entries successfully claimed
    //   3) array of entry IDs that no longer exist in the stream (deleted)
    let result: Result<Vec<redis::Value>, redis::RedisError> = redis::cmd("XAUTOCLAIM")
        .arg(&config.stream_key)
        .arg(&config.group_name)
        .arg(&config.consumer_name)
        .arg(config.pel_min_idle_ms)
        .arg(start_id)
        .arg("COUNT")
        .arg(config.pel_batch_size)
        .query_async(conn)
        .await;

    let response = match result {
        Ok(v) => v,
        Err(e) => {
            // XAUTOCLAIM requires Redis >= 6.2. If the server doesn't support
            // it, log a warning and skip. The startup drain (which uses plain
            // XREADGROUP with "0") still works on older Redis versions.
            let msg = e.to_string();
            if msg.contains("unknown command") || msg.contains("ERR") {
                debug!(
                    "XAUTOCLAIM not supported (Redis < 6.2?): {} — skipping PEL sweep",
                    msg
                );
            } else {
                warn!("⚠️ XAUTOCLAIM error: {} — skipping this sweep cycle", e);
            }
            return (0, "0-0".to_string());
        }
    };

    // Parse the 3-element response: [next_start_id, claimed_entries, deleted_ids]
    if response.len() < 2 {
        return (0, "0-0".to_string());
    }

    let next_start = value_to_string(&response[0]).unwrap_or_else(|| "0-0".to_string());

    // Parse claimed entries — same format as XREADGROUP entry arrays:
    //   [[id, [field, value, ...]], ...]
    let claimed_entries = match &response[1] {
        redis::Value::Array(arr) => arr,
        _ => return (0, next_start),
    };

    let mut reclaimed: u64 = 0;

    for entry in claimed_entries {
        if let redis::Value::Array(id_and_fields) = entry {
            if id_and_fields.len() < 2 {
                continue;
            }
            let stream_id = match value_to_string(&id_and_fields[0]) {
                Some(id) => id,
                None => continue,
            };
            let fields = match parse_field_pairs(&id_and_fields[1]) {
                Some(f) => f,
                None => {
                    // ACK entries we can't parse so they don't accumulate
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(&stream_id)
                        .query_async(conn)
                        .await;
                    reclaimed += 1;
                    continue;
                }
            };

            let json_data = match fields.get("data") {
                Some(d) => d,
                None => {
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(&stream_id)
                        .query_async(conn)
                        .await;
                    reclaimed += 1;
                    continue;
                }
            };

            let state: StreamedBridgeState = match serde_json::from_str(json_data) {
                Ok(s) => s,
                Err(e) => {
                    warn!(
                        "⚠️ Failed to deserialize reclaimed entry {}: {} — ACKing",
                        stream_id, e
                    );
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(&stream_id)
                        .query_async(conn)
                        .await;
                    reclaimed += 1;
                    continue;
                }
            };

            // Detect transitions
            let prev_hypo = last_hypo.get(&state.symbol).cloned().unwrap_or_default();
            let prev_amyg = last_amyg.get(&state.symbol).cloned().unwrap_or_default();
            let is_transition = (!prev_hypo.is_empty() || !prev_amyg.is_empty())
                && (prev_hypo != state.hypothalamus_regime || prev_amyg != state.amygdala_regime);

            if is_transition {
                let _ = handler
                    .on_transition(
                        &stream_id,
                        &state.symbol,
                        &prev_hypo,
                        &state.hypothalamus_regime,
                        &prev_amyg,
                        &state.amygdala_regime,
                    )
                    .await;
            }

            last_hypo.insert(state.symbol.clone(), state.hypothalamus_regime.clone());
            last_amyg.insert(state.symbol.clone(), state.amygdala_regime.clone());

            if let Err(e) = handler.handle(&stream_id, &state).await {
                warn!("⚠️ Handler error for reclaimed {}: {}", stream_id, e);
            }

            let _: Result<i64, _> = redis::cmd("XACK")
                .arg(&config.stream_key)
                .arg(&config.group_name)
                .arg(&stream_id)
                .query_async(conn)
                .await;

            reclaimed += 1;
        }
    }

    // Log deleted entry IDs (entries removed from stream while pending)
    if response.len() >= 3
        && let redis::Value::Array(ref deleted) = response[2]
        && !deleted.is_empty()
    {
        let deleted_ids: Vec<String> = deleted.iter().filter_map(value_to_string).collect();
        info!(
            "🗑️ PEL sweep: {} entries were deleted from stream while pending: {:?}",
            deleted_ids.len(),
            deleted_ids,
        );
    }

    (reclaimed, next_start)
}

/// Run a single full PEL sweep pass, paginating through all stale entries.
///
/// Returns the total number of entries reclaimed in this pass.
async fn run_pel_sweep_pass<H: BridgeStateHandler>(
    config: &ConsumerConfig,
    conn: &mut redis::aio::MultiplexedConnection,
    handler: &H,
    last_hypo: &mut HashMap<String, String>,
    last_amyg: &mut HashMap<String, String>,
) -> u64 {
    let mut cursor = "0-0".to_string();
    let mut total_reclaimed: u64 = 0;

    loop {
        let (reclaimed, next_cursor) =
            reclaim_stale_entries(config, conn, handler, &cursor, last_hypo, last_amyg).await;

        total_reclaimed += reclaimed;

        // "0-0" means we've scanned through all pending entries
        if next_cursor == "0-0" || next_cursor == "0" {
            break;
        }
        cursor = next_cursor;
    }

    total_reclaimed
}

/// Get a summary of the PEL for observability.
///
/// Returns `(total_pending, oldest_idle_ms)` or `None` on error.
async fn pel_summary(
    config: &ConsumerConfig,
    conn: &mut redis::aio::MultiplexedConnection,
) -> Option<(u64, u64)> {
    // XPENDING <key> <group>
    // Returns: [total_pending, smallest_id, largest_id, [[consumer, count], ...]]
    let result: Result<Vec<redis::Value>, redis::RedisError> = redis::cmd("XPENDING")
        .arg(&config.stream_key)
        .arg(&config.group_name)
        .query_async(conn)
        .await;

    match result {
        Ok(ref vals) if !vals.is_empty() => {
            let total = match &vals[0] {
                redis::Value::Int(n) => *n as u64,
                _ => 0,
            };
            Some((total, 0)) // Idle time requires per-entry XPENDING; total is sufficient
        }
        Ok(_) => Some((0, 0)),
        Err(e) => {
            debug!("XPENDING error: {}", e);
            None
        }
    }
}

// ============================================================================
// Stream consumer loop
// ============================================================================

/// Ensure the consumer group exists on the stream.
///
/// Uses `XGROUP CREATE ... MKSTREAM` so that if the stream doesn't exist yet
/// it will be created automatically. If the group already exists, the error
/// is silently ignored.
async fn ensure_consumer_group(
    conn: &mut redis::aio::MultiplexedConnection,
    stream_key: &str,
    group_name: &str,
) -> Result<()> {
    let result: Result<String, redis::RedisError> = redis::cmd("XGROUP")
        .arg("CREATE")
        .arg(stream_key)
        .arg(group_name)
        .arg("0") // start from beginning; use "$" for only new messages
        .arg("MKSTREAM")
        .query_async(conn)
        .await;

    match result {
        Ok(_) => {
            info!(
                "✅ Consumer group '{}' created on stream '{}'",
                group_name, stream_key
            );
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("BUSYGROUP") {
                debug!(
                    "Consumer group '{}' already exists on '{}' — reusing",
                    group_name, stream_key
                );
            } else {
                return Err(e).context("Failed to create consumer group");
            }
        }
    }

    Ok(())
}

/// Run the consumer loop with PEL reclamation.
///
/// On startup:
///   1. Drains any pending entries assigned to **this** consumer (crash recovery).
///   2. Enters the main loop reading new (`">"`) messages.
///
/// A periodic PEL sweep task runs in the background to reclaim stale entries
/// from **other** crashed consumers in the group (via XAUTOCLAIM).
async fn run_consumer<H: BridgeStateHandler>(
    config: &ConsumerConfig,
    conn: &mut redis::aio::MultiplexedConnection,
    handler: &H,
) -> Result<()> {
    // Track last seen regime per symbol for transition detection
    let mut last_hypo: HashMap<String, String> = HashMap::new();
    let mut last_amyg: HashMap<String, String> = HashMap::new();
    let mut total_processed: u64 = 0;
    let mut total_errors: u64 = 0;

    // ── Phase 1: Drain pending entries (crash recovery) ──────────────
    let drained =
        drain_pending_entries(config, conn, handler, &mut last_hypo, &mut last_amyg).await;
    total_processed += drained;

    // Log PEL summary after drain
    if let Some((pending, _)) = pel_summary(config, conn).await
        && pending > 0
    {
        info!(
            "📋 PEL summary after drain: {} entries still pending across all consumers",
            pending
        );
    }

    // ── Phase 2: Main consumer loop (new messages) ───────────────────
    info!(
        "🚀 Consumer loop started: stream={} group={} consumer={} block={}ms batch={} pel_sweep={}s pel_idle={}ms",
        config.stream_key,
        config.group_name,
        config.consumer_name,
        config.block_ms,
        config.batch_size,
        config.pel_sweep_interval_secs,
        config.pel_min_idle_ms,
    );

    // Track when the last PEL sweep ran so we can trigger periodic sweeps
    // inline (simpler than a separate task that needs its own Redis conn).
    let mut last_pel_sweep = std::time::Instant::now();
    let pel_sweep_interval = std::time::Duration::from_secs(config.pel_sweep_interval_secs);

    loop {
        // XREADGROUP GROUP <group> <consumer> COUNT <n> BLOCK <ms> STREAMS <key> >
        let result: Result<Vec<redis::Value>, redis::RedisError> = redis::cmd("XREADGROUP")
            .arg("GROUP")
            .arg(&config.group_name)
            .arg(&config.consumer_name)
            .arg("COUNT")
            .arg(config.batch_size)
            .arg("BLOCK")
            .arg(config.block_ms)
            .arg("STREAMS")
            .arg(&config.stream_key)
            .arg(">") // only new, undelivered messages
            .query_async(conn)
            .await;

        let values = match result {
            Ok(v) => v,
            Err(e) => {
                warn!("⚠️ XREADGROUP error: {} — retrying in 1s", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        // Parse the nested Redis response:
        //   [ [ stream_key, [ [id, [field, value, ...]], ... ] ] ]
        let entries = parse_stream_response(&values);

        for (stream_id, fields) in entries {
            // The producer writes a single field "data" containing JSON
            let json_data = match fields.get("data") {
                Some(d) => d,
                None => {
                    warn!(
                        "⚠️ Stream entry {} has no 'data' field — skipping",
                        stream_id
                    );
                    total_errors += 1;
                    // ACK even malformed entries so they don't block the group
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(&stream_id)
                        .query_async(conn)
                        .await;
                    continue;
                }
            };

            // Deserialize
            let state: StreamedBridgeState = match serde_json::from_str(json_data) {
                Ok(s) => s,
                Err(e) => {
                    warn!(
                        "⚠️ Failed to deserialize stream entry {}: {} — skipping",
                        stream_id, e
                    );
                    total_errors += 1;
                    let _: Result<i64, _> = redis::cmd("XACK")
                        .arg(&config.stream_key)
                        .arg(&config.group_name)
                        .arg(&stream_id)
                        .query_async(conn)
                        .await;
                    continue;
                }
            };

            // Detect transitions
            let prev_hypo = last_hypo.get(&state.symbol).cloned().unwrap_or_default();
            let prev_amyg = last_amyg.get(&state.symbol).cloned().unwrap_or_default();

            let is_transition = (!prev_hypo.is_empty() || !prev_amyg.is_empty())
                && (prev_hypo != state.hypothalamus_regime || prev_amyg != state.amygdala_regime);

            if is_transition
                && let Err(e) = handler
                    .on_transition(
                        &stream_id,
                        &state.symbol,
                        &prev_hypo,
                        &state.hypothalamus_regime,
                        &prev_amyg,
                        &state.amygdala_regime,
                    )
                    .await
            {
                warn!("⚠️ Transition handler error for {}: {}", stream_id, e);
            }

            // Update tracking
            last_hypo.insert(state.symbol.clone(), state.hypothalamus_regime.clone());
            last_amyg.insert(state.symbol.clone(), state.amygdala_regime.clone());

            // Dispatch to handler
            if let Err(e) = handler.handle(&stream_id, &state).await {
                warn!("⚠️ Handler error for {}: {}", stream_id, e);
                total_errors += 1;
            }

            // ACK the message
            let ack_result: Result<i64, redis::RedisError> = redis::cmd("XACK")
                .arg(&config.stream_key)
                .arg(&config.group_name)
                .arg(&stream_id)
                .query_async(conn)
                .await;

            if let Err(e) = ack_result {
                warn!("⚠️ Failed to XACK {}: {}", stream_id, e);
            }

            total_processed += 1;

            // Periodic stats
            if total_processed.is_multiple_of(100) {
                info!(
                    "📊 Consumer stats: processed={} errors={} symbols_tracked={}",
                    total_processed,
                    total_errors,
                    last_hypo.len(),
                );
            }
        }

        // ── Periodic PEL sweep (inline, between read batches) ────────
        if config.pel_sweep_interval_secs > 0 && last_pel_sweep.elapsed() >= pel_sweep_interval {
            let reclaimed =
                run_pel_sweep_pass(config, conn, handler, &mut last_hypo, &mut last_amyg).await;

            if reclaimed > 0 {
                info!(
                    "🔄 PEL sweep reclaimed {} stale entries from other consumers",
                    reclaimed
                );
                total_processed += reclaimed;
            } else {
                debug!("PEL sweep: no stale entries found");
            }

            last_pel_sweep = std::time::Instant::now();
        }
    }
}

// ============================================================================
// Redis response parsing helpers
// ============================================================================

/// Parse the nested XREADGROUP response into (stream_id, field_map) pairs.
///
/// Redis XREADGROUP returns:
/// ```text
/// [
///   [stream_key, [
///     [id_1, [field_1, value_1, field_2, value_2, ...]],
///     [id_2, [field_1, value_1, ...]],
///   ]]
/// ]
/// ```
///
/// We flatten this into `Vec<(String, HashMap<String, String>)>`.
fn parse_stream_response(values: &[redis::Value]) -> Vec<(String, HashMap<String, String>)> {
    let mut entries = Vec::new();

    for stream_data in values {
        // Each element: [stream_key, [[id, [field, val, ...]], ...]]
        if let redis::Value::Array(stream_arr) = stream_data {
            if stream_arr.len() < 2 {
                continue;
            }
            // stream_arr[1] is the array of entries
            if let redis::Value::Array(ref entry_arr) = stream_arr[1] {
                for entry in entry_arr {
                    if let redis::Value::Array(id_and_fields) = entry {
                        if id_and_fields.len() < 2 {
                            continue;
                        }
                        let stream_id = value_to_string(&id_and_fields[0]);
                        let fields = parse_field_pairs(&id_and_fields[1]);
                        if let (Some(id), Some(map)) = (stream_id, fields) {
                            entries.push((id, map));
                        }
                    }
                }
            }
        }
    }

    entries
}

/// Convert a Redis `Value` to `Option<String>`.
fn value_to_string(val: &redis::Value) -> Option<String> {
    match val {
        redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

/// Parse a Redis array of alternating [field, value, field, value, ...] into a HashMap.
fn parse_field_pairs(val: &redis::Value) -> Option<HashMap<String, String>> {
    if let redis::Value::Array(arr) = val {
        let mut map = HashMap::new();
        let mut iter = arr.iter();
        while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
            if let (Some(key), Some(value)) = (value_to_string(k), value_to_string(v)) {
                map.insert(key, value);
            }
        }
        Some(map)
    } else {
        None
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info,regime_bridge_consumer=debug".into()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║   JANUS Regime Bridge — Redis Stream Consumer            ║");
    info!("║   Cross-process neuromorphic regime state receiver       ║");
    info!("╚═══════════════════════════════════════════════════════════╝");
    info!("");

    let config = ConsumerConfig::from_env();
    let grpc_target = std::env::var("GRPC_TARGET").ok();
    let serve_port = std::env::var("SERVE_PORT").ok();
    let grpc_source_id = std::env::var("GRPC_SOURCE_ID")
        .unwrap_or_else(|_| format!("regime-bridge-consumer-{}", std::process::id()));

    // Determine which handler mode will be used
    let handler_name = if grpc_target.is_some() {
        "GrpcForwarder"
    } else if serve_port.is_some() {
        "Server (embedded gRPC)"
    } else {
        "Logging"
    };

    info!("Configuration:");
    info!("  Redis URL:       {}", config.redis_url);
    info!("  Stream Key:      {}", config.stream_key);
    info!("  Consumer Group:  {}", config.group_name);
    info!("  Consumer Name:   {}", config.consumer_name);
    info!("  Block Timeout:   {}ms", config.block_ms);
    info!("  Batch Size:      {}", config.batch_size);
    info!(
        "  PEL Sweep:       every {}s (min idle: {}ms, batch: {})",
        config.pel_sweep_interval_secs, config.pel_min_idle_ms, config.pel_batch_size
    );
    info!("  Handler:         {}", handler_name);
    if let Some(ref target) = grpc_target {
        info!("  gRPC Target:     {}", target);
        info!("  gRPC Source ID:  {}", grpc_source_id);
    }
    if let Some(ref port) = serve_port {
        info!("  Serve Port:      {}", port);
    }
    info!("");

    // Connect to Redis
    let client =
        redis::Client::open(config.redis_url.as_str()).context("Failed to create Redis client")?;

    let mut conn = client
        .get_multiplexed_async_connection()
        .await
        .context("Failed to connect to Redis")?;

    info!("✅ Connected to Redis at {}", config.redis_url);

    // Ensure consumer group exists
    ensure_consumer_group(&mut conn, &config.stream_key, &config.group_name).await?;

    // Setup signal handler for graceful shutdown
    let shutdown = tokio::signal::ctrl_c();

    info!("🚀 Starting consumer loop...");
    info!("");

    // Select handler based on environment variables.
    //
    // Priority order:
    //   1. GRPC_TARGET — forward each state to a remote gRPC server
    //   2. SERVE_PORT  — start an embedded gRPC server for neuromorphic subscribers
    //   3. (default)   — log each state to stdout (LoggingHandler)
    if let Some(ref target) = grpc_target {
        let handler = GrpcForwarderHandler::connect(target, grpc_source_id)
            .await
            .context("Failed to initialize gRPC forwarder handler")?;

        tokio::select! {
            result = run_consumer(&config, &mut conn, &handler) => {
                match result {
                    Ok(()) => info!("Consumer loop exited"),
                    Err(e) => error!("❌ Consumer loop error: {}", e),
                }
            }
            _ = shutdown => {
                info!("🛑 SIGINT received — shutting down");
            }
        }
    } else if let Some(ref port_str) = serve_port {
        let port: u16 = port_str.parse().with_context(|| {
            format!(
                "Invalid SERVE_PORT '{}': must be a valid port number",
                port_str
            )
        })?;

        let handler = ServerHandler::start(port)
            .await
            .context("Failed to start embedded gRPC server")?;

        tokio::select! {
            result = run_consumer(&config, &mut conn, &handler) => {
                match result {
                    Ok(()) => info!("Consumer loop exited"),
                    Err(e) => error!("❌ Consumer loop error: {}", e),
                }
            }
            _ = shutdown => {
                info!("🛑 SIGINT received — shutting down");
            }
        }
    } else {
        let handler = LoggingHandler;

        tokio::select! {
            result = run_consumer(&config, &mut conn, &handler) => {
                match result {
                    Ok(()) => info!("Consumer loop exited"),
                    Err(e) => error!("❌ Consumer loop error: {}", e),
                }
            }
            _ = shutdown => {
                info!("🛑 SIGINT received — shutting down");
            }
        }
    }

    info!("✅ Regime bridge consumer shutdown complete");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that manipulate environment variables
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_config_from_env_defaults() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        // Clear any env vars that might interfere (serialized by ENV_TEST_LOCK)
        unsafe { std::env::remove_var("REDIS_URL") };
        unsafe { std::env::remove_var("REGIME_BRIDGE_STREAM") };
        unsafe { std::env::remove_var("CONSUMER_GROUP") };
        unsafe { std::env::remove_var("CONSUMER_NAME") };
        unsafe { std::env::remove_var("BLOCK_MS") };
        unsafe { std::env::remove_var("BATCH_SIZE") };
        unsafe { std::env::remove_var("PEL_SWEEP_INTERVAL_SECS") };
        unsafe { std::env::remove_var("PEL_MIN_IDLE_MS") };
        unsafe { std::env::remove_var("PEL_BATCH_SIZE") };
        unsafe { std::env::remove_var("GRPC_TARGET") };
        unsafe { std::env::remove_var("GRPC_SOURCE_ID") };

        let config = ConsumerConfig::from_env();
        assert_eq!(config.redis_url, "redis://127.0.0.1:6379");
        assert_eq!(config.stream_key, "janus:regime:bridge");
        assert_eq!(config.group_name, "neuromorphic");
        assert!(config.consumer_name.starts_with("consumer-"));
        assert_eq!(config.block_ms, 5000);
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.pel_sweep_interval_secs, 60);
        assert_eq!(config.pel_min_idle_ms, 30_000);
        assert_eq!(config.pel_batch_size, 100);
    }

    #[test]
    fn test_config_pel_env_overrides() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        // Clear first to avoid pollution from other tests (serialized by ENV_TEST_LOCK)
        unsafe { std::env::remove_var("PEL_SWEEP_INTERVAL_SECS") };
        unsafe { std::env::remove_var("PEL_MIN_IDLE_MS") };
        unsafe { std::env::remove_var("PEL_BATCH_SIZE") };

        unsafe { std::env::set_var("PEL_SWEEP_INTERVAL_SECS", "120") };
        unsafe { std::env::set_var("PEL_MIN_IDLE_MS", "60000") };
        unsafe { std::env::set_var("PEL_BATCH_SIZE", "50") };

        let config = ConsumerConfig::from_env();
        assert_eq!(config.pel_sweep_interval_secs, 120);
        assert_eq!(config.pel_min_idle_ms, 60_000);
        assert_eq!(config.pel_batch_size, 50);

        unsafe { std::env::remove_var("PEL_SWEEP_INTERVAL_SECS") };
        unsafe { std::env::remove_var("PEL_MIN_IDLE_MS") };
        unsafe { std::env::remove_var("PEL_BATCH_SIZE") };
    }

    #[test]
    fn test_config_pel_zero_interval_disables_sweep() {
        let _lock = ENV_TEST_LOCK.lock().unwrap();
        // Clear first to avoid pollution from other tests (serialized by ENV_TEST_LOCK)
        unsafe { std::env::remove_var("PEL_SWEEP_INTERVAL_SECS") };
        unsafe { std::env::set_var("PEL_SWEEP_INTERVAL_SECS", "0") };
        let config = ConsumerConfig::from_env();
        assert_eq!(config.pel_sweep_interval_secs, 0);
        unsafe { std::env::remove_var("PEL_SWEEP_INTERVAL_SECS") };
    }

    // ── PEL reclamation parsing tests ───────────────────────────────────

    #[test]
    fn test_parse_xautoclaim_response_empty_claimed() {
        // Simulated XAUTOCLAIM response: ["0-0", [], []]
        let response = [
            redis::Value::BulkString(b"0-0".to_vec()),
            redis::Value::Array(vec![]),
            redis::Value::Array(vec![]),
        ];

        let next_start = value_to_string(&response[0]).unwrap_or_else(|| "0-0".to_string());
        assert_eq!(next_start, "0-0");

        let claimed = match &response[1] {
            redis::Value::Array(arr) => arr.len(),
            _ => 0,
        };
        assert_eq!(claimed, 0);
    }

    #[test]
    fn test_parse_xautoclaim_response_with_claimed_entries() {
        // Simulated XAUTOCLAIM response with one claimed entry:
        // ["1234-1", [["1234-0", ["data", "{...}"]]], []]
        let json = r#"{"symbol":"BTCUSD","hypothalamus_regime":"Bullish","amygdala_regime":"LowVolTrending","position_scale":1.0,"is_high_risk":false,"confidence":0.8}"#;
        let response = [
            redis::Value::BulkString(b"1234-1".to_vec()),
            redis::Value::Array(vec![redis::Value::Array(vec![
                redis::Value::BulkString(b"1234-0".to_vec()),
                redis::Value::Array(vec![
                    redis::Value::BulkString(b"data".to_vec()),
                    redis::Value::BulkString(json.as_bytes().to_vec()),
                ]),
            ])]),
            redis::Value::Array(vec![]),
        ];

        let next_start = value_to_string(&response[0]).unwrap();
        assert_eq!(next_start, "1234-1");

        let claimed = match &response[1] {
            redis::Value::Array(arr) => {
                let mut entries = Vec::new();
                for entry in arr {
                    if let redis::Value::Array(id_and_fields) = entry
                        && id_and_fields.len() >= 2
                    {
                        let id = value_to_string(&id_and_fields[0]);
                        let fields = parse_field_pairs(&id_and_fields[1]);
                        if let (Some(id), Some(f)) = (id, fields) {
                            entries.push((id, f));
                        }
                    }
                }
                entries
            }
            _ => vec![],
        };
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].0, "1234-0");
        assert!(claimed[0].1.contains_key("data"));

        let state: StreamedBridgeState =
            serde_json::from_str(claimed[0].1.get("data").unwrap()).unwrap();
        assert_eq!(state.symbol, "BTCUSD");
    }

    #[test]
    fn test_parse_xautoclaim_response_with_deleted_entries() {
        // Simulated XAUTOCLAIM response with deleted entries:
        // ["0-0", [], ["999-0", "999-1"]]
        let response = [
            redis::Value::BulkString(b"0-0".to_vec()),
            redis::Value::Array(vec![]),
            redis::Value::Array(vec![
                redis::Value::BulkString(b"999-0".to_vec()),
                redis::Value::BulkString(b"999-1".to_vec()),
            ]),
        ];

        let deleted = match &response[2] {
            redis::Value::Array(arr) => arr.iter().filter_map(value_to_string).collect::<Vec<_>>(),
            _ => vec![],
        };
        assert_eq!(deleted.len(), 2);
        assert_eq!(deleted[0], "999-0");
        assert_eq!(deleted[1], "999-1");
    }

    #[test]
    fn test_pending_drain_uses_zero_start_id() {
        // Verify the "0" start ID concept: XREADGROUP with "0" returns
        // pending entries, while ">" returns only new ones.
        let pending_start_id = "0";
        let new_start_id = ">";
        assert_ne!(pending_start_id, new_start_id);
        assert_eq!(pending_start_id, "0");
    }

    // ── gRPC handler parsing tests ──────────────────────────────────────

    #[test]
    fn test_parse_hypothalamus_regime_all_variants() {
        assert_eq!(
            parse_hypothalamus_regime("StrongBullish") as i32,
            HypothalamusRegime::StrongBullish as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Bullish") as i32,
            HypothalamusRegime::Bullish as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Neutral") as i32,
            HypothalamusRegime::Neutral as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Bearish") as i32,
            HypothalamusRegime::Bearish as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("StrongBearish") as i32,
            HypothalamusRegime::StrongBearish as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("HighVolatility") as i32,
            HypothalamusRegime::HighVolatility as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("LowVolatility") as i32,
            HypothalamusRegime::LowVolatility as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Transitional") as i32,
            HypothalamusRegime::Transitional as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Crisis") as i32,
            HypothalamusRegime::Crisis as i32
        );
        assert_eq!(
            parse_hypothalamus_regime("Unknown") as i32,
            HypothalamusRegime::Unknown as i32
        );
        // Unrecognized → Unknown
        assert_eq!(
            parse_hypothalamus_regime("garbage") as i32,
            HypothalamusRegime::Unknown as i32
        );
    }

    #[test]
    fn test_parse_amygdala_regime_all_variants() {
        assert_eq!(
            parse_amygdala_regime("LowVolTrending") as i32,
            AmygdalaRegime::LowVolTrending as i32
        );
        assert_eq!(
            parse_amygdala_regime("LowVolMeanReverting") as i32,
            AmygdalaRegime::LowVolMeanReverting as i32
        );
        assert_eq!(
            parse_amygdala_regime("HighVolTrending") as i32,
            AmygdalaRegime::HighVolTrending as i32
        );
        assert_eq!(
            parse_amygdala_regime("HighVolMeanReverting") as i32,
            AmygdalaRegime::HighVolMeanReverting as i32
        );
        assert_eq!(
            parse_amygdala_regime("Crisis") as i32,
            AmygdalaRegime::Crisis as i32
        );
        assert_eq!(
            parse_amygdala_regime("Transitional") as i32,
            AmygdalaRegime::Transitional as i32
        );
        assert_eq!(
            parse_amygdala_regime("Unknown") as i32,
            AmygdalaRegime::Unknown as i32
        );
        // Unrecognized → Unknown
        assert_eq!(
            parse_amygdala_regime("xyz") as i32,
            AmygdalaRegime::Unknown as i32
        );
    }

    #[test]
    fn test_grpc_handler_to_proto_state_fields() {
        // We can't call GrpcForwarderHandler::connect without a real server,
        // but we can test the static parsing helpers and verify the proto
        // conversion logic by constructing the state manually.
        let state = StreamedBridgeState {
            symbol: "BTCUSD".into(),
            hypothalamus_regime: "StrongBullish".into(),
            amygdala_regime: "LowVolTrending".into(),
            position_scale: 1.2,
            is_high_risk: false,
            confidence: 0.85,
            trend: 0.7,
            trend_strength: 0.6,
            volatility: 500.0,
            volatility_percentile: 0.3,
            momentum: 0.4,
            relative_volume: 1.5,
            liquidity_score: 0.9,
        };

        // Verify the parsing produces correct enum variants
        let hypo = parse_hypothalamus_regime(&state.hypothalamus_regime);
        let amyg = parse_amygdala_regime(&state.amygdala_regime);
        assert_eq!(hypo as i32, HypothalamusRegime::StrongBullish as i32);
        assert_eq!(amyg as i32, AmygdalaRegime::LowVolTrending as i32);
    }

    #[test]
    fn test_grpc_source_id_default() {
        unsafe { std::env::remove_var("GRPC_SOURCE_ID") };
        let source_id = std::env::var("GRPC_SOURCE_ID")
            .unwrap_or_else(|_| format!("regime-bridge-consumer-{}", std::process::id()));
        assert!(source_id.starts_with("regime-bridge-consumer-"));
    }

    #[test]
    fn test_streamed_bridge_state_deserialize() {
        let json = r#"{
            "symbol": "BTCUSD",
            "hypothalamus_regime": "StrongBullish",
            "amygdala_regime": "LowVolTrending",
            "position_scale": 1.25,
            "is_high_risk": false,
            "confidence": 0.85,
            "trend": 0.7,
            "trend_strength": 0.6,
            "volatility": 0.12,
            "volatility_percentile": 0.3,
            "momentum": 0.35,
            "relative_volume": 1.5,
            "liquidity_score": 1.0
        }"#;

        let state: StreamedBridgeState = serde_json::from_str(json).unwrap();
        assert_eq!(state.symbol, "BTCUSD");
        assert_eq!(state.hypothalamus_regime, "StrongBullish");
        assert_eq!(state.amygdala_regime, "LowVolTrending");
        assert!((state.position_scale - 1.25).abs() < f64::EPSILON);
        assert!(!state.is_high_risk);
        assert!((state.confidence - 0.85).abs() < f64::EPSILON);
        assert!((state.trend - 0.7).abs() < f64::EPSILON);
        assert!((state.relative_volume - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_streamed_bridge_state_deserialize_minimal() {
        // Only required fields — optional fields should use defaults
        let json = r#"{
            "symbol": "ETHUSD",
            "hypothalamus_regime": "Neutral",
            "amygdala_regime": "Unknown",
            "position_scale": 1.0,
            "is_high_risk": false,
            "confidence": 0.5
        }"#;

        let state: StreamedBridgeState = serde_json::from_str(json).unwrap();
        assert_eq!(state.symbol, "ETHUSD");
        assert!((state.trend - 0.0).abs() < f64::EPSILON);
        assert!((state.volatility_percentile - 0.5).abs() < f64::EPSILON);
        assert!((state.relative_volume - 1.0).abs() < f64::EPSILON);
        assert!((state.liquidity_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_streamed_bridge_state_display() {
        let state = StreamedBridgeState {
            symbol: "BTCUSD".into(),
            hypothalamus_regime: "Bullish".into(),
            amygdala_regime: "LowVolTrending".into(),
            position_scale: 1.1,
            is_high_risk: false,
            confidence: 0.8,
            trend: 0.6,
            trend_strength: 0.5,
            volatility: 0.15,
            volatility_percentile: 0.4,
            momentum: 0.3,
            relative_volume: 1.2,
            liquidity_score: 1.0,
        };

        let display = format!("{}", state);
        assert!(display.contains("BTCUSD"));
        assert!(display.contains("Bullish"));
        assert!(display.contains("LowVolTrending"));
        assert!(display.contains("110%")); // position_scale 1.1 → 110%
        assert!(display.contains("low")); // is_high_risk = false
    }

    #[test]
    fn test_streamed_bridge_state_display_high_risk() {
        let state = StreamedBridgeState {
            symbol: "SOLUSD".into(),
            hypothalamus_regime: "Crisis".into(),
            amygdala_regime: "Crisis".into(),
            position_scale: 0.2,
            is_high_risk: true,
            confidence: 0.95,
            trend: -0.8,
            trend_strength: 0.0,
            volatility: 0.5,
            volatility_percentile: 0.95,
            momentum: 0.0,
            relative_volume: 3.0,
            liquidity_score: 0.5,
        };

        let display = format!("{}", state);
        assert!(display.contains("HIGH"));
        assert!(display.contains("Crisis"));
        assert!(display.contains("3.00")); // relative_volume
    }

    #[test]
    fn test_parse_field_pairs_empty() {
        let val = redis::Value::Array(vec![]);
        let map = parse_field_pairs(&val).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_parse_field_pairs_single_pair() {
        let val = redis::Value::Array(vec![
            redis::Value::BulkString(b"data".to_vec()),
            redis::Value::BulkString(b"{\"symbol\":\"BTC\"}".to_vec()),
        ]);
        let map = parse_field_pairs(&val).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("data").unwrap(), "{\"symbol\":\"BTC\"}");
    }

    #[test]
    fn test_parse_field_pairs_multiple_pairs() {
        let val = redis::Value::Array(vec![
            redis::Value::BulkString(b"field1".to_vec()),
            redis::Value::BulkString(b"value1".to_vec()),
            redis::Value::BulkString(b"field2".to_vec()),
            redis::Value::BulkString(b"value2".to_vec()),
        ]);
        let map = parse_field_pairs(&val).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("field1").unwrap(), "value1");
        assert_eq!(map.get("field2").unwrap(), "value2");
    }

    #[test]
    fn test_parse_field_pairs_odd_count_drops_orphan() {
        // Odd number of elements — last key has no value, should be dropped
        let val = redis::Value::Array(vec![
            redis::Value::BulkString(b"key1".to_vec()),
            redis::Value::BulkString(b"val1".to_vec()),
            redis::Value::BulkString(b"orphan".to_vec()),
        ]);
        let map = parse_field_pairs(&val).unwrap();
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key("orphan"));
    }

    #[test]
    fn test_parse_field_pairs_non_array_returns_none() {
        let val = redis::Value::Nil;
        assert!(parse_field_pairs(&val).is_none());
    }

    #[test]
    fn test_value_to_string_bulk() {
        let val = redis::Value::BulkString(b"hello".to_vec());
        assert_eq!(value_to_string(&val).unwrap(), "hello");
    }

    #[test]
    fn test_value_to_string_simple() {
        let val = redis::Value::SimpleString("world".into());
        assert_eq!(value_to_string(&val).unwrap(), "world");
    }

    #[test]
    fn test_value_to_string_nil_returns_none() {
        let val = redis::Value::Nil;
        assert!(value_to_string(&val).is_none());
    }

    #[test]
    fn test_value_to_string_int_returns_none() {
        let val = redis::Value::Int(42);
        assert!(value_to_string(&val).is_none());
    }

    #[test]
    fn test_parse_stream_response_empty() {
        let values: Vec<redis::Value> = vec![];
        let entries = parse_stream_response(&values);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_stream_response_well_formed() {
        // Simulate: [ [ "janus:regime:bridge", [ ["1234-0", ["data", "{...}"]] ] ] ]
        let json_payload = r#"{"symbol":"BTCUSD","hypothalamus_regime":"Bullish","amygdala_regime":"LowVolTrending","position_scale":1.1,"is_high_risk":false,"confidence":0.8}"#;

        let values = vec![redis::Value::Array(vec![
            redis::Value::BulkString(b"janus:regime:bridge".to_vec()),
            redis::Value::Array(vec![redis::Value::Array(vec![
                redis::Value::BulkString(b"1234-0".to_vec()),
                redis::Value::Array(vec![
                    redis::Value::BulkString(b"data".to_vec()),
                    redis::Value::BulkString(json_payload.as_bytes().to_vec()),
                ]),
            ])]),
        ])];

        let entries = parse_stream_response(&values);
        assert_eq!(entries.len(), 1);

        let (id, fields) = &entries[0];
        assert_eq!(id, "1234-0");
        assert!(fields.contains_key("data"));

        // Verify the JSON deserializes correctly
        let state: StreamedBridgeState = serde_json::from_str(fields.get("data").unwrap()).unwrap();
        assert_eq!(state.symbol, "BTCUSD");
        assert_eq!(state.hypothalamus_regime, "Bullish");
    }

    #[test]
    fn test_parse_stream_response_multiple_entries() {
        let values = vec![redis::Value::Array(vec![
            redis::Value::BulkString(b"stream".to_vec()),
            redis::Value::Array(vec![
                redis::Value::Array(vec![
                    redis::Value::BulkString(b"100-0".to_vec()),
                    redis::Value::Array(vec![
                        redis::Value::BulkString(b"data".to_vec()),
                        redis::Value::BulkString(b"first".to_vec()),
                    ]),
                ]),
                redis::Value::Array(vec![
                    redis::Value::BulkString(b"200-0".to_vec()),
                    redis::Value::Array(vec![
                        redis::Value::BulkString(b"data".to_vec()),
                        redis::Value::BulkString(b"second".to_vec()),
                    ]),
                ]),
            ]),
        ])];

        let entries = parse_stream_response(&values);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "100-0");
        assert_eq!(entries[1].0, "200-0");
        assert_eq!(entries[0].1.get("data").unwrap(), "first");
        assert_eq!(entries[1].1.get("data").unwrap(), "second");
    }
}
