//! # PositionFeedback gRPC Service
//!
//! Implements [`PositionFeedbackService`] for the Janus forward service,
//! providing real-time trade guidance from live position state and recording
//! completed trade outcomes for downstream memory/learning.
//!
//! ## RPC methods
//!
//! | Method | Transport | Description |
//! |--------|-----------|-------------|
//! | `StreamPositionUpdates` | client-streaming | Consume a stream; return guidance based on the last message |
//! | `ReportPosition` | unary | Single position update → guidance |
//! | `RecordTradeOutcome` | unary | Log a completed trade; acknowledge to caller |
//!
//! ## Guidance thresholds (MVP)
//!
//! | Condition | Action |
//! |-----------|--------|
//! | `unrealized_pnl > 0.5 % × entry_price` | `TAKE_PROFIT` |
//! | `unrealized_pnl < −0.3 % × entry_price` | `TRAIL_STOP` |
//! | otherwise | `HOLD` |
//!
//! Confidence is fixed at `0.75`; reason is `"brain guidance"`.
//! Actual persistence of trade outcomes is handled by the Python
//! `MemoryRecorder` — Rust just logs and acknowledges here.

use fks_proto::feedback::{
    GuidanceAction, PositionFeedback, TradeGuidance, TradeOutcome, TradeOutcomeAck,
    position_feedback_service_server::{PositionFeedbackService, PositionFeedbackServiceServer},
};
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, info, warn};

// ── Service Implementation ─────────────────────────────────────────────────

/// Stateless gRPC service that provides real-time trade guidance.
///
/// All guidance is computed on-the-fly from the position data — no external
/// state or locks are needed for the current MVP implementation.  Wire the
/// brain pipeline here in a future iteration to make guidance model-driven.
#[derive(Debug, Default)]
pub struct FeedbackGrpcService;

#[tonic::async_trait]
impl PositionFeedbackService for FeedbackGrpcService {
    /// Consume a client-streaming sequence of position updates and return
    /// a single [`TradeGuidance`] based on the **last** message received.
    ///
    /// If the stream is empty the call is rejected with `INVALID_ARGUMENT`.
    async fn stream_position_updates(
        &self,
        request: Request<Streaming<PositionFeedback>>,
    ) -> Result<Response<TradeGuidance>, Status> {
        let mut stream = request.into_inner();
        let mut last_feedback: Option<PositionFeedback> = None;
        let mut count: u32 = 0;

        while let Some(feedback) = stream.message().await? {
            debug!(
                signal_id = %feedback.signal_id,
                symbol    = %feedback.symbol,
                pnl       = feedback.unrealized_pnl,
                count,
                "StreamPositionUpdates: received update",
            );
            count += 1;
            last_feedback = Some(feedback);
        }

        match last_feedback {
            Some(feedback) => {
                debug!(
                    signal_id = %feedback.signal_id,
                    total_messages = count,
                    "StreamPositionUpdates: stream ended — computing guidance",
                );
                Ok(Response::new(compute_guidance(&feedback)))
            }
            None => {
                warn!("StreamPositionUpdates received an empty stream — no guidance produced");
                Err(Status::invalid_argument(
                    "Empty stream: no PositionFeedback messages were received",
                ))
            }
        }
    }

    /// Handle a single position update and return guidance immediately.
    async fn report_position(
        &self,
        request: Request<PositionFeedback>,
    ) -> Result<Response<TradeGuidance>, Status> {
        let feedback = request.into_inner();
        debug!(
            signal_id      = %feedback.signal_id,
            symbol         = %feedback.symbol,
            unrealized_pnl = feedback.unrealized_pnl,
            entry_price    = feedback.entry_price,
            "ReportPosition received",
        );
        Ok(Response::new(compute_guidance(&feedback)))
    }

    /// Log a completed trade outcome and acknowledge to the caller.
    ///
    /// Actual persistence is delegated to the Python `MemoryRecorder`.
    /// Rust records the outcome at `INFO` level and returns a synthetic
    /// `memory_id` so the caller can correlate the acknowledgement.
    async fn record_trade_outcome(
        &self,
        request: Request<TradeOutcome>,
    ) -> Result<Response<TradeOutcomeAck>, Status> {
        let outcome = request.into_inner();

        info!(
            signal_id   = %outcome.signal_id,
            symbol      = %outcome.symbol,
            account_id  = %outcome.account_id,
            pnl         = outcome.pnl,
            pnl_pct     = outcome.pnl_percent,
            result      = outcome.result,
            exit_reason = %outcome.exit_reason,
            strategy    = %outcome.strategy,
            regime      = %outcome.regime,
            duration_ms = outcome.duration_ms,
            "RecordTradeOutcome: trade lifecycle captured",
        );

        // Build a deterministic-ish memory ID from signal_id + wall-clock nanos.
        // The Python MemoryRecorder will assign its own durable ID; this is
        // purely so the gRPC response carries a non-empty `memory_id`.
        let ts_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let memory_id = format!("mem-{}-{:x}", outcome.signal_id, ts_nanos);

        Ok(Response::new(TradeOutcomeAck {
            signal_id: outcome.signal_id,
            stored: true,
            memory_id,
        }))
    }
}

// ── Guidance Logic ─────────────────────────────────────────────────────────

/// Compute a [`TradeGuidance`] response from live position state.
///
/// Decision thresholds (MVP):
/// - `unrealized_pnl > 0.5 % × entry_price` → [`GuidanceAction::TakeProfit`]
/// - `unrealized_pnl < −0.3 % × entry_price` → [`GuidanceAction::TrailStop`]
/// - otherwise → [`GuidanceAction::Hold`]
///
/// Confidence is fixed at `0.75` until the brain pipeline is wired in.
fn compute_guidance(feedback: &PositionFeedback) -> TradeGuidance {
    let entry = feedback.entry_price;
    let pnl = feedback.unrealized_pnl;

    let action = if pnl > 0.005 * entry {
        GuidanceAction::TakeProfit
    } else if pnl < -0.003 * entry {
        GuidanceAction::TrailStop
    } else {
        GuidanceAction::Hold
    };

    debug!(
        signal_id = %feedback.signal_id,
        entry,
        pnl,
        action = ?action,
        "compute_guidance: action selected",
    );

    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    TradeGuidance {
        signal_id: feedback.signal_id.clone(),
        action: action as i32,
        confidence: 0.75,
        reason: "brain guidance".to_string(),
        timestamp_ms,
        ..Default::default()
    }
}

// ── Server Builder ─────────────────────────────────────────────────────────

/// Convenience wrapper that holds the port for the feedback gRPC server.
///
/// # Example
///
/// ```rust,ignore
/// let server = FeedbackGrpcServer::new(50052);
/// tonic::transport::Server::builder()
///     .add_service(server.into_service())
///     .serve("0.0.0.0:50052".parse()?)
///     .await?;
/// ```
pub struct FeedbackGrpcServer {
    /// Port the tonic server will bind to.
    pub port: u16,
}

impl FeedbackGrpcServer {
    /// Create a new server builder for the given *port*.
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    /// Consume this builder and return a configured tonic service wrapper
    /// ready to be mounted on a [`tonic::transport::Server`].
    pub fn into_service(self) -> PositionFeedbackServiceServer<FeedbackGrpcService> {
        PositionFeedbackServiceServer::new(FeedbackGrpcService)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feedback(signal_id: &str, entry: f64, pnl: f64) -> PositionFeedback {
        PositionFeedback {
            signal_id: signal_id.to_string(),
            symbol: "BTCUSD".to_string(),
            entry_price: entry,
            unrealized_pnl: pnl,
            ..Default::default()
        }
    }

    #[test]
    fn test_guidance_hold() {
        let fb = make_feedback("sig-1", 100.0, 0.0);
        let g = compute_guidance(&fb);
        assert_eq!(g.action, GuidanceAction::Hold as i32);
        assert_eq!(g.confidence, 0.75);
        assert_eq!(g.reason, "brain guidance");
        assert_eq!(g.signal_id, "sig-1");
    }

    #[test]
    fn test_guidance_take_profit() {
        // pnl = 1.0 > 0.005 * 100.0 = 0.5
        let fb = make_feedback("sig-2", 100.0, 1.0);
        let g = compute_guidance(&fb);
        assert_eq!(g.action, GuidanceAction::TakeProfit as i32);
    }

    #[test]
    fn test_guidance_trail_stop() {
        // pnl = -0.5 < -0.003 * 100.0 = -0.3
        let fb = make_feedback("sig-3", 100.0, -0.5);
        let g = compute_guidance(&fb);
        assert_eq!(g.action, GuidanceAction::TrailStop as i32);
    }

    #[test]
    fn test_guidance_boundary_take_profit() {
        // Exactly at the threshold — should NOT trigger (strict >)
        let fb = make_feedback("sig-4", 100.0, 0.5);
        let g = compute_guidance(&fb);
        assert_eq!(g.action, GuidanceAction::Hold as i32);
    }

    #[test]
    fn test_feedback_grpc_server_new() {
        let server = FeedbackGrpcServer::new(50052);
        assert_eq!(server.port, 50052);
    }

    #[test]
    fn test_feedback_grpc_server_into_service() {
        // Smoke-test that into_service() doesn't panic.
        let server = FeedbackGrpcServer::new(50052);
        let _svc = server.into_service();
    }

    #[tokio::test]
    async fn test_report_position_hold() {
        let svc = FeedbackGrpcService;
        let fb = make_feedback("sig-5", 50000.0, 10.0); // pnl < 0.005 * 50_000 = 250
        let req = Request::new(fb);
        let resp = svc.report_position(req).await.unwrap();
        assert_eq!(resp.into_inner().action, GuidanceAction::Hold as i32);
    }

    #[tokio::test]
    async fn test_report_position_take_profit() {
        let svc = FeedbackGrpcService;
        let fb = make_feedback("sig-6", 50000.0, 300.0); // pnl > 0.005 * 50_000 = 250
        let req = Request::new(fb);
        let resp = svc.report_position(req).await.unwrap();
        assert_eq!(resp.into_inner().action, GuidanceAction::TakeProfit as i32);
    }

    #[tokio::test]
    async fn test_record_trade_outcome_ack() {
        let svc = FeedbackGrpcService;
        let outcome = TradeOutcome {
            signal_id: "sig-7".to_string(),
            symbol: "ETHUSD".to_string(),
            pnl: 42.5,
            exit_reason: "take_profit".to_string(),
            ..Default::default()
        };
        let req = Request::new(outcome);
        let resp = svc.record_trade_outcome(req).await.unwrap();
        let ack = resp.into_inner();
        assert_eq!(ack.signal_id, "sig-7");
        assert!(ack.stored);
        assert!(ack.memory_id.starts_with("mem-sig-7-"));
    }
}
