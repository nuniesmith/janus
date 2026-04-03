//! # gRPC API for JANUS Service
//!
//! Implements the gRPC server for trading signal generation using tonic.
//!
//! ## Services
//!
//! - `GenerateSignal`: Generate a single trading signal
//! - `GenerateSignalBatch`: Generate multiple signals in batch
//! - `GetHealth`: Service health check
//! - `GetMetrics`: Retrieve service metrics
//! - `LoadModel`: Load an ML model
//! - `StreamSignals`: Real-time signal streaming (future)

use crate::indicators::IndicatorAnalysis;
use crate::signal::{SignalGenerator, SignalType, Timeframe, TradingSignal};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status, transport::Server};
use tracing::{debug, error, info, warn};

// Generated protobuf code
pub mod proto {
    tonic::include_proto!("janus.v1");
}

use proto::janus_service_server::{JanusService, JanusServiceServer};
use proto::*;

/// gRPC service implementation
pub struct JanusGrpcService {
    generator: Arc<RwLock<SignalGenerator>>,
    start_time: std::time::Instant,
}

impl JanusGrpcService {
    /// Create a new gRPC service instance
    pub fn new(generator: SignalGenerator) -> Self {
        Self {
            generator: Arc::new(RwLock::new(generator)),
            start_time: std::time::Instant::now(),
        }
    }

    /// Convert internal SignalType to proto SignalType
    fn convert_signal_type(signal_type: &SignalType) -> i32 {
        match signal_type {
            SignalType::StrongBuy => proto::SignalType::StrongBuy as i32,
            SignalType::Buy => proto::SignalType::Buy as i32,
            SignalType::Hold => proto::SignalType::Hold as i32,
            SignalType::Sell => proto::SignalType::Sell as i32,
            SignalType::StrongSell => proto::SignalType::StrongSell as i32,
        }
    }

    /// Convert proto Timeframe to internal Timeframe
    fn convert_timeframe(timeframe: i32) -> Result<Timeframe, Status> {
        use proto::Timeframe as ProtoTf;
        match timeframe {
            x if x == ProtoTf::Timeframe1m as i32 => Ok(Timeframe::M1),
            x if x == ProtoTf::Timeframe5m as i32 => Ok(Timeframe::M5),
            x if x == ProtoTf::Timeframe15m as i32 => Ok(Timeframe::M15),
            x if x == ProtoTf::Timeframe1h as i32 => Ok(Timeframe::H1),
            x if x == ProtoTf::Timeframe4h as i32 => Ok(Timeframe::H4),
            x if x == ProtoTf::Timeframe1d as i32 => Ok(Timeframe::D1),
            _ => Err(Status::invalid_argument("Invalid timeframe")),
        }
    }

    /// Convert proto IndicatorAnalysis to internal IndicatorAnalysis
    fn convert_indicator_analysis(proto_analysis: &proto::IndicatorAnalysis) -> IndicatorAnalysis {
        IndicatorAnalysis {
            ema_fast: Some(proto_analysis.ema_fast),
            ema_slow: Some(proto_analysis.ema_slow),
            ema_cross: proto_analysis.ema_cross,
            rsi: Some(proto_analysis.rsi),
            rsi_signal: proto_analysis.rsi_signal,
            macd_line: Some(proto_analysis.macd_line),
            macd_signal: Some(proto_analysis.macd_signal),
            macd_histogram: Some(proto_analysis.macd_histogram),
            macd_cross: proto_analysis.macd_cross,
            bb_upper: Some(proto_analysis.bb_upper),
            bb_middle: Some(proto_analysis.bb_middle),
            bb_lower: Some(proto_analysis.bb_lower),
            bb_position: proto_analysis.bb_position,
            atr: Some(proto_analysis.atr),
            trend_strength: proto_analysis.trend_strength,
            volatility: proto_analysis.volatility,
        }
    }

    /// Convert internal TradingSignal to proto TradingSignal
    fn convert_trading_signal(signal: &TradingSignal) -> proto::TradingSignal {
        proto::TradingSignal {
            signal_id: signal.signal_id.clone(),
            signal_type: Self::convert_signal_type(&signal.signal_type),
            symbol: signal.symbol.clone(),
            timeframe: match signal.timeframe {
                Timeframe::M1 => proto::Timeframe::Timeframe1m as i32,
                Timeframe::M5 => proto::Timeframe::Timeframe5m as i32,
                Timeframe::M15 => proto::Timeframe::Timeframe15m as i32,
                Timeframe::H1 => proto::Timeframe::Timeframe1h as i32,
                Timeframe::H4 => proto::Timeframe::Timeframe4h as i32,
                Timeframe::D1 => proto::Timeframe::Timeframe1d as i32,
            },
            confidence: signal.confidence,
            strength: signal.strength,
            source: Some(SignalSource {
                source: Some(match &signal.source {
                    crate::signal::SignalSource::TechnicalIndicator { name } => {
                        signal_source::Source::TechnicalIndicator(TechnicalIndicatorSource {
                            name: name.clone(),
                        })
                    }
                    crate::signal::SignalSource::MlModel { model_id, version }
                    | crate::signal::SignalSource::ModelInference {
                        model_name: model_id,
                        version,
                    } => signal_source::Source::MlModel(MlModelSource {
                        model_name: model_id.clone(),
                        version: version.clone(),
                    }),
                    crate::signal::SignalSource::Strategy { name } => {
                        signal_source::Source::Strategy(StrategySource { name: name.clone() })
                    }
                    crate::signal::SignalSource::Manual { user_id } => {
                        signal_source::Source::Manual(ManualSource {
                            user_id: user_id.clone(),
                        })
                    }
                }),
            }),
            timestamp: signal.timestamp.timestamp(),
            metadata: signal.metadata.clone(),
            stop_loss: signal.stop_loss.unwrap_or(0.0),
            take_profit: signal.take_profit.unwrap_or(0.0),
            risk_reward_ratio: signal.risk_reward_ratio().unwrap_or(0.0),
            entry_price: 0.0,
            position_size: 0.0,
            reasoning: String::new(),
            indicator_analysis: None,
        }
    }
}

#[tonic::async_trait]
impl JanusService for JanusGrpcService {
    type StreamSignalsStream = tokio_stream::wrappers::ReceiverStream<Result<SignalUpdate, Status>>;

    async fn generate_signal(
        &self,
        request: Request<GenerateSignalRequest>,
    ) -> Result<Response<GenerateSignalResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();

        debug!(
            "Generating signal for {} on timeframe {}",
            req.symbol, req.timeframe
        );

        // Validate request
        if req.symbol.is_empty() {
            return Err(Status::invalid_argument("Symbol is required"));
        }

        let analysis_proto = req
            .analysis
            .ok_or_else(|| Status::invalid_argument("Indicator analysis is required"))?;

        // Convert types
        let timeframe = Self::convert_timeframe(req.timeframe)?;
        let analysis = Self::convert_indicator_analysis(&analysis_proto);

        // Generate signal
        let generator = self.generator.read().await;
        let result = generator
            .generate_from_analysis(req.symbol, timeframe, &analysis, req.current_price)
            .await;

        let processing_time_us = start.elapsed().as_micros() as u64;

        match result {
            Ok(signal_opt) => {
                let (signal, filtered) = match signal_opt {
                    Some(sig) => (Some(Self::convert_trading_signal(&sig)), false),
                    None => (None, true),
                };

                Ok(Response::new(GenerateSignalResponse {
                    signal,
                    filtered,
                    processing_time_us,
                    warnings: vec![],
                }))
            }
            Err(e) => {
                error!("Error generating signal: {}", e);
                Err(Status::internal(format!(
                    "Failed to generate signal: {}",
                    e
                )))
            }
        }
    }

    async fn generate_signal_batch(
        &self,
        request: Request<GenerateSignalBatchRequest>,
    ) -> Result<Response<GenerateSignalBatchResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();

        debug!("Processing batch of {} signal requests", req.requests.len());

        let mut responses = Vec::new();
        let mut stats = BatchStatistics {
            total_requests: req.requests.len() as u32,
            successful: 0,
            filtered: 0,
            failed: 0,
            success_rate: 0.0,
        };

        for signal_req in req.requests {
            let req_inner = Request::new(signal_req);
            match self.generate_signal(req_inner).await {
                Ok(resp) => {
                    let response = resp.into_inner();
                    if response.signal.is_some() {
                        stats.successful += 1;
                    } else if response.filtered {
                        stats.filtered += 1;
                    }
                    responses.push(response);
                }
                Err(e) => {
                    error!("Batch signal generation error: {}", e);
                    stats.failed += 1;
                    responses.push(GenerateSignalResponse {
                        signal: None,
                        filtered: false,
                        processing_time_us: 0,
                        warnings: vec![e.message().to_string()],
                    });
                }
            }
        }

        stats.success_rate = if stats.total_requests > 0 {
            stats.successful as f64 / stats.total_requests as f64
        } else {
            0.0
        };

        let total_processing_time_us = start.elapsed().as_micros() as u64;

        Ok(Response::new(GenerateSignalBatchResponse {
            responses,
            total_processing_time_us,
            statistics: Some(stats),
        }))
    }

    async fn get_health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        debug!("Health check requested");

        let generator = self.generator.read().await;
        let metrics = generator.metrics();

        let signal_metrics = SignalMetrics {
            total_generated: metrics.total_generated(),
            total_filtered: metrics.total_filtered(),
            filter_rate: metrics.filter_rate(),
        };

        let ml_metrics = generator.ml_metrics().await.map(|ml_met| MlMetrics {
            total_inferences: ml_met.total_inferences,
            avg_latency_us: ml_met.avg_latency_us(),
            p99_latency_us: ml_met.p99_latency_us(),
            avg_confidence: ml_met.avg_confidence,
        });

        Ok(Response::new(HealthResponse {
            service: crate::SERVICE_NAME.to_string(),
            version: crate::VERSION.to_string(),
            status: proto::HealthStatus::Healthy as i32,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            signal_metrics: Some(signal_metrics),
            ml_metrics,
        }))
    }

    async fn get_metrics(
        &self,
        _request: Request<MetricsRequest>,
    ) -> Result<Response<MetricsResponse>, Status> {
        debug!("Metrics requested");

        let generator = self.generator.read().await;
        let metrics = generator.metrics();

        let signal_metrics = SignalMetrics {
            total_generated: metrics.total_generated(),
            total_filtered: metrics.total_filtered(),
            filter_rate: metrics.filter_rate(),
        };

        let ml_metrics = generator.ml_metrics().await.map(|ml_met| MlMetrics {
            total_inferences: ml_met.total_inferences,
            avg_latency_us: ml_met.avg_latency_us(),
            p99_latency_us: ml_met.p99_latency_us(),
            avg_confidence: ml_met.avg_confidence,
        });

        Ok(Response::new(MetricsResponse {
            signal_metrics: Some(signal_metrics),
            ml_metrics,
            strategy_metrics: std::collections::HashMap::new(),
            indicator_metrics: Some(IndicatorMetrics {
                total_updates: 0,
                avg_update_time_us: 0,
            }),
        }))
    }

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();

        info!("Loading model: {} from {}", req.model_name, req.model_path);

        let generator = self.generator.read().await;

        match generator.load_model(&req.model_name, &req.model_path).await {
            Ok(_) => {
                info!("Model loaded successfully: {}", req.model_name);
                Ok(Response::new(LoadModelResponse {
                    success: true,
                    error: String::new(),
                    model_info: Some(ModelInfo {
                        name: req.model_name.clone(),
                        version: "v1".to_string(),
                        size_bytes: std::fs::metadata(&req.model_path)
                            .map(|m| m.len())
                            .unwrap_or(0),
                        loaded_at: chrono::Utc::now().timestamp(),
                    }),
                }))
            }
            Err(e) => {
                error!("Failed to load model {}: {}", req.model_name, e);
                Ok(Response::new(LoadModelResponse {
                    success: false,
                    error: e.to_string(),
                    model_info: None,
                }))
            }
        }
    }

    async fn stream_signals(
        &self,
        _request: Request<StreamSignalsRequest>,
    ) -> Result<Response<Self::StreamSignalsStream>, Status> {
        warn!("Stream signals not yet implemented");
        Err(Status::unimplemented("Signal streaming not yet available"))
    }
}

/// gRPC server wrapper
pub struct GrpcServer {
    port: u16,
    generator: SignalGenerator,
}

impl GrpcServer {
    /// Create a new gRPC server
    pub fn new(port: u16, generator: SignalGenerator) -> Self {
        info!("Initializing gRPC server on port {}", port);
        Self { port, generator }
    }

    /// Start the gRPC server
    pub async fn start(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port).parse()?;
        let service = JanusGrpcService::new(self.generator);

        info!("Starting gRPC server on {}", addr);

        Server::builder()
            .add_service(JanusServiceServer::new(service))
            .serve(addr)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::SignalGeneratorConfig;

    #[test]
    fn test_signal_type_conversion() {
        assert_eq!(
            JanusGrpcService::convert_signal_type(&SignalType::StrongBuy),
            proto::SignalType::StrongBuy as i32
        );
        assert_eq!(
            JanusGrpcService::convert_signal_type(&SignalType::Buy),
            proto::SignalType::Buy as i32
        );
        assert_eq!(
            JanusGrpcService::convert_signal_type(&SignalType::Hold),
            proto::SignalType::Hold as i32
        );
    }

    #[test]
    fn test_timeframe_conversion() {
        // Valid timeframes: 1 (1m), 5 (5m), 15 (15m), 60 (1h), 240 (4h), 1440 (1d)
        assert!(JanusGrpcService::convert_timeframe(1).is_ok());
        assert!(JanusGrpcService::convert_timeframe(5).is_ok());
        assert!(JanusGrpcService::convert_timeframe(99).is_err());
    }

    #[tokio::test]
    async fn test_grpc_service_creation() {
        let config = SignalGeneratorConfig::default();
        let generator = SignalGenerator::new(config);
        let _service = JanusGrpcService::new(generator);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = SignalGeneratorConfig::default();
        let generator = SignalGenerator::new(config);
        let service = JanusGrpcService::new(generator);

        let request = Request::new(HealthRequest {});
        let response = service.get_health(request).await;

        assert!(response.is_ok());
        let health = response.unwrap().into_inner();
        assert_eq!(health.service, crate::SERVICE_NAME);
        assert_eq!(health.version, crate::VERSION);
    }

    #[tokio::test]
    async fn test_get_metrics() {
        let config = SignalGeneratorConfig::default();
        let generator = SignalGenerator::new(config);
        let service = JanusGrpcService::new(generator);

        let request = Request::new(MetricsRequest {});
        let response = service.get_metrics(request).await;

        assert!(response.is_ok());
        let metrics = response.unwrap().into_inner();
        assert!(metrics.signal_metrics.is_some());
    }
}
