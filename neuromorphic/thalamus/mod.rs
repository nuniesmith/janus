//! Thalamus: Attention & Multimodal Fusion
//!
//! The thalamus serves as the central relay station for sensory information,
//! routing data from various sources to appropriate processing regions.
//!
//! ## Components
//!
//! - **Attention**: Selective attention mechanisms for focusing on relevant data
//! - **Fusion**: Multimodal data fusion combining different data types
//! - **Gating**: Information gating based on relevance and priority
//! - **Routing**: Signal routing to appropriate processing pathways
//! - **Sources**: External data sources (news, weather, celestial)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                       THALAMUS                               │
//! │                  (Sensory Relay Hub)                         │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │   Market     │  │    News/     │  │  Celestial   │     │
//! │  │    Data      │  │  Sentiment   │  │    Data      │     │
//! │  │  (Sources)   │  │  (Sources)   │  │  (Sources)   │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            ▼                                │
//! │                   ┌────────────────┐                        │
//! │                   │    Routing     │                        │
//! │                   │   (Priority)   │                        │
//! │                   └────────┬───────┘                        │
//! │                            │                                │
//! │         ┌──────────────────┼──────────────────┐            │
//! │         ▼                  ▼                  ▼            │
//! │   ┌──────────┐      ┌──────────┐      ┌──────────┐        │
//! │   │Attention │      │  Gating  │      │  Fusion  │        │
//! │   └────┬─────┘      └────┬─────┘      └────┬─────┘        │
//! │        │                  │                  │             │
//! │        └──────────────────┼──────────────────┘             │
//! │                           ▼                                │
//! │                  ┌────────────────┐                        │
//! │                  │ To Other Brain │                        │
//! │                  │    Regions     │                        │
//! │                  └────────────────┘                        │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod attention;
pub mod fusion;
pub mod gating;
pub mod routing;
pub mod sources;

// Re-exports from fusion module
pub use fusion::{OrderbookFusion, PriceFusion, SentimentFusion, VolumeFusion};

// Re-exports from routing module
pub use routing::{Broadcast, Pathways, Priority, Router};

// Re-exports from sources module - Data types
pub use sources::{
    CelestialData, CelestialSource, ExternalDataAggregator, ExternalDataPoint, MoonPhase,
    NewsArticle, NewsSentiment, NewsSource, SpaceWeather, WeatherData, WeatherSource,
};

// Re-exports from sources module - BERT Sentiment Pipeline
pub use sources::{
    BatchSentimentResult, BertModel, BertModelConfig, BertModelVariant, BertSentimentAnalyzer,
    BertSentimentConfig, BertSentimentStats, BertTokenizer, InferenceDevice, ModelOutput,
    SentimentResult, Token, TokenizedInput,
};

// Re-exports from sources module - Sentiment–Qdrant Bridge
pub use sources::{
    RetrievedSentiment, SentimentBridgeStats, SentimentQdrantBridge, SentimentQdrantConfig,
    SimilarityQuery, StorageContext,
};

// Re-exports from sources module - Sentiment Pipeline Orchestrator
pub use sources::{PipelineOutput, PipelineStats, SentimentPipeline, SentimentPipelineConfig};

// Re-exports from sources module - Chronos Time Series Forecasting
pub use sources::{
    BinningMethod, ChronosConfig, ChronosError, ChronosForecast, ChronosInference,
    ChronosTokenizer, EngineState, ForecastCombiner, HorizonSummary, InferenceStats, SpecialTokens,
    TokenizerConfig,
};

// Re-exports from sources module - API Clients
pub use sources::{
    ApiClient, ApiClientConfig, CryptoCompareClient, CryptoPanicClient, NewsApiClient,
    OpenWeatherMapClient, RateLimiter, SpaceWeatherClient,
};

// Re-exports from sources module - Configuration management
pub use sources::{
    AggregatorConfig, ApiKeyConfig, ConfigBuilder, ConfigError, DataSourceConfig,
    ExternalDataConfig, NewsSourceConfig, RedisConfig, WeatherSourceConfig,
};

// Re-exports from sources module - Cache layer
pub use sources::{CacheEntry, CacheError, CacheKey, CacheStats, CacheStatsSnapshot, RedisCache};

// Re-exports from sources module - Data source bridge
pub use sources::{
    BridgeError, BridgeStats, CircuitBreaker, CircuitState, DataSourceBridge,
    DataSourceBridgeBuilder, ExternalDataEvent,
};

// Re-exports from sources module - Aggregator service
pub use sources::{
    AggregatorError, AggregatorMetrics, AggregatorMetricsSnapshot, AggregatorService,
    AggregatorServiceBuilder, AggregatorServiceConfig, CompositeScores, ServiceState,
    UnifiedDataFeed,
};

// Not yet implemented as unified facade types:
// pub use attention::AttentionMechanism;  — unified attention dispatch facade
// pub use gating::GatingMechanism;       — unified gating dispatch facade
//
// Use the individual types directly instead:
//   attention::{CrossAttention, Focus, Gate, Saliency}
//   gating::{SensoryGate, Relevance}
