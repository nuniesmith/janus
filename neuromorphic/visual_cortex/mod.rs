//! Visual Cortex: Pattern Recognition & Vision

pub mod feature_extraction;
pub mod gaf;
pub mod parametric_umap;
pub mod preprocessing;
pub mod vivit;

pub use feature_extraction::FeatureExtractor;
pub use preprocessing::Preprocessor;

// Re-exports from parametric_umap module
pub use parametric_umap::{
    // Core UMAP types
    ClusterStats,
    DistanceMetric,
    DriftReport,
    DriftSeverity,
    FuzzySimplicialSet,
    GraphEdge,
    ParametricUmap,
    ProjectedPoint,
    // Qdrant integration
    RetrievedUmapPoint,
    UmapBridgeStats,
    UmapConfig,
    UmapError,
    UmapProjectionMeta,
    UmapQdrantBridge,
    UmapQdrantConfig,
    UmapState,
    UmapTrainingStats,
    // Cluster analysis
    analyze_regime_clusters,
    inter_cluster_distances,
};

// Not yet implemented as unified facade types:
// pub use gaf::DiffGAF;                    — differentiable GAF transform facade
// pub use vivit::VideoVisionTransformer;    — ViViT inference facade
//
// Use the individual types directly instead:
//   gaf::{Differentiable}
//   vivit::{FactorizedAttention, VivitCandle, Persistence}
//   feature_extraction::FeatureExtractor
//   preprocessing::Preprocessor
