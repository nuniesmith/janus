//! Configuration for vision pipeline

use serde::{Deserialize, Serialize};

/// Vision pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// DiffGAF configuration
    pub diffgaf: super::diffgaf::DiffGAFConfig,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            diffgaf: super::diffgaf::DiffGAFConfig::default(),
        }
    }
}
