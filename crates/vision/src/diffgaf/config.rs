//! DiffGAF configuration

use serde::{Deserialize, Serialize};

/// Configuration for DiffGAF transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffGAFConfig {
    /// Input feature dimension
    pub input_features: usize,

    /// Output image size (H = W)
    pub output_size: usize,

    /// Normalization range (min, max)
    pub norm_range: (f32, f32),

    /// Use smooth arccos approximation
    pub use_smooth_arccos: bool,

    /// Use learnable aggregation weights
    pub use_aggregation_weights: bool,

    /// Epsilon for numerical stability
    pub eps: f32,
}

impl Default for DiffGAFConfig {
    fn default() -> Self {
        Self {
            input_features: 8,       // OHLCV + indicators
            output_size: 224,        // Standard vision model size
            norm_range: (-1.0, 1.0), // Standard GAF range
            use_smooth_arccos: true, // Gradient-safe
            use_aggregation_weights: false,
            eps: 1e-7,
        }
    }
}

impl DiffGAFConfig {
    /// Create new configuration
    pub fn new(input_features: usize, output_size: usize) -> Self {
        Self {
            input_features,
            output_size,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.input_features == 0 {
            return Err(crate::error::VisionError::InvalidConfig(
                "input_features must be > 0".to_string(),
            ));
        }

        if self.output_size == 0 {
            return Err(crate::error::VisionError::InvalidConfig(
                "output_size must be > 0".to_string(),
            ));
        }

        if self.norm_range.0 >= self.norm_range.1 {
            return Err(crate::error::VisionError::InvalidConfig(
                "norm_range.0 must be < norm_range.1".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DiffGAFConfig::default();
        assert_eq!(config.input_features, 8);
        assert_eq!(config.output_size, 224);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation() {
        let mut config = DiffGAFConfig::default();

        // Invalid input_features
        config.input_features = 0;
        assert!(config.validate().is_err());

        // Invalid norm_range
        config.input_features = 8;
        config.norm_range = (1.0, -1.0);
        assert!(config.validate().is_err());
    }
}
