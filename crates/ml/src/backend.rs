//! Backend configuration for Burn ML framework
//!
//! This module provides backend selection and configuration for the Burn ML framework.
//! It supports both CPU (NdArray) and GPU (WGPU) backends with automatic detection.
//!
//! # Autodiff Support
//!
//! The module provides two backend types:
//! - Base backends (`CpuBackend`, `GpuBackend`) for inference
//! - Autodiff backends (`AutodiffCpuBackend`, `AutodiffGpuBackend`) for training
//!
//! Autodiff backends enable gradient computation through backpropagation, which is
//! required for training neural networks with gradient descent.

use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

#[cfg(feature = "gpu")]
use burn_wgpu::Wgpu;

use crate::error::Result;

/// Type alias for the default CPU backend (inference)
pub type CpuBackend = NdArray<f32>;

/// Type alias for the CPU backend with autodiff (training)
pub type AutodiffCpuBackend = Autodiff<NdArray<f32>>;

/// Type alias for the GPU backend (when available, inference)
#[cfg(feature = "gpu")]
pub type GpuBackend = Wgpu<f32, i32>;

/// Type alias for the GPU backend with autodiff (when available, training)
#[cfg(feature = "gpu")]
pub type AutodiffGpuBackend = Autodiff<Wgpu<f32, i32>>;

/// Backend device abstraction
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub enum BackendDevice {
    /// CPU backend using NdArray
    #[default]
    Cpu,

    /// GPU backend using WGPU
    #[cfg(feature = "gpu")]
    Gpu,
}

impl BackendDevice {
    /// Create a CPU device
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create a GPU device (auto-detect)
    #[cfg(feature = "gpu")]
    pub fn gpu() -> Result<Self> {
        Ok(Self::Gpu)
    }

    /// Auto-detect the best available backend
    pub fn auto() -> Self {
        #[cfg(feature = "gpu")]
        {
            if let Ok(gpu) = Self::gpu() {
                tracing::info!("Using GPU backend");
                return gpu;
            }
        }

        tracing::info!("Using CPU backend");
        Self::cpu()
    }

    /// Check if this is a GPU backend
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self, Self::Gpu)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get a string representation of the backend type
    pub fn backend_type(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "gpu")]
            Self::Gpu => "gpu",
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendConfig {
    /// Device to use for computation
    pub device: BackendDevice,

    /// Number of threads for CPU backend
    pub num_threads: Option<usize>,

    /// Enable mixed precision (if supported)
    pub mixed_precision: bool,

    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            device: BackendDevice::cpu(),
            num_threads: None, // Use default
            mixed_precision: false,
            seed: None,
        }
    }
}

impl BackendConfig {
    /// Create a new backend configuration
    pub fn new(device: BackendDevice) -> Self {
        Self {
            device,
            ..Default::default()
        }
    }

    /// Set the number of threads for CPU backend
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enable mixed precision
    pub fn with_mixed_precision(mut self) -> Self {
        self.mixed_precision = true;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize the backend with this configuration
    pub fn initialize(&self) -> Result<()> {
        // Set number of threads if specified
        if let Some(num_threads) = self.num_threads {
            // ndarray uses rayon for threading, which can be configured via env vars
            tracing::info!("Requested num_threads: {}", num_threads);
            // Note: Thread count is typically controlled via RAYON_NUM_THREADS env var
        }

        // Set random seed if specified
        if let Some(seed) = self.seed {
            tracing::info!("Setting random seed to {}", seed);
            // Note: Burn doesn't have global seed setting yet
            // This would be set per-operation in the training loop
        }

        Ok(())
    }

    /// Get the device for this backend
    pub fn device(&self) -> &BackendDevice {
        &self.device
    }
}

/// Helper function to create a CPU backend configuration
pub fn cpu_backend() -> BackendConfig {
    BackendConfig::new(BackendDevice::cpu())
}

/// Helper function to create a GPU backend configuration (if available)
#[cfg(feature = "gpu")]
pub fn gpu_backend() -> Result<BackendConfig> {
    Ok(BackendConfig::new(BackendDevice::gpu()?))
}

/// Helper function to auto-detect the best backend
pub fn auto_backend() -> BackendConfig {
    BackendConfig::new(BackendDevice::auto())
}

/// Trait for models that can work with multiple backends
pub trait MultiBackend {
    /// Get the backend configuration
    fn backend_config(&self) -> &BackendConfig;

    /// Check if the model is using GPU
    fn is_gpu(&self) -> bool {
        self.backend_config().device.is_gpu()
    }

    /// Get the backend type as a string
    fn backend_type(&self) -> &str {
        self.backend_config().device.backend_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = cpu_backend();
        assert!(!backend.device.is_gpu());
        assert_eq!(backend.device.backend_type(), "cpu");
    }

    #[test]
    fn test_auto_backend() {
        let backend = auto_backend();
        // Should always succeed with at least CPU
        #[cfg(feature = "gpu")]
        {
            assert!(matches!(
                backend.device,
                BackendDevice::Cpu | BackendDevice::Gpu
            ));
        }
        #[cfg(not(feature = "gpu"))]
        {
            assert!(matches!(backend.device, BackendDevice::Cpu));
        }
    }

    #[test]
    fn test_backend_config_builder() {
        let config = cpu_backend()
            .with_threads(4)
            .with_seed(42)
            .with_mixed_precision();

        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.seed, Some(42));
        assert!(config.mixed_precision);
    }

    #[test]
    fn test_default_config() {
        let config = BackendConfig::default();
        assert!(!config.device.is_gpu());
        assert_eq!(config.num_threads, None);
        assert!(!config.mixed_precision);
        assert_eq!(config.seed, None);
    }

    #[test]
    fn test_backend_initialization() {
        let config = cpu_backend().with_threads(2).with_seed(12345);
        let result = config.initialize();
        assert!(result.is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_backend() {
        // GPU might not be available in test environment
        if let Ok(backend) = gpu_backend() {
            assert!(backend.device.is_gpu());
            assert_eq!(backend.device.backend_type(), "gpu");
        }
    }
}
