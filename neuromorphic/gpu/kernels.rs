//! Kernel Registry — Pipeline Cache & Dispatch Infrastructure
//!
//! Provides a registry of compute kernels identified by `KernelId`,
//! workgroup size configuration, and dispatch helpers. When wgpu is
//! wired as a real dependency the registry also caches compiled
//! `ComputePipeline` handles so that repeated dispatches of the same
//! kernel avoid redundant shader compilation.
//!
//! In simulation mode the registry validates kernel configurations and
//! routes dispatches to CPU fallback paths in `runtime.rs`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::runtime::{ComputeDispatch, GpuError, GpuRuntime};
use super::shaders::{ShaderCache, ShaderModule};

// ---------------------------------------------------------------------------
// Kernel identifiers
// ---------------------------------------------------------------------------

/// Identifies a specific compute kernel by name and variant.
///
/// Built-in kernels correspond to the WGSL shaders defined in
/// `shaders.rs`. Custom kernels can be registered at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelId {
    /// Tiled matrix multiplication.
    MatMul,
    /// Row-wise softmax.
    Softmax,
    /// GELU activation (tanh approximation).
    Gelu,
    /// Layer normalization with affine parameters.
    LayerNorm,
    /// Scaled dot-product attention (single-head).
    Attention,
    /// Parallel reduction (sum / max / min).
    Reduce,
    /// Embedding table lookup.
    Embedding,
    /// Pairwise Euclidean distance matrix.
    PairwiseDistance,
    /// Element-wise multiply.
    ElementwiseMul,
    /// Element-wise add.
    ElementwiseAdd,
    /// A user-defined kernel identified by name.
    Custom(String),
}

impl KernelId {
    /// Return the shader name used to look up the WGSL source in
    /// the `ShaderCache`.
    pub fn shader_name(&self) -> &str {
        match self {
            Self::MatMul => "matmul",
            Self::Softmax => "softmax",
            Self::Gelu => "gelu",
            Self::LayerNorm => "layernorm",
            Self::Attention => "attention",
            Self::Reduce => "reduce",
            Self::Embedding => "embedding",
            Self::PairwiseDistance => "distance",
            Self::ElementwiseMul => "elementwise_mul",
            Self::ElementwiseAdd => "elementwise_add",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Whether this is a built-in kernel (has a pre-defined WGSL shader).
    pub fn is_builtin(&self) -> bool {
        !matches!(
            self,
            Self::Custom(_) | Self::ElementwiseMul | Self::ElementwiseAdd
        )
    }

    /// List all built-in kernel IDs.
    pub fn all_builtins() -> Vec<KernelId> {
        vec![
            Self::MatMul,
            Self::Softmax,
            Self::Gelu,
            Self::LayerNorm,
            Self::Attention,
            Self::Reduce,
            Self::Embedding,
            Self::PairwiseDistance,
        ]
    }
}

impl fmt::Display for KernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MatMul => write!(f, "MatMul"),
            Self::Softmax => write!(f, "Softmax"),
            Self::Gelu => write!(f, "GELU"),
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::Attention => write!(f, "Attention"),
            Self::Reduce => write!(f, "Reduce"),
            Self::Embedding => write!(f, "Embedding"),
            Self::PairwiseDistance => write!(f, "PairwiseDistance"),
            Self::ElementwiseMul => write!(f, "ElementwiseMul"),
            Self::ElementwiseAdd => write!(f, "ElementwiseAdd"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Workgroup size
// ---------------------------------------------------------------------------

/// Workgroup dimensions for a compute dispatch.
///
/// wgpu requires the workgroup size to be specified both in the shader
/// source (`@workgroup_size(x, y, z)`) and at dispatch time (number of
/// workgroups). This struct describes the per-invocation workgroup
/// dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkgroupSize {
    /// Workgroup extent in X.
    pub x: u32,
    /// Workgroup extent in Y.
    pub y: u32,
    /// Workgroup extent in Z.
    pub z: u32,
}

impl WorkgroupSize {
    /// 1-D workgroup.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// 2-D workgroup.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// 3-D workgroup.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of invocations per workgroup.
    pub const fn total_invocations(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Compute the number of workgroups needed to cover `n` elements
    /// along the X dimension.
    pub const fn workgroups_for_x(&self, n: u32) -> u32 {
        n.div_ceil(self.x)
    }

    /// Compute workgroups for a 2-D grid of (cols, rows).
    pub const fn workgroups_for_2d(&self, cols: u32, rows: u32) -> (u32, u32) {
        (cols.div_ceil(self.x), rows.div_ceil(self.y))
    }

    /// Standard 1-D workgroup of 256 threads.
    pub const STANDARD_1D: Self = Self::new_1d(256);

    /// Standard 2-D workgroup of 16×16 threads.
    pub const STANDARD_2D: Self = Self::new_2d(16, 16);

    /// Small 1-D workgroup of 64 threads.
    pub const SMALL_1D: Self = Self::new_1d(64);
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self::STANDARD_1D
    }
}

impl fmt::Display for WorkgroupSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z == 1 && self.y == 1 {
            write!(f, "wg({})", self.x)
        } else if self.z == 1 {
            write!(f, "wg({}×{})", self.x, self.y)
        } else {
            write!(f, "wg({}×{}×{})", self.x, self.y, self.z)
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel configuration
// ---------------------------------------------------------------------------

/// Configuration for a registered kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// The kernel identifier.
    pub id: KernelId,
    /// Workgroup size used by the shader.
    pub workgroup_size: WorkgroupSize,
    /// Number of input buffers required.
    pub num_inputs: u32,
    /// Number of output buffers produced.
    pub num_outputs: u32,
    /// Optional description.
    pub description: String,
    /// Whether the kernel supports batching.
    pub supports_batching: bool,
    /// Maximum elements this kernel can handle in a single dispatch.
    /// 0 means no limit.
    pub max_elements: u64,
}

impl KernelConfig {
    /// Create a new kernel configuration.
    pub fn new(
        id: KernelId,
        workgroup_size: WorkgroupSize,
        num_inputs: u32,
        num_outputs: u32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            workgroup_size,
            num_inputs,
            num_outputs,
            description: description.into(),
            supports_batching: false,
            max_elements: 0,
        }
    }

    /// Builder: set batching support.
    pub fn with_batching(mut self, supports: bool) -> Self {
        self.supports_batching = supports;
        self
    }

    /// Builder: set max elements.
    pub fn with_max_elements(mut self, max: u64) -> Self {
        self.max_elements = max;
        self
    }
}

// ---------------------------------------------------------------------------
// Kernel statistics
// ---------------------------------------------------------------------------

/// Statistics for a single kernel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KernelStats {
    /// Number of times this kernel has been dispatched.
    pub dispatch_count: u64,
    /// Total execution time (wall clock, including CPU fallback).
    pub total_duration: Duration,
    /// Minimum execution time.
    pub min_duration: Duration,
    /// Maximum execution time.
    pub max_duration: Duration,
    /// Total elements processed.
    pub total_elements: u64,
    /// Number of errors encountered.
    pub error_count: u64,
}

impl KernelStats {
    /// Record a successful dispatch.
    pub fn record(&mut self, duration: Duration, elements: u64) {
        self.dispatch_count += 1;
        self.total_elements += elements;
        self.total_duration += duration;

        if self.dispatch_count == 1 || duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }
    }

    /// Record an error.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Average execution time per dispatch.
    pub fn avg_duration(&self) -> Duration {
        if self.dispatch_count == 0 {
            return Duration::ZERO;
        }
        self.total_duration / self.dispatch_count as u32
    }

    /// Throughput in elements per second (based on average duration).
    pub fn throughput_eps(&self) -> f64 {
        let avg_secs = self.avg_duration().as_secs_f64();
        if avg_secs == 0.0 || self.dispatch_count == 0 {
            return 0.0;
        }
        let avg_elements = self.total_elements as f64 / self.dispatch_count as f64;
        avg_elements / avg_secs
    }
}

impl fmt::Display for KernelStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dispatches={}, avg={:?}, min={:?}, max={:?}, elements={}, errors={}",
            self.dispatch_count,
            self.avg_duration(),
            self.min_duration,
            self.max_duration,
            self.total_elements,
            self.error_count,
        )
    }
}

// ---------------------------------------------------------------------------
// Kernel Registry
// ---------------------------------------------------------------------------

/// Registry of compute kernels and their compiled pipelines.
///
/// The registry serves three purposes:
///
/// 1. **Configuration** — stores per-kernel metadata (workgroup size,
///    binding counts, description).
/// 2. **Pipeline cache** — when wgpu is linked, compiled
///    `ComputePipeline` handles are cached here to avoid redundant
///    shader compilation on repeated dispatches.
/// 3. **Profiling** — records per-kernel dispatch statistics.
#[derive(Debug)]
pub struct KernelRegistry {
    /// Registered kernel configurations.
    configs: HashMap<KernelId, KernelConfig>,
    /// Per-kernel dispatch statistics.
    stats: HashMap<KernelId, KernelStats>,
    /// Shader cache (owned reference).
    shader_cache: ShaderCache,
    /// Whether pipeline compilation has been done.
    /// In simulation mode this is always `false`.
    pipelines_compiled: bool,
    /// Optional maximum number of custom kernels.
    max_custom_kernels: usize,
}

impl KernelRegistry {
    // ── Construction ───────────────────────────────────────────────────

    /// Create a new registry with built-in kernels registered.
    pub fn new() -> Self {
        let mut registry = Self {
            configs: HashMap::new(),
            stats: HashMap::new(),
            shader_cache: ShaderCache::with_builtins(),
            pipelines_compiled: false,
            max_custom_kernels: 64,
        };
        registry.register_builtins();
        registry
    }

    /// Create a registry with a custom shader cache.
    pub fn with_shader_cache(shader_cache: ShaderCache) -> Self {
        let mut registry = Self {
            configs: HashMap::new(),
            stats: HashMap::new(),
            shader_cache,
            pipelines_compiled: false,
            max_custom_kernels: 64,
        };
        registry.register_builtins();
        registry
    }

    /// Register all built-in kernel configurations.
    fn register_builtins(&mut self) {
        // MatMul — tiled 16×16
        self.register(
            KernelConfig::new(
                KernelId::MatMul,
                WorkgroupSize::STANDARD_2D,
                2, // A, B
                1, // C
                "Tiled 16×16 matrix multiplication (C = A × B)",
            )
            .with_batching(true),
        );

        // Softmax — one workgroup per row, 256 threads
        self.register(
            KernelConfig::new(
                KernelId::Softmax,
                WorkgroupSize::STANDARD_1D,
                1,
                1,
                "Row-wise softmax with parallel max/sum reduction",
            )
            .with_batching(true),
        );

        // GELU — element-wise, 256 threads
        self.register(KernelConfig::new(
            KernelId::Gelu,
            WorkgroupSize::STANDARD_1D,
            1,
            1,
            "GELU activation (tanh approximation)",
        ));

        // LayerNorm — one workgroup per row, 256 threads
        self.register(
            KernelConfig::new(
                KernelId::LayerNorm,
                WorkgroupSize::STANDARD_1D,
                3, // input, gamma, beta
                1, // output
                "Layer normalization with affine transform",
            )
            .with_batching(true),
        );

        // Attention — 16×16 grid
        self.register(KernelConfig::new(
            KernelId::Attention,
            WorkgroupSize::STANDARD_2D,
            3, // Q, K, V
            1, // output
            "Scaled dot-product attention (single-head, fused)",
        ));

        // Reduce — 256 threads
        self.register(KernelConfig::new(
            KernelId::Reduce,
            WorkgroupSize::STANDARD_1D,
            1,
            1,
            "Parallel reduction (sum / max / min)",
        ));

        // Embedding — 256 threads
        self.register(KernelConfig::new(
            KernelId::Embedding,
            WorkgroupSize::STANDARD_1D,
            2, // table, indices
            1, // output
            "Embedding table lookup from token indices",
        ));

        // Pairwise Distance — 16×16 grid
        self.register(KernelConfig::new(
            KernelId::PairwiseDistance,
            WorkgroupSize::STANDARD_2D,
            1, // points
            1, // distances
            "Pairwise Euclidean distance matrix",
        ));

        // Element-wise multiply (CPU-only kernel, no dedicated WGSL shader)
        self.register(KernelConfig::new(
            KernelId::ElementwiseMul,
            WorkgroupSize::STANDARD_1D,
            2,
            1,
            "Element-wise multiplication (c = a * b)",
        ));

        // Element-wise add (CPU-only kernel, no dedicated WGSL shader)
        self.register(KernelConfig::new(
            KernelId::ElementwiseAdd,
            WorkgroupSize::STANDARD_1D,
            2,
            1,
            "Element-wise addition (c = a + b)",
        ));
    }

    // ── Registration ───────────────────────────────────────────────────

    /// Register a kernel configuration.
    pub fn register(&mut self, config: KernelConfig) {
        debug!(kernel = %config.id, "Registering kernel");
        let id = config.id.clone();
        self.configs.insert(id.clone(), config);
        self.stats.entry(id).or_default();
    }

    /// Register a custom kernel with a WGSL shader source.
    pub fn register_custom(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
        workgroup_size: WorkgroupSize,
        num_inputs: u32,
        num_outputs: u32,
        description: impl Into<String>,
    ) -> Result<KernelId, GpuError> {
        let name = name.into();

        // Check custom kernel limit.
        let custom_count = self
            .configs
            .keys()
            .filter(|k| matches!(k, KernelId::Custom(_)))
            .count();
        if custom_count >= self.max_custom_kernels {
            return Err(GpuError::Internal(format!(
                "Maximum custom kernels reached ({})",
                self.max_custom_kernels
            )));
        }

        let id = KernelId::Custom(name.clone());

        // Add shader to cache.
        let shader = super::shaders::ShaderModule::new(
            &name,
            source,
            "main",
            num_inputs + num_outputs,
            &description.into(),
        );
        self.shader_cache.insert(shader);

        // Register config.
        self.register(KernelConfig::new(
            id.clone(),
            workgroup_size,
            num_inputs,
            num_outputs,
            format!("Custom kernel: {name}"),
        ));

        info!(kernel = %name, "Custom kernel registered");
        Ok(id)
    }

    /// Unregister a kernel by ID. Built-in kernels cannot be removed.
    pub fn unregister(&mut self, id: &KernelId) -> Result<(), GpuError> {
        if id.is_builtin() {
            return Err(GpuError::Internal(
                "Cannot unregister built-in kernels".into(),
            ));
        }

        self.configs.remove(id);
        self.stats.remove(id);
        if let KernelId::Custom(name) = id {
            self.shader_cache.remove(name);
        }
        Ok(())
    }

    // ── Query ──────────────────────────────────────────────────────────

    /// Get the configuration for a kernel.
    pub fn get_config(&self, id: &KernelId) -> Option<&KernelConfig> {
        self.configs.get(id)
    }

    /// Get the shader module for a kernel.
    pub fn get_shader(&self, id: &KernelId) -> Option<Arc<ShaderModule>> {
        self.shader_cache.get(id.shader_name())
    }

    /// Check if a kernel is registered.
    pub fn is_registered(&self, id: &KernelId) -> bool {
        self.configs.contains_key(id)
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// List all registered kernel IDs.
    pub fn kernel_ids(&self) -> Vec<KernelId> {
        self.configs.keys().cloned().collect()
    }

    /// Number of custom (non-builtin) kernels.
    pub fn custom_kernel_count(&self) -> usize {
        self.configs
            .keys()
            .filter(|k| matches!(k, KernelId::Custom(_)))
            .count()
    }

    /// Get a reference to the shader cache.
    pub fn shader_cache(&self) -> &ShaderCache {
        &self.shader_cache
    }

    /// Get a mutable reference to the shader cache.
    pub fn shader_cache_mut(&mut self) -> &mut ShaderCache {
        &mut self.shader_cache
    }

    // ── Statistics ─────────────────────────────────────────────────────

    /// Record a successful dispatch for a kernel.
    pub fn record_dispatch(&mut self, id: &KernelId, duration: Duration, elements: u64) {
        if let Some(stats) = self.stats.get_mut(id) {
            stats.record(duration, elements);
        }
    }

    /// Record an error for a kernel.
    pub fn record_error(&mut self, id: &KernelId) {
        if let Some(stats) = self.stats.get_mut(id) {
            stats.record_error();
        }
    }

    /// Get statistics for a specific kernel.
    pub fn get_stats(&self, id: &KernelId) -> Option<&KernelStats> {
        self.stats.get(id)
    }

    /// Get statistics for all kernels.
    pub fn all_stats(&self) -> &HashMap<KernelId, KernelStats> {
        &self.stats
    }

    /// Total dispatches across all kernels.
    pub fn total_dispatches(&self) -> u64 {
        self.stats.values().map(|s| s.dispatch_count).sum()
    }

    /// Total errors across all kernels.
    pub fn total_errors(&self) -> u64 {
        self.stats.values().map(|s| s.error_count).sum()
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        for stats in self.stats.values_mut() {
            *stats = KernelStats::default();
        }
    }

    /// Generate a performance report for all kernels with dispatches.
    pub fn performance_report(&self) -> Vec<KernelReport> {
        let mut reports = Vec::new();
        for (id, stats) in &self.stats {
            if stats.dispatch_count == 0 {
                continue;
            }
            let config = self.configs.get(id);
            reports.push(KernelReport {
                kernel: id.to_string(),
                description: config.map(|c| c.description.clone()).unwrap_or_default(),
                dispatch_count: stats.dispatch_count,
                avg_duration: stats.avg_duration(),
                min_duration: stats.min_duration,
                max_duration: stats.max_duration,
                total_elements: stats.total_elements,
                throughput_eps: stats.throughput_eps(),
                error_count: stats.error_count,
            });
        }
        reports.sort_by(|a, b| b.dispatch_count.cmp(&a.dispatch_count));
        reports
    }

    // ── Dispatch helpers ───────────────────────────────────────────────

    /// Compute the `ComputeDispatch` (workgroup counts) for a 1-D
    /// kernel over `n` elements.
    pub fn dispatch_1d(&self, id: &KernelId, n: u32) -> Result<ComputeDispatch, GpuError> {
        let config = self
            .configs
            .get(id)
            .ok_or_else(|| GpuError::Internal(format!("Kernel {id} not registered")))?;

        Ok(ComputeDispatch::new_1d(
            config.workgroup_size.workgroups_for_x(n),
        ))
    }

    /// Compute the `ComputeDispatch` for a 2-D kernel over a
    /// (cols, rows) grid.
    pub fn dispatch_2d(
        &self,
        id: &KernelId,
        cols: u32,
        rows: u32,
    ) -> Result<ComputeDispatch, GpuError> {
        let config = self
            .configs
            .get(id)
            .ok_or_else(|| GpuError::Internal(format!("Kernel {id} not registered")))?;

        let (wg_x, wg_y) = config.workgroup_size.workgroups_for_2d(cols, rows);
        Ok(ComputeDispatch::new_2d(wg_x, wg_y))
    }

    /// Validate that a dispatch has the correct number of input/output
    /// buffers for the given kernel.
    pub fn validate_dispatch(
        &self,
        id: &KernelId,
        num_inputs: u32,
        num_outputs: u32,
    ) -> Result<(), GpuError> {
        let config = self
            .configs
            .get(id)
            .ok_or_else(|| GpuError::Internal(format!("Kernel {id} not registered")))?;

        if num_inputs != config.num_inputs {
            return Err(GpuError::DispatchError(format!(
                "Kernel {id} expects {} input buffers, got {num_inputs}",
                config.num_inputs
            )));
        }
        if num_outputs != config.num_outputs {
            return Err(GpuError::DispatchError(format!(
                "Kernel {id} expects {} output buffers, got {num_outputs}",
                config.num_outputs
            )));
        }
        Ok(())
    }

    /// Helper: execute a kernel through the runtime, recording stats.
    ///
    /// This is a convenience method that validates buffer counts,
    /// computes the dispatch size, runs the kernel through the
    /// runtime's `dispatch_compute`, and records timing statistics.
    pub fn execute<F>(
        &mut self,
        runtime: &mut GpuRuntime,
        id: &KernelId,
        dispatch: &ComputeDispatch,
        input_ids: &[u64],
        output_id: u64,
        elements: u64,
        cpu_fallback: F,
    ) -> Result<(), GpuError>
    where
        F: FnOnce(&[&[u8]]) -> Vec<u8>,
    {
        // Validate.
        let config = self
            .configs
            .get(id)
            .ok_or_else(|| GpuError::Internal(format!("Kernel {id} not registered")))?;

        if input_ids.len() as u32 != config.num_inputs {
            warn!(
                kernel = %id,
                expected = config.num_inputs,
                got = input_ids.len(),
                "Input buffer count mismatch (proceeding anyway)"
            );
        }

        // Time the dispatch.
        let start = Instant::now();
        let label = id.to_string();

        let result = runtime.dispatch_compute(&label, dispatch, input_ids, output_id, cpu_fallback);

        let duration = start.elapsed();

        match &result {
            Ok(()) => {
                self.record_dispatch(id, duration, elements);
            }
            Err(_) => {
                self.record_error(id);
            }
        }

        result
    }

    // ── Pipeline compilation (wgpu integration point) ──────────────────

    /// Compile all registered shaders into compute pipelines.
    ///
    /// In simulation mode this is a no-op. When wgpu is linked this
    /// will iterate over all registered kernels, compile their WGSL
    /// sources, and cache the resulting `ComputePipeline` handles.
    pub fn compile_all_pipelines(&mut self, _runtime: &GpuRuntime) -> Result<(), GpuError> {
        if self.pipelines_compiled {
            debug!("Pipelines already compiled, skipping");
            return Ok(());
        }

        let start = Instant::now();
        let mut compiled = 0u32;

        for id in self.configs.keys() {
            if let Some(shader) = self.shader_cache.get(id.shader_name()) {
                debug!(kernel = %id, source_len = shader.source_len(), "Compiling pipeline (simulated)");
                compiled += 1;
            } else if id.is_builtin() {
                warn!(kernel = %id, "Built-in kernel missing shader source");
            }
        }

        self.pipelines_compiled = true;
        let elapsed = start.elapsed();
        info!(
            compiled,
            elapsed_ms = elapsed.as_millis(),
            "Pipeline compilation completed (simulated)"
        );

        Ok(())
    }

    /// Whether pipelines have been compiled.
    pub fn pipelines_compiled(&self) -> bool {
        self.pipelines_compiled
    }

    /// Force recompilation of all pipelines.
    pub fn invalidate_pipelines(&mut self) {
        self.pipelines_compiled = false;
        info!("Pipeline cache invalidated");
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Performance report
// ---------------------------------------------------------------------------

/// A summary of a kernel's performance characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelReport {
    /// Kernel name.
    pub kernel: String,
    /// Kernel description.
    pub description: String,
    /// Total dispatch count.
    pub dispatch_count: u64,
    /// Average duration per dispatch.
    pub avg_duration: Duration,
    /// Minimum duration.
    pub min_duration: Duration,
    /// Maximum duration.
    pub max_duration: Duration,
    /// Total elements processed.
    pub total_elements: u64,
    /// Throughput (elements/second).
    pub throughput_eps: f64,
    /// Number of errors.
    pub error_count: u64,
}

impl fmt::Display for KernelReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: dispatches={}, avg={:?}, throughput={:.0} elem/s, errors={}",
            self.kernel,
            self.dispatch_count,
            self.avg_duration,
            self.throughput_eps,
            self.error_count,
        )
    }
}

// ---------------------------------------------------------------------------
// Dispatch plan — for composing multi-kernel pipelines
// ---------------------------------------------------------------------------

/// A step in a multi-kernel dispatch plan.
#[derive(Debug, Clone)]
pub struct DispatchStep {
    /// The kernel to dispatch.
    pub kernel_id: KernelId,
    /// Dispatch dimensions.
    pub dispatch: ComputeDispatch,
    /// Input buffer IDs (resolved at execution time).
    pub input_labels: Vec<String>,
    /// Output buffer label.
    pub output_label: String,
    /// Estimated element count for profiling.
    pub estimated_elements: u64,
}

/// A plan of sequential kernel dispatches forming a compute graph.
///
/// This is a convenience for building multi-step GPU pipelines
/// (e.g., matmul → GELU → layer_norm → attention) without manually
/// managing intermediate buffers.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Ordered list of dispatch steps.
    pub steps: Vec<DispatchStep>,
    /// Human-readable label for the plan.
    pub label: String,
}

impl DispatchPlan {
    /// Create a new empty dispatch plan.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            label: label.into(),
        }
    }

    /// Add a step to the plan.
    pub fn add_step(
        &mut self,
        kernel_id: KernelId,
        dispatch: ComputeDispatch,
        input_labels: Vec<String>,
        output_label: impl Into<String>,
        estimated_elements: u64,
    ) -> &mut Self {
        self.steps.push(DispatchStep {
            kernel_id,
            dispatch,
            input_labels,
            output_label: output_label.into(),
            estimated_elements,
        });
        self
    }

    /// Number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Validate that all kernels in the plan are registered.
    pub fn validate(&self, registry: &KernelRegistry) -> Result<(), GpuError> {
        for (i, step) in self.steps.iter().enumerate() {
            if !registry.is_registered(&step.kernel_id) {
                return Err(GpuError::DispatchError(format!(
                    "Step {i}: kernel {} not registered",
                    step.kernel_id
                )));
            }
        }
        Ok(())
    }

    /// Total estimated elements across all steps.
    pub fn total_estimated_elements(&self) -> u64 {
        self.steps.iter().map(|s| s.estimated_elements).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::runtime::{GpuConfig, GpuRuntime};

    // ── KernelId tests ─────────────────────────────────────────────────

    #[test]
    fn test_kernel_id_shader_name() {
        assert_eq!(KernelId::MatMul.shader_name(), "matmul");
        assert_eq!(KernelId::Softmax.shader_name(), "softmax");
        assert_eq!(KernelId::Gelu.shader_name(), "gelu");
        assert_eq!(KernelId::LayerNorm.shader_name(), "layernorm");
        assert_eq!(KernelId::Attention.shader_name(), "attention");
        assert_eq!(KernelId::Reduce.shader_name(), "reduce");
        assert_eq!(KernelId::Embedding.shader_name(), "embedding");
        assert_eq!(KernelId::PairwiseDistance.shader_name(), "distance");
        assert_eq!(KernelId::ElementwiseMul.shader_name(), "elementwise_mul");
        assert_eq!(KernelId::ElementwiseAdd.shader_name(), "elementwise_add");
        assert_eq!(
            KernelId::Custom("my_kernel".into()).shader_name(),
            "my_kernel"
        );
    }

    #[test]
    fn test_kernel_id_is_builtin() {
        assert!(KernelId::MatMul.is_builtin());
        assert!(KernelId::Softmax.is_builtin());
        assert!(!KernelId::Custom("x".into()).is_builtin());
        assert!(!KernelId::ElementwiseMul.is_builtin());
        assert!(!KernelId::ElementwiseAdd.is_builtin());
    }

    #[test]
    fn test_kernel_id_all_builtins() {
        let builtins = KernelId::all_builtins();
        assert_eq!(builtins.len(), 8);
        assert!(builtins.contains(&KernelId::MatMul));
        assert!(builtins.contains(&KernelId::Attention));
    }

    #[test]
    fn test_kernel_id_display() {
        assert_eq!(format!("{}", KernelId::MatMul), "MatMul");
        assert_eq!(format!("{}", KernelId::Gelu), "GELU");
        assert_eq!(
            format!("{}", KernelId::Custom("test".into())),
            "Custom(test)"
        );
    }

    #[test]
    fn test_kernel_id_eq_hash() {
        let a = KernelId::MatMul;
        let b = KernelId::MatMul;
        assert_eq!(a, b);

        let c = KernelId::Custom("x".into());
        let d = KernelId::Custom("x".into());
        assert_eq!(c, d);

        let e = KernelId::Custom("y".into());
        assert_ne!(c, e);
    }

    // ── WorkgroupSize tests ────────────────────────────────────────────

    #[test]
    fn test_workgroup_size_1d() {
        let wg = WorkgroupSize::new_1d(256);
        assert_eq!(wg.x, 256);
        assert_eq!(wg.y, 1);
        assert_eq!(wg.z, 1);
        assert_eq!(wg.total_invocations(), 256);
    }

    #[test]
    fn test_workgroup_size_2d() {
        let wg = WorkgroupSize::new_2d(16, 16);
        assert_eq!(wg.total_invocations(), 256);
    }

    #[test]
    fn test_workgroup_size_3d() {
        let wg = WorkgroupSize::new_3d(8, 8, 4);
        assert_eq!(wg.total_invocations(), 256);
    }

    #[test]
    fn test_workgroup_size_for_x() {
        let wg = WorkgroupSize::new_1d(256);
        assert_eq!(wg.workgroups_for_x(256), 1);
        assert_eq!(wg.workgroups_for_x(257), 2);
        assert_eq!(wg.workgroups_for_x(512), 2);
        assert_eq!(wg.workgroups_for_x(1), 1);
        assert_eq!(wg.workgroups_for_x(1000), 4);
    }

    #[test]
    fn test_workgroup_size_for_2d() {
        let wg = WorkgroupSize::new_2d(16, 16);
        let (wx, wy) = wg.workgroups_for_2d(32, 32);
        assert_eq!(wx, 2);
        assert_eq!(wy, 2);

        let (wx, wy) = wg.workgroups_for_2d(17, 1);
        assert_eq!(wx, 2);
        assert_eq!(wy, 1);
    }

    #[test]
    fn test_workgroup_size_constants() {
        assert_eq!(WorkgroupSize::STANDARD_1D.total_invocations(), 256);
        assert_eq!(WorkgroupSize::STANDARD_2D.total_invocations(), 256);
        assert_eq!(WorkgroupSize::SMALL_1D.total_invocations(), 64);
    }

    #[test]
    fn test_workgroup_size_display() {
        assert_eq!(format!("{}", WorkgroupSize::new_1d(256)), "wg(256)");
        assert_eq!(format!("{}", WorkgroupSize::new_2d(16, 16)), "wg(16×16)");
        assert_eq!(format!("{}", WorkgroupSize::new_3d(8, 8, 4)), "wg(8×8×4)");
    }

    #[test]
    fn test_workgroup_size_default() {
        let wg = WorkgroupSize::default();
        assert_eq!(wg, WorkgroupSize::STANDARD_1D);
    }

    // ── KernelConfig tests ─────────────────────────────────────────────

    #[test]
    fn test_kernel_config_new() {
        let config = KernelConfig::new(
            KernelId::MatMul,
            WorkgroupSize::STANDARD_2D,
            2,
            1,
            "Test matmul",
        );
        assert_eq!(config.id, KernelId::MatMul);
        assert_eq!(config.num_inputs, 2);
        assert_eq!(config.num_outputs, 1);
        assert!(!config.supports_batching);
        assert_eq!(config.max_elements, 0);
    }

    #[test]
    fn test_kernel_config_builder() {
        let config = KernelConfig::new(
            KernelId::Softmax,
            WorkgroupSize::STANDARD_1D,
            1,
            1,
            "Softmax",
        )
        .with_batching(true)
        .with_max_elements(1_000_000);

        assert!(config.supports_batching);
        assert_eq!(config.max_elements, 1_000_000);
    }

    // ── KernelStats tests ──────────────────────────────────────────────

    #[test]
    fn test_kernel_stats_default() {
        let stats = KernelStats::default();
        assert_eq!(stats.dispatch_count, 0);
        assert_eq!(stats.total_elements, 0);
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.avg_duration(), Duration::ZERO);
        assert_eq!(stats.throughput_eps(), 0.0);
    }

    #[test]
    fn test_kernel_stats_record() {
        let mut stats = KernelStats::default();
        stats.record(Duration::from_micros(100), 1000);
        assert_eq!(stats.dispatch_count, 1);
        assert_eq!(stats.total_elements, 1000);
        assert_eq!(stats.min_duration, Duration::from_micros(100));
        assert_eq!(stats.max_duration, Duration::from_micros(100));

        stats.record(Duration::from_micros(50), 500);
        assert_eq!(stats.dispatch_count, 2);
        assert_eq!(stats.total_elements, 1500);
        assert_eq!(stats.min_duration, Duration::from_micros(50));
        assert_eq!(stats.max_duration, Duration::from_micros(100));
    }

    #[test]
    fn test_kernel_stats_avg_duration() {
        let mut stats = KernelStats::default();
        stats.record(Duration::from_micros(100), 0);
        stats.record(Duration::from_micros(200), 0);
        let avg = stats.avg_duration();
        assert_eq!(avg, Duration::from_micros(150));
    }

    #[test]
    fn test_kernel_stats_error() {
        let mut stats = KernelStats::default();
        stats.record_error();
        stats.record_error();
        assert_eq!(stats.error_count, 2);
    }

    #[test]
    fn test_kernel_stats_display() {
        let mut stats = KernelStats::default();
        stats.record(Duration::from_micros(100), 1000);
        let s = format!("{stats}");
        assert!(s.contains("dispatches=1"));
        assert!(s.contains("elements=1000"));
    }

    // ── KernelRegistry tests ───────────────────────────────────────────

    #[test]
    fn test_registry_new_has_builtins() {
        let registry = KernelRegistry::new();
        assert!(registry.len() >= 10);
        assert!(registry.is_registered(&KernelId::MatMul));
        assert!(registry.is_registered(&KernelId::Softmax));
        assert!(registry.is_registered(&KernelId::Gelu));
        assert!(registry.is_registered(&KernelId::LayerNorm));
        assert!(registry.is_registered(&KernelId::Attention));
        assert!(registry.is_registered(&KernelId::Reduce));
        assert!(registry.is_registered(&KernelId::Embedding));
        assert!(registry.is_registered(&KernelId::PairwiseDistance));
        assert!(registry.is_registered(&KernelId::ElementwiseMul));
        assert!(registry.is_registered(&KernelId::ElementwiseAdd));
    }

    #[test]
    fn test_registry_default() {
        let registry = KernelRegistry::default();
        assert!(registry.len() >= 10);
    }

    #[test]
    fn test_registry_get_config() {
        let registry = KernelRegistry::new();
        let config = registry.get_config(&KernelId::MatMul).unwrap();
        assert_eq!(config.id, KernelId::MatMul);
        assert_eq!(config.num_inputs, 2);
        assert_eq!(config.num_outputs, 1);
        assert!(config.supports_batching);
    }

    #[test]
    fn test_registry_get_shader() {
        let registry = KernelRegistry::new();
        let shader = registry.get_shader(&KernelId::MatMul);
        assert!(shader.is_some());
        let shader = shader.unwrap();
        assert_eq!(shader.name, "matmul");
    }

    #[test]
    fn test_registry_register_custom() {
        let mut registry = KernelRegistry::new();
        let initial = registry.len();

        let id = registry
            .register_custom(
                "test_kernel",
                "@compute @workgroup_size(64) fn main() {}",
                WorkgroupSize::SMALL_1D,
                1,
                1,
                "Test custom kernel",
            )
            .unwrap();

        assert_eq!(id, KernelId::Custom("test_kernel".into()));
        assert_eq!(registry.len(), initial + 1);
        assert!(registry.is_registered(&id));
        assert_eq!(registry.custom_kernel_count(), 1);

        // Should have shader in cache.
        let shader = registry.get_shader(&id);
        assert!(shader.is_some());
    }

    #[test]
    fn test_registry_unregister_custom() {
        let mut registry = KernelRegistry::new();
        let id = registry
            .register_custom(
                "to_remove",
                "fn main() {}",
                WorkgroupSize::SMALL_1D,
                1,
                1,
                "Temporary",
            )
            .unwrap();

        assert!(registry.is_registered(&id));
        registry.unregister(&id).unwrap();
        assert!(!registry.is_registered(&id));
    }

    #[test]
    fn test_registry_cannot_unregister_builtin() {
        let mut registry = KernelRegistry::new();
        let result = registry.unregister(&KernelId::MatMul);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_kernel_ids() {
        let registry = KernelRegistry::new();
        let ids = registry.kernel_ids();
        assert!(ids.contains(&KernelId::MatMul));
        assert!(ids.contains(&KernelId::Softmax));
    }

    #[test]
    fn test_registry_dispatch_1d() {
        let registry = KernelRegistry::new();
        let dispatch = registry.dispatch_1d(&KernelId::Gelu, 1000).unwrap();
        // GELU uses 256-thread workgroups → ceil(1000/256) = 4
        assert_eq!(dispatch.workgroups_x, 4);
        assert_eq!(dispatch.workgroups_y, 1);
    }

    #[test]
    fn test_registry_dispatch_2d() {
        let registry = KernelRegistry::new();
        let dispatch = registry.dispatch_2d(&KernelId::MatMul, 32, 32).unwrap();
        // MatMul uses 16×16 workgroups → ceil(32/16) = 2 in each dim
        assert_eq!(dispatch.workgroups_x, 2);
        assert_eq!(dispatch.workgroups_y, 2);
    }

    #[test]
    fn test_registry_dispatch_unregistered() {
        let registry = KernelRegistry::new();
        let result = registry.dispatch_1d(&KernelId::Custom("nope".into()), 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_validate_dispatch() {
        let registry = KernelRegistry::new();

        // Correct buffer counts
        assert!(registry.validate_dispatch(&KernelId::MatMul, 2, 1).is_ok());

        // Wrong input count
        assert!(registry.validate_dispatch(&KernelId::MatMul, 1, 1).is_err());

        // Wrong output count
        assert!(registry.validate_dispatch(&KernelId::MatMul, 2, 2).is_err());
    }

    #[test]
    fn test_registry_stats_recording() {
        let mut registry = KernelRegistry::new();

        registry.record_dispatch(&KernelId::MatMul, Duration::from_micros(500), 10000);
        registry.record_dispatch(&KernelId::MatMul, Duration::from_micros(300), 5000);

        let stats = registry.get_stats(&KernelId::MatMul).unwrap();
        assert_eq!(stats.dispatch_count, 2);
        assert_eq!(stats.total_elements, 15000);
        assert_eq!(stats.min_duration, Duration::from_micros(300));
        assert_eq!(stats.max_duration, Duration::from_micros(500));
    }

    #[test]
    fn test_registry_total_dispatches() {
        let mut registry = KernelRegistry::new();
        registry.record_dispatch(&KernelId::MatMul, Duration::from_micros(100), 0);
        registry.record_dispatch(&KernelId::Gelu, Duration::from_micros(100), 0);
        registry.record_dispatch(&KernelId::Gelu, Duration::from_micros(100), 0);
        assert_eq!(registry.total_dispatches(), 3);
    }

    #[test]
    fn test_registry_total_errors() {
        let mut registry = KernelRegistry::new();
        registry.record_error(&KernelId::MatMul);
        registry.record_error(&KernelId::Softmax);
        assert_eq!(registry.total_errors(), 2);
    }

    #[test]
    fn test_registry_reset_stats() {
        let mut registry = KernelRegistry::new();
        registry.record_dispatch(&KernelId::MatMul, Duration::from_micros(100), 1000);
        assert_eq!(registry.total_dispatches(), 1);

        registry.reset_stats();
        assert_eq!(registry.total_dispatches(), 0);
    }

    #[test]
    fn test_registry_performance_report() {
        let mut registry = KernelRegistry::new();
        registry.record_dispatch(&KernelId::MatMul, Duration::from_micros(500), 10000);
        registry.record_dispatch(&KernelId::Gelu, Duration::from_micros(100), 5000);
        registry.record_dispatch(&KernelId::Gelu, Duration::from_micros(200), 5000);

        let report = registry.performance_report();
        assert_eq!(report.len(), 2);

        // Sorted by dispatch_count descending — GELU has 2, MatMul has 1
        assert_eq!(report[0].kernel, "GELU");
        assert_eq!(report[0].dispatch_count, 2);
        assert_eq!(report[1].kernel, "MatMul");
        assert_eq!(report[1].dispatch_count, 1);
    }

    #[test]
    fn test_registry_performance_report_empty() {
        let registry = KernelRegistry::new();
        let report = registry.performance_report();
        assert!(report.is_empty());
    }

    #[test]
    fn test_registry_compile_pipelines_simulated() {
        let mut registry = KernelRegistry::new();
        let runtime = GpuRuntime::new_simulated(GpuConfig::default());

        assert!(!registry.pipelines_compiled());
        registry.compile_all_pipelines(&runtime).unwrap();
        assert!(registry.pipelines_compiled());

        // Second call should be a no-op.
        registry.compile_all_pipelines(&runtime).unwrap();
    }

    #[test]
    fn test_registry_invalidate_pipelines() {
        let mut registry = KernelRegistry::new();
        let runtime = GpuRuntime::new_simulated(GpuConfig::default());

        registry.compile_all_pipelines(&runtime).unwrap();
        assert!(registry.pipelines_compiled());

        registry.invalidate_pipelines();
        assert!(!registry.pipelines_compiled());
    }

    #[test]

    fn test_registry_execute() {
        use crate::gpu::runtime::GpuBufferUsage;

        let mut registry = KernelRegistry::new();
        let mut runtime = GpuRuntime::new_simulated(GpuConfig::default());

        // Upload input
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let in_id = runtime
            .upload_buffer(&bytes, GpuBufferUsage::Storage, Some("in"))
            .unwrap();

        // Allocate output
        let out_id = runtime
            .allocate_buffer(16, GpuBufferUsage::StorageReadback, Some("out"))
            .unwrap();

        let dispatch = ComputeDispatch::for_elements(4, 256);

        registry
            .execute(
                &mut runtime,
                &KernelId::Gelu,
                &dispatch,
                &[in_id],
                out_id,
                4,
                |inputs| {
                    let n = inputs[0].len() / 4;
                    let mut result = Vec::with_capacity(inputs[0].len());
                    for i in 0..n {
                        let arr = [
                            inputs[0][i * 4],
                            inputs[0][i * 4 + 1],
                            inputs[0][i * 4 + 2],
                            inputs[0][i * 4 + 3],
                        ];
                        let x = f32::from_le_bytes(arr);
                        let sqrt_2_over_pi: f32 = (2.0f32 / std::f32::consts::PI).sqrt();
                        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                        let y = 0.5 * x * (1.0 + inner.tanh());
                        result.extend_from_slice(&y.to_le_bytes());
                    }
                    result
                },
            )
            .unwrap();

        let stats = registry.get_stats(&KernelId::Gelu).unwrap();
        assert_eq!(stats.dispatch_count, 1);
        assert_eq!(stats.total_elements, 4);
    }

    // ── DispatchPlan tests ─────────────────────────────────────────────

    #[test]
    fn test_dispatch_plan_new() {
        let plan = DispatchPlan::new("test_plan");
        assert_eq!(plan.label, "test_plan");
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn test_dispatch_plan_add_steps() {
        let mut plan = DispatchPlan::new("matmul_gelu");

        plan.add_step(
            KernelId::MatMul,
            ComputeDispatch::new_2d(2, 2),
            vec!["A".into(), "B".into()],
            "C",
            1024,
        );

        plan.add_step(
            KernelId::Gelu,
            ComputeDispatch::new_1d(4),
            vec!["C".into()],
            "D",
            1024,
        );

        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
        assert_eq!(plan.total_estimated_elements(), 2048);
    }

    #[test]
    fn test_dispatch_plan_validate_ok() {
        let registry = KernelRegistry::new();
        let mut plan = DispatchPlan::new("valid");
        plan.add_step(
            KernelId::MatMul,
            ComputeDispatch::new_1d(1),
            vec![],
            "out",
            0,
        );
        assert!(plan.validate(&registry).is_ok());
    }

    #[test]
    fn test_dispatch_plan_validate_missing_kernel() {
        let registry = KernelRegistry::new();
        let mut plan = DispatchPlan::new("invalid");
        plan.add_step(
            KernelId::Custom("nonexistent".into()),
            ComputeDispatch::new_1d(1),
            vec![],
            "out",
            0,
        );
        assert!(plan.validate(&registry).is_err());
    }

    // ── KernelReport tests ─────────────────────────────────────────────

    #[test]
    fn test_kernel_report_display() {
        let report = KernelReport {
            kernel: "MatMul".into(),
            description: "Matrix multiply".into(),
            dispatch_count: 10,
            avg_duration: Duration::from_micros(500),
            min_duration: Duration::from_micros(100),
            max_duration: Duration::from_micros(1000),
            total_elements: 100_000,
            throughput_eps: 200_000_000.0,
            error_count: 0,
        };
        let s = format!("{report}");
        assert!(s.contains("MatMul"));
        assert!(s.contains("dispatches=10"));
        assert!(s.contains("errors=0"));
    }
}
