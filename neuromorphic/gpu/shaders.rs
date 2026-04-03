//! WGSL Compute Shader Sources
//!
//! Contains all WGSL shader source strings for GPU compute kernels.
//! Each shader is designed for wgpu's compute pipeline and uses
//! workgroup-level optimizations where applicable.
//!
//! When the runtime is in simulation mode these shaders are not
//! actually compiled — the CPU fallback paths in `runtime.rs` execute
//! instead. Once wgpu is wired as a real dependency, these shaders
//! are fed to `device.create_shader_module()`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shader source constants
// ---------------------------------------------------------------------------

/// Matrix multiplication: C = A × B (tiled 16×16 workgroups).
///
/// Bindings:
///   @group(0) @binding(0) A — storage, read  (M × K, row-major f32)
///   @group(0) @binding(1) B — storage, read  (K × N, row-major f32)
///   @group(0) @binding(2) C — storage, write (M × N, row-major f32)
///   @group(0) @binding(3) dims — uniform { M: u32, K: u32, N: u32 }
pub const SHADER_MATMUL: &str = r#"
// ─── Matrix Multiply (tiled 16×16) ────────────────────────────────────

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    var acc: f32 = 0.0;
    let numTiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A
        let aCol = t * TILE + lid.x;
        if (row < dims.M && aCol < dims.K) {
            tileA[lid.y][lid.x] = A[row * dims.K + aCol];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }

        // Load tile of B
        let bRow = t * TILE + lid.y;
        if (bRow < dims.K && col < dims.N) {
            tileB[lid.y][lid.x] = B[bRow * dims.N + col];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        // Accumulate
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tileA[lid.y][k] * tileB[k][lid.x];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
"#;

/// Softmax over rows of a 2-D matrix.
///
/// Two-pass algorithm:
///   Pass 1: find row max (reduction).
///   Pass 2: exp(x - max) and normalise.
///
/// For simplicity this shader handles one row per workgroup.
///
/// Bindings:
///   @group(0) @binding(0) input  — storage, read  (rows × cols f32)
///   @group(0) @binding(1) output — storage, write (rows × cols f32)
///   @group(0) @binding(2) params — uniform { rows: u32, cols: u32 }
pub const SHADER_SOFTMAX: &str = r#"
// ─── Row-wise Softmax ─────────────────────────────────────────────────

struct Params {
    rows: u32,
    cols: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wgid: vec3<u32>) {
    let row = wgid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let base = row * params.cols;

    // ── Pass 1: parallel max reduction ─────────────────────────────
    var local_max: f32 = -3.402823e+38; // -FLT_MAX
    var i: u32 = tid;
    while (i < params.cols) {
        local_max = max(local_max, input[base + i]);
        i = i + WG;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    var stride: u32 = WG / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // ── Pass 2: exp and sum ────────────────────────────────────────
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < params.cols) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        local_sum = local_sum + e;
        i = i + WG;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    stride = WG / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total = shared_sum[0];
    workgroupBarrier();

    // ── Pass 3: normalise ──────────────────────────────────────────
    let inv_total = 1.0 / total;
    i = tid;
    while (i < params.cols) {
        output[base + i] = output[base + i] * inv_total;
        i = i + WG;
    }
}
"#;

/// GELU activation (tanh approximation).
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Bindings:
///   @group(0) @binding(0) input  — storage, read
///   @group(0) @binding(1) output — storage, write
///   @group(0) @binding(2) params — uniform { len: u32 }
pub const SHADER_GELU: &str = r#"
// ─── GELU Activation ──────────────────────────────────────────────────

struct Params {
    len: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
const COEFF: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) { return; }

    let x = input[idx];
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

/// Layer normalization.
///
/// For each row of (batch, dim):
///   y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// One workgroup per row.
///
/// Bindings:
///   @group(0) @binding(0) input  — storage, read  (batch × dim)
///   @group(0) @binding(1) gamma  — storage, read   (dim)
///   @group(0) @binding(2) beta   — storage, read   (dim)
///   @group(0) @binding(3) output — storage, write  (batch × dim)
///   @group(0) @binding(4) params — uniform { batch: u32, dim: u32, eps: f32 }
pub const SHADER_LAYERNORM: &str = r#"
// ─── Layer Normalization ──────────────────────────────────────────────

struct Params {
    batch: u32,
    dim:   u32,
    eps:   f32,
    _pad:  u32,
};

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(2) var<storage, read>       beta:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

const WG: u32 = 256u;

var<workgroup> shared_sum:  array<f32, 256>;
var<workgroup> shared_sum2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id)        wgid: vec3<u32>) {
    let row = wgid.x;
    if (row >= params.batch) { return; }

    let tid  = lid.x;
    let base = row * params.dim;

    // ── Parallel sum and sum-of-squares ────────────────────────────
    var s:  f32 = 0.0;
    var s2: f32 = 0.0;
    var i: u32 = tid;
    while (i < params.dim) {
        let v = input[base + i];
        s  = s  + v;
        s2 = s2 + v * v;
        i = i + WG;
    }
    shared_sum[tid]  = s;
    shared_sum2[tid] = s2;
    workgroupBarrier();

    // Tree reduction
    var stride: u32 = WG / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid]  = shared_sum[tid]  + shared_sum[tid + stride];
            shared_sum2[tid] = shared_sum2[tid] + shared_sum2[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean     = shared_sum[0]  / f32(params.dim);
    let mean_sq  = shared_sum2[0] / f32(params.dim);
    let variance = mean_sq - mean * mean;
    let inv_std  = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // ── Normalise + affine ─────────────────────────────────────────
    i = tid;
    while (i < params.dim) {
        let x_hat = (input[base + i] - mean) * inv_std;
        output[base + i] = gamma[i] * x_hat + beta[i];
        i = i + WG;
    }
}
"#;

/// Scaled dot-product attention (single-head, fused QK^T + softmax + V).
///
/// Not tiled; straightforward for moderate sequence lengths (≤ 2048).
///
/// Bindings:
///   @group(0) @binding(0) Q      — storage, read  (seq_q × d_k)
///   @group(0) @binding(1) K      — storage, read  (seq_k × d_k)
///   @group(0) @binding(2) V      — storage, read  (seq_k × d_v)
///   @group(0) @binding(3) output — storage, write  (seq_q × d_v)
///   @group(0) @binding(4) params — uniform { seq_q, seq_k, d_k, d_v }
pub const SHADER_ATTENTION: &str = r#"
// ─── Scaled Dot-Product Attention ─────────────────────────────────────

struct Params {
    seq_q: u32,
    seq_k: u32,
    d_k:   u32,
    d_v:   u32,
};

@group(0) @binding(0) var<storage, read>       Q:      array<f32>;
@group(0) @binding(1) var<storage, read>       K:      array<f32>;
@group(0) @binding(2) var<storage, read>       V:      array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

// Each invocation computes one element of the output matrix.
// Dispatch: (d_v, seq_q, 1)  with workgroup_size(16, 16).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.y; // row of output  (query index)
    let v_col = gid.x; // col of output  (value dim)

    if (q_idx >= params.seq_q || v_col >= params.d_v) { return; }

    let scale = 1.0 / sqrt(f32(params.d_k));

    // ── Compute attention scores for this query row ────────────────
    // scores[j] = dot(Q[q_idx, :], K[j, :]) * scale
    var max_score: f32 = -3.402823e+38;
    for (var j: u32 = 0u; j < params.seq_k; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var k: u32 = 0u; k < params.d_k; k = k + 1u) {
            dot = dot + Q[q_idx * params.d_k + k] * K[j * params.d_k + k];
        }
        let s = dot * scale;
        max_score = max(max_score, s);
    }

    // ── Softmax numerator and denominator ──────────────────────────
    var denom: f32 = 0.0;
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.seq_k; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var k: u32 = 0u; k < params.d_k; k = k + 1u) {
            dot = dot + Q[q_idx * params.d_k + k] * K[j * params.d_k + k];
        }
        let w = exp(dot * scale - max_score);
        denom = denom + w;
        acc   = acc   + w * V[j * params.d_v + v_col];
    }

    output[q_idx * params.d_v + v_col] = acc / denom;
}
"#;

/// Parallel reduction (sum / max / min).
///
/// The `mode` uniform selects the operation:
///   0 = sum, 1 = max, 2 = min
///
/// Bindings:
///   @group(0) @binding(0) input  — storage, read  (len f32)
///   @group(0) @binding(1) output — storage, write (1 f32 per workgroup)
///   @group(0) @binding(2) params — uniform { len: u32, mode: u32 }
pub const SHADER_REDUCE: &str = r#"
// ─── Parallel Reduction ───────────────────────────────────────────────

struct Params {
    len:  u32,
    mode: u32, // 0=sum, 1=max, 2=min
    _p1:  u32,
    _p2:  u32,
};

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

const WG: u32 = 256u;
var<workgroup> shared: array<f32, 256>;

fn identity(mode: u32) -> f32 {
    switch (mode) {
        case 0u: { return 0.0; }               // sum identity
        case 1u: { return -3.402823e+38; }      // max identity
        case 2u: { return  3.402823e+38; }      // min identity
        default: { return 0.0; }
    }
}

fn combine(a: f32, b: f32, mode: u32) -> f32 {
    switch (mode) {
        case 0u: { return a + b; }
        case 1u: { return max(a, b); }
        case 2u: { return min(a, b); }
        default: { return a + b; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id)  gid:  vec3<u32>,
        @builtin(local_invocation_id)   lid:  vec3<u32>,
        @builtin(workgroup_id)          wgid: vec3<u32>) {
    let tid = lid.x;

    // Each thread loads multiple elements with striding
    var acc = identity(params.mode);
    var i = gid.x;
    let grid_stride = WG * 65535u; // total threads across all workgroups (approx)
    while (i < params.len) {
        acc = combine(acc, input[i], params.mode);
        i = i + WG;
    }
    shared[tid] = acc;
    workgroupBarrier();

    // Tree reduction within workgroup
    var stride: u32 = WG / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            shared[tid] = combine(shared[tid], shared[tid + stride], params.mode);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        output[wgid.x] = shared[0];
    }
}
"#;

/// Embedding table lookup.
///
/// Given token indices, look up rows from an embedding table.
///
/// Bindings:
///   @group(0) @binding(0) table   — storage, read  (vocab × dim f32)
///   @group(0) @binding(1) indices — storage, read  (n u32)
///   @group(0) @binding(2) output  — storage, write (n × dim f32)
///   @group(0) @binding(3) params  — uniform { n: u32, dim: u32, vocab: u32 }
pub const SHADER_EMBEDDING: &str = r#"
// ─── Embedding Lookup ─────────────────────────────────────────────────

struct Params {
    n:     u32,
    dim:   u32,
    vocab: u32,
    _pad:  u32,
};

@group(0) @binding(0) var<storage, read>       table:   array<f32>;
@group(0) @binding(1) var<storage, read>       indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let token_idx = flat / params.dim;
    let dim_idx   = flat % params.dim;

    if (token_idx >= params.n) { return; }

    let vocab_idx = indices[token_idx];
    if (vocab_idx >= params.vocab) {
        output[flat] = 0.0;
        return;
    }

    output[flat] = table[vocab_idx * params.dim + dim_idx];
}
"#;

/// Pairwise Euclidean distance matrix.
///
/// Input: n points of d dimensions (row-major).
/// Output: n × n symmetric distance matrix.
///
/// Bindings:
///   @group(0) @binding(0) points — storage, read  (n × d f32)
///   @group(0) @binding(1) dists  — storage, write (n × n f32)
///   @group(0) @binding(2) params — uniform { n: u32, d: u32 }
pub const SHADER_DISTANCE: &str = r#"
// ─── Pairwise Euclidean Distance ──────────────────────────────────────

struct Params {
    n: u32,
    d: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read>       points: array<f32>;
@group(0) @binding(1) var<storage, read_write> dists:  array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y;
    let j = gid.x;
    if (i >= params.n || j >= params.n) { return; }

    // Only compute upper triangle + diagonal; mirror.
    if (j < i) {
        dists[i * params.n + j] = dists[j * params.n + i];
        return;
    }

    if (i == j) {
        dists[i * params.n + j] = 0.0;
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.d; k = k + 1u) {
        let diff = points[i * params.d + k] - points[j * params.d + k];
        sum = sum + diff * diff;
    }
    let dist = sqrt(sum);
    dists[i * params.n + j] = dist;
    dists[j * params.n + i] = dist;
}
"#;

// ---------------------------------------------------------------------------
// Shader module descriptor
// ---------------------------------------------------------------------------

/// Metadata for a compiled (or to-be-compiled) shader module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderModule {
    /// Human-readable name / identifier.
    pub name: String,
    /// The WGSL source code.
    pub source: String,
    /// Entry point function name.
    pub entry_point: String,
    /// Number of bindings expected.
    pub num_bindings: u32,
    /// Description of what the shader does.
    pub description: String,
}

impl ShaderModule {
    /// Create a new shader module descriptor.
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        entry_point: impl Into<String>,
        num_bindings: u32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            entry_point: entry_point.into(),
            num_bindings,
            description: description.into(),
        }
    }

    /// WGSL source length in bytes.
    pub fn source_len(&self) -> usize {
        self.source.len()
    }
}

// ---------------------------------------------------------------------------
// Shader cache
// ---------------------------------------------------------------------------

/// A cache of shader modules keyed by name.
///
/// In a real wgpu integration this would also hold compiled
/// `wgpu::ShaderModule` handles. For now it stores the WGSL sources
/// and metadata so that the kernel registry can look them up.
#[derive(Debug, Clone)]
pub struct ShaderCache {
    modules: HashMap<String, Arc<ShaderModule>>,
}

impl ShaderCache {
    /// Create a new empty shader cache.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Create a shader cache pre-populated with all built-in shaders.
    pub fn with_builtins() -> Self {
        let mut cache = Self::new();
        cache.register_builtins();
        cache
    }

    /// Register all built-in shader modules.
    pub fn register_builtins(&mut self) {
        self.insert(ShaderModule::new(
            "matmul",
            SHADER_MATMUL,
            "main",
            4,
            "Tiled 16×16 matrix multiplication: C = A × B",
        ));

        self.insert(ShaderModule::new(
            "softmax",
            SHADER_SOFTMAX,
            "main",
            3,
            "Row-wise softmax with parallel max/sum reduction",
        ));

        self.insert(ShaderModule::new(
            "gelu",
            SHADER_GELU,
            "main",
            3,
            "GELU activation (tanh approximation)",
        ));

        self.insert(ShaderModule::new(
            "layernorm",
            SHADER_LAYERNORM,
            "main",
            5,
            "Layer normalization with affine transform",
        ));

        self.insert(ShaderModule::new(
            "attention",
            SHADER_ATTENTION,
            "main",
            5,
            "Scaled dot-product attention (single-head, fused)",
        ));

        self.insert(ShaderModule::new(
            "reduce",
            SHADER_REDUCE,
            "main",
            3,
            "Parallel reduction (sum / max / min)",
        ));

        self.insert(ShaderModule::new(
            "embedding",
            SHADER_EMBEDDING,
            "main",
            4,
            "Embedding table lookup from token indices",
        ));

        self.insert(ShaderModule::new(
            "distance",
            SHADER_DISTANCE,
            "main",
            3,
            "Pairwise Euclidean distance matrix",
        ));
    }

    /// Insert a shader module into the cache.
    pub fn insert(&mut self, module: ShaderModule) {
        self.modules.insert(module.name.clone(), Arc::new(module));
    }

    /// Look up a shader by name.
    pub fn get(&self, name: &str) -> Option<Arc<ShaderModule>> {
        self.modules.get(name).cloned()
    }

    /// Check if a shader with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Number of registered shader modules.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Iterate over all registered modules.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<ShaderModule>)> {
        self.modules.iter()
    }

    /// List all registered shader names.
    pub fn names(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// Remove a shader from the cache.
    pub fn remove(&mut self, name: &str) -> Option<Arc<ShaderModule>> {
        self.modules.remove(name)
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.modules.clear();
    }

    /// Total WGSL source bytes across all cached shaders.
    pub fn total_source_bytes(&self) -> usize {
        self.modules.values().map(|m| m.source_len()).sum()
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources_not_empty() {
        assert!(!SHADER_MATMUL.is_empty());
        assert!(!SHADER_SOFTMAX.is_empty());
        assert!(!SHADER_GELU.is_empty());
        assert!(!SHADER_LAYERNORM.is_empty());
        assert!(!SHADER_ATTENTION.is_empty());
        assert!(!SHADER_REDUCE.is_empty());
        assert!(!SHADER_EMBEDDING.is_empty());
        assert!(!SHADER_DISTANCE.is_empty());
    }

    #[test]
    fn test_shader_sources_contain_entry_point() {
        assert!(SHADER_MATMUL.contains("fn main"));
        assert!(SHADER_SOFTMAX.contains("fn main"));
        assert!(SHADER_GELU.contains("fn main"));
        assert!(SHADER_LAYERNORM.contains("fn main"));
        assert!(SHADER_ATTENTION.contains("fn main"));
        assert!(SHADER_REDUCE.contains("fn main"));
        assert!(SHADER_EMBEDDING.contains("fn main"));
        assert!(SHADER_DISTANCE.contains("fn main"));
    }

    #[test]
    fn test_shader_sources_contain_compute_attribute() {
        assert!(SHADER_MATMUL.contains("@compute"));
        assert!(SHADER_SOFTMAX.contains("@compute"));
        assert!(SHADER_GELU.contains("@compute"));
        assert!(SHADER_LAYERNORM.contains("@compute"));
        assert!(SHADER_ATTENTION.contains("@compute"));
        assert!(SHADER_REDUCE.contains("@compute"));
        assert!(SHADER_EMBEDDING.contains("@compute"));
        assert!(SHADER_DISTANCE.contains("@compute"));
    }

    #[test]
    fn test_shader_sources_have_bindings() {
        // Every shader should declare at least one binding
        assert!(SHADER_MATMUL.contains("@binding"));
        assert!(SHADER_SOFTMAX.contains("@binding"));
        assert!(SHADER_GELU.contains("@binding"));
        assert!(SHADER_LAYERNORM.contains("@binding"));
        assert!(SHADER_ATTENTION.contains("@binding"));
        assert!(SHADER_REDUCE.contains("@binding"));
        assert!(SHADER_EMBEDDING.contains("@binding"));
        assert!(SHADER_DISTANCE.contains("@binding"));
    }

    #[test]
    fn test_shader_module_new() {
        let m = ShaderModule::new("test_shader", "fn main() {}", "main", 2, "A test shader");
        assert_eq!(m.name, "test_shader");
        assert_eq!(m.entry_point, "main");
        assert_eq!(m.num_bindings, 2);
        assert_eq!(m.source_len(), 12);
    }

    #[test]
    fn test_shader_cache_builtins() {
        let cache = ShaderCache::with_builtins();
        assert_eq!(cache.len(), 8);
        assert!(!cache.is_empty());

        assert!(cache.contains("matmul"));
        assert!(cache.contains("softmax"));
        assert!(cache.contains("gelu"));
        assert!(cache.contains("layernorm"));
        assert!(cache.contains("attention"));
        assert!(cache.contains("reduce"));
        assert!(cache.contains("embedding"));
        assert!(cache.contains("distance"));
    }

    #[test]
    fn test_shader_cache_get() {
        let cache = ShaderCache::with_builtins();
        let matmul = cache.get("matmul").unwrap();
        assert_eq!(matmul.name, "matmul");
        assert_eq!(matmul.entry_point, "main");
        assert_eq!(matmul.num_bindings, 4);

        let missing = cache.get("nonexistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_shader_cache_insert_custom() {
        let mut cache = ShaderCache::new();
        assert!(cache.is_empty());

        cache.insert(ShaderModule::new(
            "custom",
            "@compute @workgroup_size(64) fn main() {}",
            "main",
            1,
            "Custom shader",
        ));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("custom"));
    }

    #[test]
    fn test_shader_cache_remove() {
        let mut cache = ShaderCache::with_builtins();
        let count = cache.len();
        let removed = cache.remove("matmul");
        assert!(removed.is_some());
        assert_eq!(cache.len(), count - 1);
        assert!(!cache.contains("matmul"));
    }

    #[test]
    fn test_shader_cache_clear() {
        let mut cache = ShaderCache::with_builtins();
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_shader_cache_names() {
        let cache = ShaderCache::with_builtins();
        let mut names = cache.names();
        names.sort();
        assert!(names.contains(&"matmul".to_string()));
        assert!(names.contains(&"softmax".to_string()));
        assert!(names.contains(&"distance".to_string()));
    }

    #[test]
    fn test_shader_cache_total_source_bytes() {
        let cache = ShaderCache::with_builtins();
        let total = cache.total_source_bytes();
        // All shaders combined should be at least a few KB
        assert!(total > 1000, "Total source bytes too small: {total}");
    }

    #[test]
    fn test_shader_cache_default() {
        let cache = ShaderCache::default();
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn test_shader_cache_iter() {
        let cache = ShaderCache::with_builtins();
        let count = cache.iter().count();
        assert_eq!(count, 8);
    }

    #[test]
    fn test_matmul_shader_has_tile_size() {
        assert!(SHADER_MATMUL.contains("TILE"));
        assert!(SHADER_MATMUL.contains("workgroupBarrier"));
        assert!(SHADER_MATMUL.contains("var<workgroup>"));
    }

    #[test]
    fn test_softmax_shader_has_reduction() {
        assert!(SHADER_SOFTMAX.contains("shared_max"));
        assert!(SHADER_SOFTMAX.contains("shared_sum"));
        assert!(SHADER_SOFTMAX.contains("workgroupBarrier"));
    }

    #[test]
    fn test_gelu_shader_has_constants() {
        assert!(SHADER_GELU.contains("SQRT_2_OVER_PI"));
        assert!(SHADER_GELU.contains("0.044715"));
        assert!(SHADER_GELU.contains("tanh"));
    }

    #[test]
    fn test_layernorm_shader_has_eps() {
        assert!(SHADER_LAYERNORM.contains("eps"));
        assert!(SHADER_LAYERNORM.contains("inv_std"));
        assert!(SHADER_LAYERNORM.contains("gamma"));
        assert!(SHADER_LAYERNORM.contains("beta"));
    }

    #[test]
    fn test_attention_shader_has_scale() {
        assert!(SHADER_ATTENTION.contains("scale"));
        assert!(SHADER_ATTENTION.contains("exp"));
        assert!(SHADER_ATTENTION.contains("denom"));
    }

    #[test]
    fn test_reduce_shader_has_modes() {
        assert!(SHADER_REDUCE.contains("mode"));
        assert!(SHADER_REDUCE.contains("combine"));
        assert!(SHADER_REDUCE.contains("identity"));
    }

    #[test]
    fn test_embedding_shader_has_vocab_check() {
        assert!(SHADER_EMBEDDING.contains("vocab"));
        assert!(SHADER_EMBEDDING.contains("indices"));
    }

    #[test]
    fn test_distance_shader_has_symmetry() {
        // The distance shader should exploit symmetry
        assert!(SHADER_DISTANCE.contains("j < i"));
        assert!(SHADER_DISTANCE.contains("sqrt"));
    }

    #[test]
    fn test_all_shaders_have_workgroup_size() {
        let shaders = [
            SHADER_MATMUL,
            SHADER_SOFTMAX,
            SHADER_GELU,
            SHADER_LAYERNORM,
            SHADER_ATTENTION,
            SHADER_REDUCE,
            SHADER_EMBEDDING,
            SHADER_DISTANCE,
        ];
        for (i, shader) in shaders.iter().enumerate() {
            assert!(
                shader.contains("@workgroup_size"),
                "Shader {i} missing @workgroup_size"
            );
        }
    }

    #[test]
    fn test_all_shaders_have_group_binding() {
        let shaders = [
            SHADER_MATMUL,
            SHADER_SOFTMAX,
            SHADER_GELU,
            SHADER_LAYERNORM,
            SHADER_ATTENTION,
            SHADER_REDUCE,
            SHADER_EMBEDDING,
            SHADER_DISTANCE,
        ];
        for (i, shader) in shaders.iter().enumerate() {
            assert!(shader.contains("@group(0)"), "Shader {i} missing @group(0)");
        }
    }
}
