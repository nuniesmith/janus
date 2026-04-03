//! Sensory Gate — Multi-dimensional signal admission control
//!
//! Part of the Thalamus region
//! Component: gating
//!
//! The SensoryGate acts as the thalamic relay for incoming market signals,
//! deciding which signals to admit, attenuate, or block based on multiple
//! quality dimensions:
//!
//! - **SNR filtering**: Rejects signals below a signal-to-noise threshold
//! - **Staleness gating**: Rejects signals that are too old
//! - **Rate limiting**: Throttles excessive signal arrival per source
//! - **Priority-based admission**: Higher-priority signals bypass relaxed thresholds
//! - **Regime-aware gating**: Tightens or loosens thresholds based on market regime
//! - **Capacity management**: Limits total concurrent admitted signals
//! - **EMA-smoothed quality tracking**: Tracks per-source reliability over time
//!
//! Admitted signals receive a gate score (0..1) that downstream consumers
//! can use as an additional confidence weight.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Market regime for regime-aware threshold adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatingRegime {
    /// Low volatility — relax thresholds (admit more signals)
    Calm,
    /// Normal conditions
    Normal,
    /// Elevated volatility — tighten thresholds
    Volatile,
    /// Extreme stress — very tight filtering, only high-priority signals
    Crisis,
}

impl Default for GatingRegime {
    fn default() -> Self {
        GatingRegime::Normal
    }
}

impl GatingRegime {
    /// Multiplier applied to the SNR threshold.
    /// > 1.0 means tighter (higher bar), < 1.0 means looser.
    pub fn snr_multiplier(&self) -> f64 {
        match self {
            GatingRegime::Calm => 0.7,
            GatingRegime::Normal => 1.0,
            GatingRegime::Volatile => 1.3,
            GatingRegime::Crisis => 1.8,
        }
    }

    /// Multiplier applied to the staleness threshold.
    /// < 1.0 means stricter (shorter allowed age).
    pub fn staleness_multiplier(&self) -> f64 {
        match self {
            GatingRegime::Calm => 1.5,
            GatingRegime::Normal => 1.0,
            GatingRegime::Volatile => 0.7,
            GatingRegime::Crisis => 0.4,
        }
    }

    /// Minimum signal priority required to pass in this regime (0..10 scale)
    pub fn min_priority(&self) -> u8 {
        match self {
            GatingRegime::Calm => 0,
            GatingRegime::Normal => 0,
            GatingRegime::Volatile => 2,
            GatingRegime::Crisis => 5,
        }
    }
}

/// Signal priority level (0 = lowest, 10 = highest / critical)
pub type Priority = u8;

/// Configuration for the `SensoryGate`
#[derive(Debug, Clone)]
pub struct SensoryGateConfig {
    /// Minimum signal-to-noise ratio to admit a signal (before regime adjustment)
    pub snr_threshold: f64,
    /// Maximum signal age in milliseconds before it's considered stale
    pub max_staleness_ms: f64,
    /// Maximum signals per source within the rate window
    pub rate_limit_per_source: usize,
    /// Rate limit window duration in milliseconds
    pub rate_window_ms: f64,
    /// Maximum total admitted signals in the active set
    pub max_capacity: usize,
    /// EMA decay for per-source quality tracking (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum source quality (EMA) to continue admitting from that source
    pub min_source_quality: f64,
    /// Whether regime-aware gating is enabled
    pub regime_aware: bool,
    /// Sliding window size for gate history
    pub window_size: usize,
    /// Priority threshold below which signals are always blocked (0 = allow all priorities)
    pub global_min_priority: u8,
    /// Attenuation factor for signals near the SNR boundary (0..1).
    /// Signals with SNR between threshold*attenuation_zone and threshold
    /// receive a reduced gate score instead of full rejection.
    pub attenuation_zone: f64,
    /// Weight for SNR in the composite gate score
    pub score_weight_snr: f64,
    /// Weight for freshness in the composite gate score
    pub score_weight_freshness: f64,
    /// Weight for source quality in the composite gate score
    pub score_weight_source: f64,
    /// Weight for priority in the composite gate score
    pub score_weight_priority: f64,
}

impl Default for SensoryGateConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 1.5,
            max_staleness_ms: 5000.0,
            rate_limit_per_source: 100,
            rate_window_ms: 1000.0,
            max_capacity: 1000,
            ema_decay: 0.05,
            min_source_quality: 0.2,
            regime_aware: true,
            window_size: 200,
            global_min_priority: 0,
            attenuation_zone: 0.8,
            score_weight_snr: 0.35,
            score_weight_freshness: 0.25,
            score_weight_source: 0.20,
            score_weight_priority: 0.20,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// An incoming signal to be evaluated by the gate
#[derive(Debug, Clone)]
pub struct IncomingSignal {
    /// Unique identifier for the signal
    pub id: String,
    /// Source identifier (exchange, model, etc.)
    pub source_id: String,
    /// Signal-to-noise ratio estimate (higher = cleaner)
    pub snr: f64,
    /// Age of the signal in milliseconds
    pub age_ms: f64,
    /// Priority level (0 = lowest, 10 = highest)
    pub priority: Priority,
    /// Raw signal strength (arbitrary units, for downstream use)
    pub strength: f64,
    /// Timestamp of signal creation (for rate limiting), in milliseconds
    pub timestamp_ms: f64,
    /// Optional symbol/instrument the signal pertains to
    pub symbol: Option<String>,
}

/// The gate's decision on an incoming signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Signal is admitted with the associated gate score
    Admit,
    /// Signal is attenuated (partial admission with reduced score)
    Attenuate,
    /// Signal is blocked
    Block,
}

/// The reason a signal was blocked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    /// SNR too low
    LowSnr,
    /// Signal is too old
    Stale,
    /// Rate limit exceeded for this source
    RateLimited,
    /// Capacity full
    CapacityFull,
    /// Source quality too low
    LowSourceQuality,
    /// Priority too low for current regime
    InsufficientPriority,
    /// Global minimum priority not met
    BelowGlobalPriority,
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockReason::LowSnr => write!(f, "SNR below threshold"),
            BlockReason::Stale => write!(f, "signal too stale"),
            BlockReason::RateLimited => write!(f, "rate limited"),
            BlockReason::CapacityFull => write!(f, "capacity full"),
            BlockReason::LowSourceQuality => write!(f, "low source quality"),
            BlockReason::InsufficientPriority => write!(f, "priority too low for regime"),
            BlockReason::BelowGlobalPriority => write!(f, "below global minimum priority"),
        }
    }
}

/// Result of evaluating a signal against the gate
#[derive(Debug, Clone)]
pub struct GateResult {
    /// The gate's decision
    pub decision: GateDecision,
    /// Composite gate score (0..1). Only meaningful for Admit/Attenuate.
    pub gate_score: f64,
    /// Per-dimension scores
    pub snr_score: f64,
    pub freshness_score: f64,
    pub source_score: f64,
    pub priority_score: f64,
    /// Reason for blocking (None if admitted/attenuated)
    pub block_reason: Option<BlockReason>,
    /// Effective SNR threshold (after regime adjustment)
    pub effective_snr_threshold: f64,
    /// Effective staleness threshold (after regime adjustment)
    pub effective_staleness_ms: f64,
    /// Current regime at evaluation time
    pub regime: GatingRegime,
}

// ---------------------------------------------------------------------------
// Per-source state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SourceState {
    /// EMA of signal quality (composite of SNR and freshness)
    quality_ema: f64,
    quality_initialized: bool,
    /// Recent signal timestamps for rate limiting
    recent_timestamps: VecDeque<f64>,
    /// Total signals from this source
    total_signals: u64,
    /// Total admitted signals from this source
    admitted_signals: u64,
    /// Total blocked signals from this source
    blocked_signals: u64,
}

impl SourceState {
    fn new() -> Self {
        Self {
            quality_ema: 0.5,
            quality_initialized: false,
            recent_timestamps: VecDeque::new(),
            total_signals: 0,
            admitted_signals: 0,
            blocked_signals: 0,
        }
    }

    fn admission_rate(&self) -> f64 {
        if self.total_signals == 0 {
            return 0.0;
        }
        self.admitted_signals as f64 / self.total_signals as f64
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Per-source statistics exposed to callers
#[derive(Debug, Clone)]
pub struct SourceStats {
    /// Total signals evaluated from this source
    pub total_signals: u64,
    /// Total admitted signals
    pub admitted_signals: u64,
    /// Total blocked signals
    pub blocked_signals: u64,
    /// Current EMA quality score
    pub quality: f64,
    /// Admission rate (admitted / total)
    pub admission_rate: f64,
}

/// Running statistics for the gate
#[derive(Debug, Clone, Default)]
pub struct SensoryGateStats {
    /// Total signals evaluated
    pub total_evaluated: u64,
    /// Total signals admitted
    pub total_admitted: u64,
    /// Total signals attenuated
    pub total_attenuated: u64,
    /// Total signals blocked
    pub total_blocked: u64,
    /// Blocks by reason
    pub blocked_low_snr: u64,
    pub blocked_stale: u64,
    pub blocked_rate_limited: u64,
    pub blocked_capacity: u64,
    pub blocked_low_source_quality: u64,
    pub blocked_insufficient_priority: u64,
    pub blocked_below_global_priority: u64,
    /// Sum of gate scores (for mean calculation)
    pub sum_gate_score: f64,
    /// Sum of squared gate scores
    pub sum_sq_gate_score: f64,
    /// Peak gate score observed
    pub peak_gate_score: f64,
    /// Minimum gate score observed (among admitted/attenuated)
    pub min_gate_score: f64,
    /// Distinct sources seen
    pub distinct_sources: usize,
    /// Current active signal count
    pub current_active: usize,
}

impl SensoryGateStats {
    /// Overall admission rate
    pub fn admission_rate(&self) -> f64 {
        if self.total_evaluated == 0 {
            return 0.0;
        }
        (self.total_admitted + self.total_attenuated) as f64 / self.total_evaluated as f64
    }

    /// Mean gate score (among admitted/attenuated signals)
    pub fn mean_gate_score(&self) -> f64 {
        let total = self.total_admitted + self.total_attenuated;
        if total == 0 {
            return 0.0;
        }
        self.sum_gate_score / total as f64
    }

    /// Gate score variance
    pub fn gate_score_variance(&self) -> f64 {
        let total = self.total_admitted + self.total_attenuated;
        if total < 2 {
            return 0.0;
        }
        let n = total as f64;
        let mean = self.sum_gate_score / n;
        (self.sum_sq_gate_score / n - mean * mean).max(0.0)
    }

    /// Gate score standard deviation
    pub fn gate_score_std(&self) -> f64 {
        self.gate_score_variance().sqrt()
    }

    /// Block rate
    pub fn block_rate(&self) -> f64 {
        if self.total_evaluated == 0 {
            return 0.0;
        }
        self.total_blocked as f64 / self.total_evaluated as f64
    }

    /// Most common block reason
    pub fn dominant_block_reason(&self) -> Option<BlockReason> {
        let reasons = [
            (self.blocked_low_snr, BlockReason::LowSnr),
            (self.blocked_stale, BlockReason::Stale),
            (self.blocked_rate_limited, BlockReason::RateLimited),
            (self.blocked_capacity, BlockReason::CapacityFull),
            (
                self.blocked_low_source_quality,
                BlockReason::LowSourceQuality,
            ),
            (
                self.blocked_insufficient_priority,
                BlockReason::InsufficientPriority,
            ),
            (
                self.blocked_below_global_priority,
                BlockReason::BelowGlobalPriority,
            ),
        ];

        reasons
            .iter()
            .filter(|(count, _)| *count > 0)
            .max_by_key(|(count, _)| *count)
            .map(|(_, reason)| *reason)
    }
}

// ---------------------------------------------------------------------------
// Internal history record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GateRecord {
    decision: GateDecision,
    gate_score: f64,
    snr: f64,
    age_ms: f64,
    priority: u8,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Multi-dimensional sensory gate for incoming market signals
pub struct SensoryGate {
    config: SensoryGateConfig,

    /// Current gating regime
    current_regime: GatingRegime,

    /// Per-source state
    sources: HashMap<String, SourceState>,

    /// Current number of active (admitted) signals
    active_count: usize,

    /// Normalised score weights (sum to 1.0)
    score_weights: [f64; 4],

    /// Sliding window of recent gate results
    recent: VecDeque<GateRecord>,

    /// Running statistics
    stats: SensoryGateStats,
}

impl Default for SensoryGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SensoryGate {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(SensoryGateConfig::default()).unwrap()
    }

    /// Create with a specific configuration
    pub fn with_config(config: SensoryGateConfig) -> Result<Self> {
        // Validate
        if config.snr_threshold < 0.0 {
            return Err(Error::InvalidInput(
                "snr_threshold must be non-negative".into(),
            ));
        }
        if config.max_staleness_ms <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_ms must be > 0".into()));
        }
        if config.rate_limit_per_source == 0 {
            return Err(Error::InvalidInput(
                "rate_limit_per_source must be > 0".into(),
            ));
        }
        if config.rate_window_ms <= 0.0 {
            return Err(Error::InvalidInput("rate_window_ms must be > 0".into()));
        }
        if config.max_capacity == 0 {
            return Err(Error::InvalidInput("max_capacity must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.min_source_quality < 0.0 || config.min_source_quality > 1.0 {
            return Err(Error::InvalidInput(
                "min_source_quality must be in [0, 1]".into(),
            ));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.attenuation_zone < 0.0 || config.attenuation_zone > 1.0 {
            return Err(Error::InvalidInput(
                "attenuation_zone must be in [0, 1]".into(),
            ));
        }

        let w_snr = config.score_weight_snr;
        let w_fresh = config.score_weight_freshness;
        let w_src = config.score_weight_source;
        let w_pri = config.score_weight_priority;

        if w_snr < 0.0 || w_fresh < 0.0 || w_src < 0.0 || w_pri < 0.0 {
            return Err(Error::InvalidInput(
                "score weights must be non-negative".into(),
            ));
        }

        let w_sum = w_snr + w_fresh + w_src + w_pri;
        if w_sum < 1e-12 {
            return Err(Error::InvalidInput(
                "at least one score weight must be > 0".into(),
            ));
        }

        let score_weights = [w_snr / w_sum, w_fresh / w_sum, w_src / w_sum, w_pri / w_sum];

        Ok(Self {
            config,
            current_regime: GatingRegime::Normal,
            sources: HashMap::new(),
            active_count: 0,
            score_weights,
            recent: VecDeque::new(),
            stats: SensoryGateStats {
                min_gate_score: f64::MAX,
                ..Default::default()
            },
        })
    }

    /// Convenience factory — validates config and returns self
    pub fn process(config: SensoryGateConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Core evaluation
    // -----------------------------------------------------------------------

    /// Evaluate an incoming signal against all gate dimensions.
    /// Returns the gate decision, score, and detailed breakdown.
    pub fn evaluate(&mut self, signal: &IncomingSignal) -> Result<GateResult> {
        // Validate input
        if signal.snr < 0.0 {
            return Err(Error::InvalidInput("snr must be non-negative".into()));
        }
        if signal.age_ms < 0.0 {
            return Err(Error::InvalidInput("age_ms must be non-negative".into()));
        }
        if signal.priority > 10 {
            return Err(Error::InvalidInput("priority must be in [0, 10]".into()));
        }

        // Compute effective thresholds
        let (eff_snr, eff_staleness) = self.effective_thresholds();
        let eff_min_priority = self.effective_min_priority();

        // Ensure source state exists
        let source_state = self
            .sources
            .entry(signal.source_id.clone())
            .or_insert_with(SourceState::new);

        source_state.total_signals += 1;

        // --- Check blocking conditions in priority order ---

        // 1. Global minimum priority
        if signal.priority < self.config.global_min_priority {
            source_state.blocked_signals += 1;
            return Ok(self.make_blocked(
                BlockReason::BelowGlobalPriority,
                signal,
                eff_snr,
                eff_staleness,
            ));
        }

        // 2. Regime-based priority
        if self.config.regime_aware && signal.priority < eff_min_priority {
            source_state.blocked_signals += 1;
            return Ok(self.make_blocked(
                BlockReason::InsufficientPriority,
                signal,
                eff_snr,
                eff_staleness,
            ));
        }

        // 3. Staleness
        if signal.age_ms > eff_staleness {
            source_state.blocked_signals += 1;
            self.update_source_quality(&signal.source_id, 0.0);
            return Ok(self.make_blocked(BlockReason::Stale, signal, eff_snr, eff_staleness));
        }

        // 4. Source quality
        if source_state.quality_initialized
            && source_state.quality_ema < self.config.min_source_quality
        {
            source_state.blocked_signals += 1;
            return Ok(self.make_blocked(
                BlockReason::LowSourceQuality,
                signal,
                eff_snr,
                eff_staleness,
            ));
        }

        // 5. Rate limiting
        // Prune old timestamps
        let rate_cutoff = signal.timestamp_ms - self.config.rate_window_ms;
        {
            let src = self.sources.get_mut(&signal.source_id).unwrap();
            while let Some(&front) = src.recent_timestamps.front() {
                if front < rate_cutoff {
                    src.recent_timestamps.pop_front();
                } else {
                    break;
                }
            }
            if src.recent_timestamps.len() >= self.config.rate_limit_per_source {
                src.blocked_signals += 1;
                return Ok(self.make_blocked(
                    BlockReason::RateLimited,
                    signal,
                    eff_snr,
                    eff_staleness,
                ));
            }
            src.recent_timestamps.push_back(signal.timestamp_ms);
        }

        // 6. Capacity
        if self.active_count >= self.config.max_capacity {
            let src = self.sources.get_mut(&signal.source_id).unwrap();
            src.blocked_signals += 1;
            return Ok(self.make_blocked(
                BlockReason::CapacityFull,
                signal,
                eff_snr,
                eff_staleness,
            ));
        }

        // 7. SNR check (with attenuation zone)
        let attenuation_lower = eff_snr * self.config.attenuation_zone;

        if signal.snr < attenuation_lower {
            // Hard block — SNR too low even for attenuation
            let src = self.sources.get_mut(&signal.source_id).unwrap();
            src.blocked_signals += 1;
            self.update_source_quality(&signal.source_id, 0.1);
            return Ok(self.make_blocked(BlockReason::LowSnr, signal, eff_snr, eff_staleness));
        }

        // --- Signal passes all blocking checks ---

        // Compute per-dimension scores
        let snr_score = self.compute_snr_score(signal.snr, eff_snr);
        let freshness_score = self.compute_freshness_score(signal.age_ms, eff_staleness);
        let source_score = self
            .sources
            .get(&signal.source_id)
            .map(|s| {
                if s.quality_initialized {
                    s.quality_ema
                } else {
                    0.5
                }
            })
            .unwrap_or(0.5);
        let priority_score = signal.priority as f64 / 10.0;

        // Composite gate score
        let gate_score = (snr_score * self.score_weights[0]
            + freshness_score * self.score_weights[1]
            + source_score * self.score_weights[2]
            + priority_score * self.score_weights[3])
            .clamp(0.0, 1.0);

        // Determine decision
        let decision = if signal.snr < eff_snr {
            // In the attenuation zone
            GateDecision::Attenuate
        } else {
            GateDecision::Admit
        };

        // Update source quality
        let quality_input = (snr_score * 0.6 + freshness_score * 0.4).clamp(0.0, 1.0);
        self.update_source_quality(&signal.source_id, quality_input);

        // Update source admit count
        let src = self.sources.get_mut(&signal.source_id).unwrap();
        src.admitted_signals += 1;

        // Update active count
        self.active_count += 1;

        // Update stats
        self.stats.total_evaluated += 1;
        match decision {
            GateDecision::Admit => self.stats.total_admitted += 1,
            GateDecision::Attenuate => self.stats.total_attenuated += 1,
            GateDecision::Block => {} // unreachable here
        }
        self.stats.sum_gate_score += gate_score;
        self.stats.sum_sq_gate_score += gate_score * gate_score;
        if gate_score > self.stats.peak_gate_score {
            self.stats.peak_gate_score = gate_score;
        }
        if gate_score < self.stats.min_gate_score {
            self.stats.min_gate_score = gate_score;
        }
        self.stats.distinct_sources = self.sources.len();
        self.stats.current_active = self.active_count;

        // Window
        let record = GateRecord {
            decision,
            gate_score,
            snr: signal.snr,
            age_ms: signal.age_ms,
            priority: signal.priority,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(GateResult {
            decision,
            gate_score,
            snr_score,
            freshness_score,
            source_score,
            priority_score,
            block_reason: None,
            effective_snr_threshold: eff_snr,
            effective_staleness_ms: eff_staleness,
            regime: self.current_regime,
        })
    }

    /// Release a signal from the active set (when processing is complete)
    pub fn release(&mut self) {
        if self.active_count > 0 {
            self.active_count -= 1;
            self.stats.current_active = self.active_count;
        }
    }

    /// Release multiple signals
    pub fn release_n(&mut self, n: usize) {
        let to_release = n.min(self.active_count);
        self.active_count -= to_release;
        self.stats.current_active = self.active_count;
    }

    // -----------------------------------------------------------------------
    // Blocking helper
    // -----------------------------------------------------------------------

    fn make_blocked(
        &mut self,
        reason: BlockReason,
        signal: &IncomingSignal,
        eff_snr: f64,
        eff_staleness: f64,
    ) -> GateResult {
        self.stats.total_evaluated += 1;
        self.stats.total_blocked += 1;

        match reason {
            BlockReason::LowSnr => self.stats.blocked_low_snr += 1,
            BlockReason::Stale => self.stats.blocked_stale += 1,
            BlockReason::RateLimited => self.stats.blocked_rate_limited += 1,
            BlockReason::CapacityFull => self.stats.blocked_capacity += 1,
            BlockReason::LowSourceQuality => self.stats.blocked_low_source_quality += 1,
            BlockReason::InsufficientPriority => self.stats.blocked_insufficient_priority += 1,
            BlockReason::BelowGlobalPriority => self.stats.blocked_below_global_priority += 1,
        }

        self.stats.distinct_sources = self.sources.len();

        // Record in window
        let record = GateRecord {
            decision: GateDecision::Block,
            gate_score: 0.0,
            snr: signal.snr,
            age_ms: signal.age_ms,
            priority: signal.priority,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        GateResult {
            decision: GateDecision::Block,
            gate_score: 0.0,
            snr_score: 0.0,
            freshness_score: 0.0,
            source_score: 0.0,
            priority_score: 0.0,
            block_reason: Some(reason),
            effective_snr_threshold: eff_snr,
            effective_staleness_ms: eff_staleness,
            regime: self.current_regime,
        }
    }

    // -----------------------------------------------------------------------
    // Score computations
    // -----------------------------------------------------------------------

    /// SNR score: 0 at attenuation boundary, 1 at 2× threshold
    fn compute_snr_score(&self, snr: f64, threshold: f64) -> f64 {
        if threshold < 1e-12 {
            return 1.0;
        }
        // Linear ramp from attenuation_lower to 2×threshold
        let lower = threshold * self.config.attenuation_zone;
        let upper = threshold * 2.0;
        if snr >= upper {
            1.0
        } else if snr <= lower {
            0.0
        } else {
            (snr - lower) / (upper - lower)
        }
    }

    /// Freshness score: 1 at age=0, 0 at age=max_staleness
    fn compute_freshness_score(&self, age_ms: f64, max_staleness: f64) -> f64 {
        if max_staleness < 1e-12 {
            return 0.0;
        }
        (1.0 - age_ms / max_staleness).clamp(0.0, 1.0)
    }

    // -----------------------------------------------------------------------
    // Source quality EMA
    // -----------------------------------------------------------------------

    fn update_source_quality(&mut self, source_id: &str, quality: f64) {
        if let Some(src) = self.sources.get_mut(source_id) {
            if !src.quality_initialized {
                src.quality_ema = quality;
                src.quality_initialized = true;
            } else {
                src.quality_ema += self.config.ema_decay * (quality - src.quality_ema);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Effective thresholds
    // -----------------------------------------------------------------------

    fn effective_thresholds(&self) -> (f64, f64) {
        if self.config.regime_aware {
            let snr = self.config.snr_threshold * self.current_regime.snr_multiplier();
            let staleness =
                self.config.max_staleness_ms * self.current_regime.staleness_multiplier();
            (snr, staleness)
        } else {
            (self.config.snr_threshold, self.config.max_staleness_ms)
        }
    }

    fn effective_min_priority(&self) -> u8 {
        if self.config.regime_aware {
            self.current_regime
                .min_priority()
                .max(self.config.global_min_priority)
        } else {
            self.config.global_min_priority
        }
    }

    // -----------------------------------------------------------------------
    // Regime management
    // -----------------------------------------------------------------------

    /// Set the current gating regime
    pub fn set_regime(&mut self, regime: GatingRegime) {
        self.current_regime = regime;
    }

    /// Get the current gating regime
    pub fn regime(&self) -> GatingRegime {
        self.current_regime
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current number of active (admitted, not yet released) signals
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.config.max_capacity.saturating_sub(self.active_count)
    }

    /// Number of distinct sources tracked
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get statistics for a specific source
    pub fn source_stats(&self, source_id: &str) -> Option<SourceStats> {
        self.sources.get(source_id).map(|s| SourceStats {
            total_signals: s.total_signals,
            admitted_signals: s.admitted_signals,
            blocked_signals: s.blocked_signals,
            quality: if s.quality_initialized {
                s.quality_ema
            } else {
                0.5
            },
            admission_rate: s.admission_rate(),
        })
    }

    /// Get the source quality EMA for a source
    pub fn source_quality(&self, source_id: &str) -> Option<f64> {
        self.sources
            .get(source_id)
            .filter(|s| s.quality_initialized)
            .map(|s| s.quality_ema)
    }

    /// Access running statistics
    pub fn stats(&self) -> &SensoryGateStats {
        &self.stats
    }

    /// Access the configuration
    pub fn config(&self) -> &SensoryGateConfig {
        &self.config
    }

    /// Number of records in the sliding window
    pub fn window_count(&self) -> usize {
        self.recent.len()
    }

    /// Current effective SNR threshold
    pub fn effective_snr_threshold(&self) -> f64 {
        self.effective_thresholds().0
    }

    /// Current effective staleness threshold
    pub fn effective_staleness_ms(&self) -> f64 {
        self.effective_thresholds().1
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Windowed admission rate (fraction of recent signals that were admitted or attenuated)
    pub fn windowed_admission_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let admitted = self
            .recent
            .iter()
            .filter(|r| r.decision != GateDecision::Block)
            .count();
        admitted as f64 / self.recent.len() as f64
    }

    /// Windowed mean gate score (among admitted/attenuated signals)
    pub fn windowed_mean_gate_score(&self) -> f64 {
        let admitted: Vec<_> = self
            .recent
            .iter()
            .filter(|r| r.decision != GateDecision::Block)
            .collect();
        if admitted.is_empty() {
            return 0.0;
        }
        let sum: f64 = admitted.iter().map(|r| r.gate_score).sum();
        sum / admitted.len() as f64
    }

    /// Windowed mean SNR
    pub fn windowed_mean_snr(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.snr).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean age (freshness)
    pub fn windowed_mean_age_ms(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.age_ms).sum();
        sum / self.recent.len() as f64
    }

    /// Whether admission rate is declining (first half vs second half)
    pub fn is_admission_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;

        let first_admitted = self
            .recent
            .iter()
            .take(half)
            .filter(|r| r.decision != GateDecision::Block)
            .count() as f64
            / half as f64;

        let second_half_len = self.recent.len() - half;
        let second_admitted = self
            .recent
            .iter()
            .skip(half)
            .filter(|r| r.decision != GateDecision::Block)
            .count() as f64
            / second_half_len as f64;

        second_admitted < first_admitted * 0.8
    }

    /// Whether signal quality is improving (admission rate going up)
    pub fn is_quality_improving(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;

        let first_admitted = self
            .recent
            .iter()
            .take(half)
            .filter(|r| r.decision != GateDecision::Block)
            .count() as f64
            / half as f64;

        let second_half_len = self.recent.len() - half;
        let second_admitted = self
            .recent
            .iter()
            .skip(half)
            .filter(|r| r.decision != GateDecision::Block)
            .count() as f64
            / second_half_len as f64;

        second_admitted > first_admitted * 1.2
    }

    // -----------------------------------------------------------------------
    // Configuration adjustment
    // -----------------------------------------------------------------------

    /// Dynamically adjust the SNR threshold
    pub fn set_snr_threshold(&mut self, threshold: f64) -> Result<()> {
        if threshold < 0.0 {
            return Err(Error::InvalidInput(
                "snr_threshold must be non-negative".into(),
            ));
        }
        self.config.snr_threshold = threshold;
        Ok(())
    }

    /// Dynamically adjust the staleness threshold
    pub fn set_max_staleness_ms(&mut self, max_ms: f64) -> Result<()> {
        if max_ms <= 0.0 {
            return Err(Error::InvalidInput("max_staleness_ms must be > 0".into()));
        }
        self.config.max_staleness_ms = max_ms;
        Ok(())
    }

    /// Dynamically adjust the global minimum priority
    pub fn set_global_min_priority(&mut self, priority: u8) {
        self.config.global_min_priority = priority.min(10);
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all state, keeping configuration
    pub fn reset(&mut self) {
        self.current_regime = GatingRegime::Normal;
        self.sources.clear();
        self.active_count = 0;
        self.recent.clear();
        self.stats = SensoryGateStats {
            min_gate_score: f64::MAX,
            ..Default::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(source: &str, snr: f64, age_ms: f64, priority: u8) -> IncomingSignal {
        IncomingSignal {
            id: format!("sig-{}", snr as u32),
            source_id: source.to_string(),
            snr,
            age_ms,
            priority,
            strength: 1.0,
            timestamp_ms: 1000.0,
            symbol: None,
        }
    }

    fn make_signal_ts(
        source: &str,
        snr: f64,
        age_ms: f64,
        priority: u8,
        ts: f64,
    ) -> IncomingSignal {
        IncomingSignal {
            id: format!("sig-{}", snr as u32),
            source_id: source.to_string(),
            snr,
            age_ms,
            priority,
            strength: 1.0,
            timestamp_ms: ts,
            symbol: None,
        }
    }

    fn good_signal() -> IncomingSignal {
        make_signal("src_a", 3.0, 100.0, 5)
    }

    #[test]
    fn test_basic() {
        let instance = SensoryGate::new();
        assert_eq!(instance.active_count(), 0);
    }

    #[test]
    fn test_default_config() {
        let sg = SensoryGate::new();
        assert!((sg.config().snr_threshold - 1.5).abs() < 1e-10);
        assert_eq!(sg.regime(), GatingRegime::Normal);
    }

    // -- Admission tests --

    #[test]
    fn test_good_signal_admitted() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&good_signal()).unwrap();
        assert_eq!(result.decision, GateDecision::Admit);
        assert!(result.gate_score > 0.0);
        assert!(result.block_reason.is_none());
    }

    #[test]
    fn test_admission_increments_active_count() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        assert_eq!(sg.active_count(), 1);
    }

    #[test]
    fn test_gate_score_bounded() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&good_signal()).unwrap();
        assert!(result.gate_score >= 0.0);
        assert!(result.gate_score <= 1.0);
    }

    #[test]
    fn test_higher_snr_higher_score() {
        let mut sg = SensoryGate::new();
        let r1 = sg.evaluate(&make_signal("a", 2.0, 100.0, 5)).unwrap();
        sg.release();
        let r2 = sg.evaluate(&make_signal("b", 5.0, 100.0, 5)).unwrap();

        assert!(r2.gate_score >= r1.gate_score);
    }

    #[test]
    fn test_fresher_signal_higher_score() {
        let mut sg = SensoryGate::new();
        let r1 = sg.evaluate(&make_signal("a", 3.0, 3000.0, 5)).unwrap();
        sg.release();
        let r2 = sg.evaluate(&make_signal("b", 3.0, 100.0, 5)).unwrap();

        assert!(r2.gate_score > r1.gate_score);
    }

    #[test]
    fn test_higher_priority_higher_score() {
        let mut sg = SensoryGate::new();
        let r1 = sg.evaluate(&make_signal("a", 3.0, 100.0, 2)).unwrap();
        sg.release();
        let r2 = sg.evaluate(&make_signal("b", 3.0, 100.0, 9)).unwrap();

        assert!(r2.gate_score > r1.gate_score);
    }

    // -- SNR blocking tests --

    #[test]
    fn test_low_snr_blocked() {
        let mut sg = SensoryGate::new();
        // Default snr_threshold = 1.5, attenuation_zone = 0.8
        // Hard block below 1.5 * 0.8 = 1.2
        let result = sg.evaluate(&make_signal("src", 0.5, 100.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::LowSnr));
    }

    #[test]
    fn test_snr_in_attenuation_zone() {
        let mut sg = SensoryGate::new();
        // Attenuation zone: 1.2 to 1.5
        let result = sg.evaluate(&make_signal("src", 1.3, 100.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Attenuate);
    }

    #[test]
    fn test_snr_above_threshold_admitted() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&make_signal("src", 2.0, 100.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Admit);
    }

    // -- Staleness tests --

    #[test]
    fn test_stale_signal_blocked() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&make_signal("src", 3.0, 10000.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::Stale));
    }

    #[test]
    fn test_fresh_signal_admitted() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&make_signal("src", 3.0, 10.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Admit);
    }

    #[test]
    fn test_exactly_at_staleness_limit_blocked() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            max_staleness_ms: 1000.0,
            ..Default::default()
        })
        .unwrap();

        let result = sg.evaluate(&make_signal("src", 3.0, 1001.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
    }

    // -- Rate limiting tests --

    #[test]
    fn test_rate_limiting() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            rate_limit_per_source: 3,
            rate_window_ms: 1000.0,
            ..Default::default()
        })
        .unwrap();

        for i in 0..3 {
            let result = sg
                .evaluate(&make_signal_ts("src_a", 3.0, 100.0, 5, 100.0 + i as f64))
                .unwrap();
            assert_ne!(result.decision, GateDecision::Block);
        }

        // 4th signal should be rate limited
        let result = sg
            .evaluate(&make_signal_ts("src_a", 3.0, 100.0, 5, 103.0))
            .unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::RateLimited));
    }

    #[test]
    fn test_rate_limit_per_source_independent() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            rate_limit_per_source: 2,
            rate_window_ms: 1000.0,
            ..Default::default()
        })
        .unwrap();

        // Source A: 2 signals (maxed out)
        sg.evaluate(&make_signal_ts("a", 3.0, 100.0, 5, 100.0))
            .unwrap();
        sg.evaluate(&make_signal_ts("a", 3.0, 100.0, 5, 101.0))
            .unwrap();

        // Source B should still work
        let result = sg
            .evaluate(&make_signal_ts("b", 3.0, 100.0, 5, 102.0))
            .unwrap();
        assert_ne!(result.decision, GateDecision::Block);
    }

    #[test]
    fn test_rate_limit_window_expiry() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            rate_limit_per_source: 2,
            rate_window_ms: 100.0,
            ..Default::default()
        })
        .unwrap();

        // Fill rate limit
        sg.evaluate(&make_signal_ts("src", 3.0, 10.0, 5, 100.0))
            .unwrap();
        sg.evaluate(&make_signal_ts("src", 3.0, 10.0, 5, 101.0))
            .unwrap();

        // After the window expires, should be allowed again
        let result = sg
            .evaluate(&make_signal_ts("src", 3.0, 10.0, 5, 250.0))
            .unwrap();
        assert_ne!(result.decision, GateDecision::Block);
    }

    // -- Capacity tests --

    #[test]
    fn test_capacity_blocking() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            max_capacity: 2,
            ..Default::default()
        })
        .unwrap();

        sg.evaluate(&make_signal("a", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("b", 3.0, 100.0, 5)).unwrap();

        let result = sg.evaluate(&make_signal("c", 3.0, 100.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::CapacityFull));
    }

    #[test]
    fn test_release_frees_capacity() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            max_capacity: 2,
            ..Default::default()
        })
        .unwrap();

        sg.evaluate(&make_signal("a", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("b", 3.0, 100.0, 5)).unwrap();
        sg.release();

        let result = sg.evaluate(&make_signal("c", 3.0, 100.0, 5)).unwrap();
        assert_ne!(result.decision, GateDecision::Block);
    }

    #[test]
    fn test_release_n() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            max_capacity: 5,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            sg.evaluate(&good_signal()).unwrap();
        }
        assert_eq!(sg.active_count(), 5);

        sg.release_n(3);
        assert_eq!(sg.active_count(), 2);
    }

    #[test]
    fn test_release_n_doesnt_underflow() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        sg.release_n(100);
        assert_eq!(sg.active_count(), 0);
    }

    #[test]
    fn test_remaining_capacity() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            max_capacity: 10,
            ..Default::default()
        })
        .unwrap();

        sg.evaluate(&good_signal()).unwrap();
        assert_eq!(sg.remaining_capacity(), 9);
    }

    // -- Priority tests --

    #[test]
    fn test_global_min_priority_blocks() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            global_min_priority: 5,
            ..Default::default()
        })
        .unwrap();

        let result = sg.evaluate(&make_signal("src", 3.0, 100.0, 3)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::BelowGlobalPriority));
    }

    #[test]
    fn test_global_min_priority_allows() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            global_min_priority: 3,
            ..Default::default()
        })
        .unwrap();

        let result = sg.evaluate(&make_signal("src", 3.0, 100.0, 5)).unwrap();
        assert_ne!(result.decision, GateDecision::Block);
    }

    #[test]
    fn test_regime_priority_blocking() {
        let mut sg = SensoryGate::new();
        sg.set_regime(GatingRegime::Crisis); // min_priority = 5

        let result = sg.evaluate(&make_signal("src", 5.0, 50.0, 3)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::InsufficientPriority));
    }

    #[test]
    fn test_regime_priority_allows_high_priority() {
        let mut sg = SensoryGate::new();
        sg.set_regime(GatingRegime::Crisis);

        let result = sg.evaluate(&make_signal("src", 5.0, 50.0, 8)).unwrap();
        // Should not be blocked for priority, might be for SNR if threshold is too high
        assert_ne!(result.block_reason, Some(BlockReason::InsufficientPriority));
    }

    // -- Regime tests --

    #[test]
    fn test_regime_affects_snr_threshold() {
        let mut sg = SensoryGate::new();

        let normal_threshold = sg.effective_snr_threshold();

        sg.set_regime(GatingRegime::Crisis);
        let crisis_threshold = sg.effective_snr_threshold();

        sg.set_regime(GatingRegime::Calm);
        let calm_threshold = sg.effective_snr_threshold();

        assert!(crisis_threshold > normal_threshold);
        assert!(calm_threshold < normal_threshold);
    }

    #[test]
    fn test_regime_affects_staleness_threshold() {
        let mut sg = SensoryGate::new();

        let normal_staleness = sg.effective_staleness_ms();

        sg.set_regime(GatingRegime::Crisis);
        let crisis_staleness = sg.effective_staleness_ms();

        sg.set_regime(GatingRegime::Calm);
        let calm_staleness = sg.effective_staleness_ms();

        assert!(crisis_staleness < normal_staleness);
        assert!(calm_staleness > normal_staleness);
    }

    #[test]
    fn test_regime_disabled() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            regime_aware: false,
            snr_threshold: 1.5,
            ..Default::default()
        })
        .unwrap();

        sg.set_regime(GatingRegime::Crisis);
        let threshold = sg.effective_snr_threshold();
        assert!((threshold - 1.5).abs() < 1e-10); // Not affected by regime
    }

    // -- Source quality tests --

    #[test]
    fn test_source_quality_tracking() {
        let mut sg = SensoryGate::new();

        // Good signals → quality goes up
        for _ in 0..10 {
            sg.evaluate(&make_signal("src_good", 5.0, 10.0, 5)).unwrap();
            sg.release();
        }

        let quality = sg.source_quality("src_good").unwrap();
        assert!(quality > 0.5);
    }

    #[test]
    fn test_source_quality_unknown_source() {
        let sg = SensoryGate::new();
        assert!(sg.source_quality("nonexistent").is_none());
    }

    #[test]
    fn test_source_stats() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("src_a", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("src_a", 0.1, 100.0, 5)).unwrap(); // blocked

        let stats = sg.source_stats("src_a").unwrap();
        assert_eq!(stats.total_signals, 2);
        assert_eq!(stats.admitted_signals, 1);
        assert_eq!(stats.blocked_signals, 1);
    }

    #[test]
    fn test_source_stats_unknown_source() {
        let sg = SensoryGate::new();
        assert!(sg.source_stats("nonexistent").is_none());
    }

    #[test]
    fn test_source_count() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("b", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("a", 3.0, 100.0, 5)).unwrap();

        assert_eq!(sg.source_count(), 2);
    }

    // -- Low source quality blocking --

    #[test]
    fn test_low_source_quality_blocks() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            min_source_quality: 0.3,
            ema_decay: 0.9, // fast adaptation
            ..Default::default()
        })
        .unwrap();

        // Send many stale signals to degrade source quality
        for _ in 0..20 {
            sg.evaluate(&make_signal("bad_src", 3.0, 10000.0, 5))
                .unwrap(); // stale → quality ema → 0
        }

        // Now a good signal from the same source should be blocked
        let result = sg.evaluate(&make_signal("bad_src", 3.0, 100.0, 5)).unwrap();
        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.block_reason, Some(BlockReason::LowSourceQuality));
    }

    // -- Stats tests --

    #[test]
    fn test_stats_tracking() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 3.0, 100.0, 5)).unwrap(); // admit
        sg.evaluate(&make_signal("b", 0.1, 100.0, 5)).unwrap(); // block
        sg.evaluate(&make_signal("c", 1.3, 100.0, 5)).unwrap(); // attenuate

        assert_eq!(sg.stats().total_evaluated, 3);
        assert_eq!(sg.stats().total_admitted, 1);
        assert_eq!(sg.stats().total_blocked, 1);
        assert_eq!(sg.stats().total_attenuated, 1);
    }

    #[test]
    fn test_stats_admission_rate() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        sg.evaluate(&good_signal()).unwrap();
        sg.evaluate(&make_signal("x", 0.1, 100.0, 5)).unwrap(); // blocked

        let rate = sg.stats().admission_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_block_rate() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        sg.evaluate(&make_signal("x", 0.1, 100.0, 5)).unwrap();

        assert!((sg.stats().block_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean_gate_score() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        assert!(sg.stats().mean_gate_score() > 0.0);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = SensoryGateStats::default();
        assert_eq!(stats.total_evaluated, 0);
        assert_eq!(stats.admission_rate(), 0.0);
        assert_eq!(stats.mean_gate_score(), 0.0);
        assert_eq!(stats.block_rate(), 0.0);
        assert_eq!(stats.gate_score_std(), 0.0);
    }

    #[test]
    fn test_stats_dominant_block_reason() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 0.1, 100.0, 5)).unwrap(); // LowSnr
        sg.evaluate(&make_signal("b", 0.1, 100.0, 5)).unwrap(); // LowSnr
        sg.evaluate(&make_signal("c", 3.0, 10000.0, 5)).unwrap(); // Stale

        assert_eq!(
            sg.stats().dominant_block_reason(),
            Some(BlockReason::LowSnr)
        );
    }

    #[test]
    fn test_stats_dominant_block_reason_none() {
        let stats = SensoryGateStats::default();
        assert!(stats.dominant_block_reason().is_none());
    }

    #[test]
    fn test_stats_gate_score_variance_constant() {
        let mut sg = SensoryGate::new();
        // All identical signals
        for _ in 0..10 {
            sg.evaluate(&good_signal()).unwrap();
            sg.release();
        }
        // Variance should be very low (EMA may introduce slight drift)
        assert!(sg.stats().gate_score_variance() < 0.01);
    }

    #[test]
    fn test_stats_block_reasons_tracked() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 0.1, 100.0, 5)).unwrap(); // LowSnr
        sg.evaluate(&make_signal("b", 3.0, 10000.0, 5)).unwrap(); // Stale

        assert_eq!(sg.stats().blocked_low_snr, 1);
        assert_eq!(sg.stats().blocked_stale, 1);
    }

    // -- Window tests --

    #[test]
    fn test_windowed_admission_rate() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        sg.evaluate(&make_signal("x", 0.1, 100.0, 5)).unwrap();

        let rate = sg.windowed_admission_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_gate_score() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        assert!(sg.windowed_mean_gate_score() > 0.0);
    }

    #[test]
    fn test_windowed_mean_snr() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 2.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("b", 4.0, 100.0, 5)).unwrap();

        assert!((sg.windowed_mean_snr() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_age() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&make_signal("a", 3.0, 200.0, 5)).unwrap();
        sg.evaluate(&make_signal("b", 3.0, 400.0, 5)).unwrap();

        assert!((sg.windowed_mean_age_ms() - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_empty() {
        let sg = SensoryGate::new();
        assert_eq!(sg.windowed_admission_rate(), 0.0);
        assert_eq!(sg.windowed_mean_gate_score(), 0.0);
        assert_eq!(sg.windowed_mean_snr(), 0.0);
        assert_eq!(sg.windowed_mean_age_ms(), 0.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            window_size: 3,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            sg.evaluate(&good_signal()).unwrap();
            sg.release();
        }

        assert_eq!(sg.window_count(), 3);
    }

    // -- Admission declining --

    #[test]
    fn test_is_admission_declining() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        // First half: good signals
        for _ in 0..10 {
            sg.evaluate(&make_signal("a", 5.0, 10.0, 5)).unwrap();
            sg.release();
        }
        // Second half: bad signals (blocked)
        for _ in 0..10 {
            sg.evaluate(&make_signal("b", 0.1, 100.0, 5)).unwrap();
        }

        assert!(sg.is_admission_declining());
    }

    #[test]
    fn test_not_declining_consistent() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            sg.evaluate(&good_signal()).unwrap();
            sg.release();
        }

        assert!(!sg.is_admission_declining());
    }

    #[test]
    fn test_not_declining_insufficient_data() {
        let mut sg = SensoryGate::new();
        sg.evaluate(&good_signal()).unwrap();
        assert!(!sg.is_admission_declining());
    }

    #[test]
    fn test_is_quality_improving() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        // First half: bad signals
        for _ in 0..10 {
            sg.evaluate(&make_signal("a", 0.1, 100.0, 5)).unwrap();
        }
        // Second half: good signals
        for _ in 0..10 {
            sg.evaluate(&make_signal("b", 5.0, 10.0, 5)).unwrap();
            sg.release();
        }

        assert!(sg.is_quality_improving());
    }

    // -- Dynamic configuration --

    #[test]
    fn test_set_snr_threshold() {
        let mut sg = SensoryGate::new();
        sg.set_snr_threshold(5.0).unwrap();
        assert!((sg.config().snr_threshold - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_snr_threshold_negative_rejected() {
        let mut sg = SensoryGate::new();
        assert!(sg.set_snr_threshold(-1.0).is_err());
    }

    #[test]
    fn test_set_max_staleness() {
        let mut sg = SensoryGate::new();
        sg.set_max_staleness_ms(1000.0).unwrap();
        assert!((sg.config().max_staleness_ms - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_max_staleness_zero_rejected() {
        let mut sg = SensoryGate::new();
        assert!(sg.set_max_staleness_ms(0.0).is_err());
    }

    #[test]
    fn test_set_global_min_priority() {
        let mut sg = SensoryGate::new();
        sg.set_global_min_priority(5);
        assert_eq!(sg.config().global_min_priority, 5);
    }

    #[test]
    fn test_set_global_min_priority_clamped() {
        let mut sg = SensoryGate::new();
        sg.set_global_min_priority(15);
        assert_eq!(sg.config().global_min_priority, 10);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut sg = SensoryGate::new();
        sg.set_regime(GatingRegime::Crisis);
        for _ in 0..10 {
            sg.evaluate(&good_signal()).unwrap();
        }

        sg.reset();

        assert_eq!(sg.active_count(), 0);
        assert_eq!(sg.source_count(), 0);
        assert_eq!(sg.window_count(), 0);
        assert_eq!(sg.regime(), GatingRegime::Normal);
        assert_eq!(sg.stats().total_evaluated, 0);
    }

    // -- Input validation --

    #[test]
    fn test_invalid_negative_snr() {
        let mut sg = SensoryGate::new();
        let signal = IncomingSignal {
            snr: -1.0,
            ..good_signal()
        };
        assert!(sg.evaluate(&signal).is_err());
    }

    #[test]
    fn test_invalid_negative_age() {
        let mut sg = SensoryGate::new();
        let signal = IncomingSignal {
            age_ms: -10.0,
            ..good_signal()
        };
        assert!(sg.evaluate(&signal).is_err());
    }

    #[test]
    fn test_invalid_priority_above_10() {
        let mut sg = SensoryGate::new();
        let signal = IncomingSignal {
            priority: 11,
            ..good_signal()
        };
        assert!(sg.evaluate(&signal).is_err());
    }

    // -- Config validation --

    #[test]
    fn test_invalid_config_negative_snr_threshold() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            snr_threshold: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_staleness() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            max_staleness_ms: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_rate_limit() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            rate_limit_per_source: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_rate_window() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            rate_window_ms: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_capacity() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            max_capacity: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let r1 = SensoryGate::with_config(SensoryGateConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = SensoryGate::with_config(SensoryGateConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_source_quality() {
        let r1 = SensoryGate::with_config(SensoryGateConfig {
            min_source_quality: -0.1,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = SensoryGate::with_config(SensoryGateConfig {
            min_source_quality: 1.5,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_attenuation_zone() {
        let r1 = SensoryGate::with_config(SensoryGateConfig {
            attenuation_zone: -0.1,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = SensoryGate::with_config(SensoryGateConfig {
            attenuation_zone: 1.1,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_negative_score_weight() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            score_weight_snr: -0.1,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_all_zero_weights() {
        let result = SensoryGate::with_config(SensoryGateConfig {
            score_weight_snr: 0.0,
            score_weight_freshness: 0.0,
            score_weight_source: 0.0,
            score_weight_priority: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Process convenience --

    #[test]
    fn test_process_returns_instance() {
        let sg = SensoryGate::process(SensoryGateConfig::default());
        assert!(sg.is_ok());
    }

    #[test]
    fn test_process_rejects_bad_config() {
        let result = SensoryGate::process(SensoryGateConfig {
            max_capacity: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- GateResult structure --

    #[test]
    fn test_gate_result_admitted_structure() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&good_signal()).unwrap();

        assert_eq!(result.decision, GateDecision::Admit);
        assert!(result.gate_score > 0.0);
        assert!(result.snr_score >= 0.0 && result.snr_score <= 1.0);
        assert!(result.freshness_score >= 0.0 && result.freshness_score <= 1.0);
        assert!(result.source_score >= 0.0 && result.source_score <= 1.0);
        assert!(result.priority_score >= 0.0 && result.priority_score <= 1.0);
        assert!(result.block_reason.is_none());
        assert!(result.effective_snr_threshold > 0.0);
        assert!(result.effective_staleness_ms > 0.0);
        assert_eq!(result.regime, GatingRegime::Normal);
    }

    #[test]
    fn test_gate_result_blocked_structure() {
        let mut sg = SensoryGate::new();
        let result = sg.evaluate(&make_signal("src", 0.1, 100.0, 5)).unwrap();

        assert_eq!(result.decision, GateDecision::Block);
        assert_eq!(result.gate_score, 0.0);
        assert!(result.block_reason.is_some());
    }

    // -- Regime methods --

    #[test]
    fn test_regime_snr_multipliers() {
        assert!(GatingRegime::Calm.snr_multiplier() < 1.0);
        assert!((GatingRegime::Normal.snr_multiplier() - 1.0).abs() < 1e-10);
        assert!(GatingRegime::Volatile.snr_multiplier() > 1.0);
        assert!(GatingRegime::Crisis.snr_multiplier() > GatingRegime::Volatile.snr_multiplier());
    }

    #[test]
    fn test_regime_staleness_multipliers() {
        assert!(GatingRegime::Calm.staleness_multiplier() > 1.0);
        assert!((GatingRegime::Normal.staleness_multiplier() - 1.0).abs() < 1e-10);
        assert!(
            GatingRegime::Crisis.staleness_multiplier()
                < GatingRegime::Normal.staleness_multiplier()
        );
    }

    #[test]
    fn test_regime_min_priorities() {
        assert_eq!(GatingRegime::Calm.min_priority(), 0);
        assert_eq!(GatingRegime::Normal.min_priority(), 0);
        assert!(GatingRegime::Volatile.min_priority() > 0);
        assert!(GatingRegime::Crisis.min_priority() > GatingRegime::Volatile.min_priority());
    }

    // -- SNR score computation --

    #[test]
    fn test_snr_score_at_double_threshold() {
        let sg = SensoryGate::new();
        let score = sg.compute_snr_score(3.0, 1.5); // 2× threshold
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_score_above_double_threshold() {
        let sg = SensoryGate::new();
        let score = sg.compute_snr_score(10.0, 1.5);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_score_at_attenuation_lower() {
        let sg = SensoryGate::new();
        let lower = 1.5 * 0.8; // 1.2
        let score = sg.compute_snr_score(lower, 1.5);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_snr_score_midpoint() {
        let sg = SensoryGate::new();
        // lower = 1.2, upper = 3.0, midpoint = 2.1
        let score = sg.compute_snr_score(2.1, 1.5);
        assert!(score > 0.4 && score < 0.6);
    }

    // -- Freshness score --

    #[test]
    fn test_freshness_score_zero_age() {
        let sg = SensoryGate::new();
        let score = sg.compute_freshness_score(0.0, 5000.0);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_freshness_score_at_limit() {
        let sg = SensoryGate::new();
        let score = sg.compute_freshness_score(5000.0, 5000.0);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_freshness_score_midpoint() {
        let sg = SensoryGate::new();
        let score = sg.compute_freshness_score(2500.0, 5000.0);
        assert!((score - 0.5).abs() < 1e-10);
    }

    // -- BlockReason display --

    #[test]
    fn test_block_reason_display() {
        assert_eq!(format!("{}", BlockReason::LowSnr), "SNR below threshold");
        assert_eq!(format!("{}", BlockReason::Stale), "signal too stale");
        assert_eq!(format!("{}", BlockReason::RateLimited), "rate limited");
    }

    // -- Zero SNR threshold --

    #[test]
    fn test_zero_snr_threshold_admits_all() {
        let mut sg = SensoryGate::with_config(SensoryGateConfig {
            snr_threshold: 0.0,
            ..Default::default()
        })
        .unwrap();

        let result = sg.evaluate(&make_signal("src", 0.001, 100.0, 5)).unwrap();
        assert_ne!(result.decision, GateDecision::Block);
    }

    // -- Multiple sources --

    #[test]
    fn test_multiple_sources_tracked_independently() {
        let mut sg = SensoryGate::new();

        sg.evaluate(&make_signal("src_a", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("src_b", 3.0, 100.0, 5)).unwrap();
        sg.evaluate(&make_signal("src_a", 3.0, 100.0, 5)).unwrap();

        let stats_a = sg.source_stats("src_a").unwrap();
        let stats_b = sg.source_stats("src_b").unwrap();

        assert_eq!(stats_a.total_signals, 2);
        assert_eq!(stats_b.total_signals, 1);
    }
}
