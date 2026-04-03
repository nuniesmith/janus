//! Memory consolidation engine for experience replay and long-term storage
//!
//! Part of the Cortex region
//! Component: memory
//!
//! Consolidates short-term trading experiences into long-term memory via
//! importance-weighted experience replay. Implements exponential forgetting
//! curves so that older, less-relevant memories decay while high-impact
//! experiences are retained. Provides a prioritised replay buffer that
//! upstream learners can sample from to improve training stability.
//!
//! Key features:
//! - Prioritised experience replay buffer with configurable capacity
//! - Importance scoring based on TD-error magnitude, reward magnitude, novelty
//! - Exponential forgetting curves with configurable half-life
//! - Consolidation cycles that promote high-importance short-term memories
//! - Sampling with priority-proportional probability (rank-based)
//! - EMA-smoothed tracking of buffer utilisation, replay quality, and churn
//! - Sliding window of recent consolidation cycle summaries
//! - Running statistics for diagnostics and audit

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the consolidation engine
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Maximum capacity of the short-term buffer
    pub short_term_capacity: usize,
    /// Maximum capacity of the long-term buffer
    pub long_term_capacity: usize,
    /// Half-life for the forgetting curve (in consolidation cycles)
    pub forgetting_half_life: f64,
    /// Minimum importance score to promote from short-term to long-term
    pub promotion_threshold: f64,
    /// Fraction of short-term buffer to evaluate per consolidation cycle
    pub consolidation_fraction: f64,
    /// Priority exponent α for prioritised replay (0 = uniform, 1 = full priority)
    pub priority_exponent: f64,
    /// Importance-sampling correction exponent β (0 = no correction, 1 = full)
    pub is_correction_exponent: f64,
    /// Novelty bonus weight in importance scoring
    pub novelty_weight: f64,
    /// TD-error weight in importance scoring
    pub td_error_weight: f64,
    /// Reward magnitude weight in importance scoring
    pub reward_weight: f64,
    /// Small constant ε added to priorities to prevent zero-probability
    pub priority_epsilon: f64,
    /// EMA decay factor for statistics tracking (0 < α < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent consolidation summaries
    pub window_size: usize,
    /// Maximum age (in cycles) before a short-term memory is auto-evicted
    pub max_short_term_age: u64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            short_term_capacity: 10_000,
            long_term_capacity: 100_000,
            forgetting_half_life: 50.0,
            promotion_threshold: 0.3,
            consolidation_fraction: 0.20,
            priority_exponent: 0.6,
            is_correction_exponent: 0.4,
            novelty_weight: 0.25,
            td_error_weight: 0.50,
            reward_weight: 0.25,
            priority_epsilon: 1e-6,
            ema_decay: 0.1,
            window_size: 100,
            max_short_term_age: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Experience / Memory types
// ---------------------------------------------------------------------------

/// A single experience (transition) stored in the replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    /// Unique identifier
    pub id: u64,
    /// State representation (flattened feature vector)
    pub state: Vec<f64>,
    /// Action taken (flattened)
    pub action: Vec<f64>,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: Vec<f64>,
    /// Whether this transition ended an episode
    pub terminal: bool,
    /// Temporal-difference error magnitude (updated by learner)
    pub td_error: f64,
    /// Novelty score (higher = more unlike existing memories)
    pub novelty: f64,
    /// Computed importance score
    pub importance: f64,
    /// Priority for replay sampling
    pub priority: f64,
    /// Age in consolidation cycles since this experience was stored
    pub age: u64,
    /// Current memory strength (decays over time via forgetting curve)
    pub strength: f64,
    /// Number of times this experience has been replayed
    pub replay_count: u64,
    /// Whether this experience has been promoted to long-term memory
    pub promoted: bool,
    /// Optional metadata tag (e.g. strategy name, regime label)
    pub tag: String,
}

impl Experience {
    /// Create a new experience with basic fields, computing importance
    pub fn new(
        id: u64,
        state: Vec<f64>,
        action: Vec<f64>,
        reward: f64,
        next_state: Vec<f64>,
        terminal: bool,
    ) -> Self {
        Self {
            id,
            state,
            action,
            reward,
            next_state,
            terminal,
            td_error: 0.0,
            novelty: 0.0,
            importance: 0.0,
            priority: 1.0,
            age: 0,
            strength: 1.0,
            replay_count: 0,
            promoted: false,
            tag: String::new(),
        }
    }

    /// Create with tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = tag.into();
        self
    }

    /// Create with TD-error
    pub fn with_td_error(mut self, td_error: f64) -> Self {
        self.td_error = td_error.abs();
        self
    }

    /// Create with novelty
    pub fn with_novelty(mut self, novelty: f64) -> Self {
        self.novelty = novelty.clamp(0.0, 1.0);
        self
    }
}

/// Indices and weights for a sampled mini-batch
#[derive(Debug, Clone)]
pub struct ReplayBatch {
    /// Indices into the buffer from which experiences were drawn
    pub indices: Vec<usize>,
    /// Importance-sampling weights (for bias correction)
    pub is_weights: Vec<f64>,
    /// The sampled experiences (cloned)
    pub experiences: Vec<Experience>,
}

/// Summary of a single consolidation cycle
#[derive(Debug, Clone)]
pub struct ConsolidationSummary {
    /// Number of short-term memories evaluated
    pub evaluated: usize,
    /// Number promoted to long-term
    pub promoted: usize,
    /// Number evicted from short-term (age or capacity)
    pub evicted: usize,
    /// Number decayed below threshold and removed from long-term
    pub forgotten: usize,
    /// Average importance of promoted memories
    pub avg_promoted_importance: f64,
    /// Average strength of long-term memories after decay
    pub avg_long_term_strength: f64,
    /// Short-term buffer utilisation (fraction of capacity)
    pub short_term_utilisation: f64,
    /// Long-term buffer utilisation
    pub long_term_utilisation: f64,
    /// Cycle number
    pub cycle: u64,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the consolidation engine
#[derive(Debug, Clone)]
pub struct ConsolidationStats {
    /// Total consolidation cycles performed
    pub total_cycles: u64,
    /// Total experiences stored (lifetime)
    pub total_stored: u64,
    /// Total promotions from short-term to long-term
    pub total_promoted: u64,
    /// Total evictions
    pub total_evicted: u64,
    /// Total experiences forgotten (decayed below threshold)
    pub total_forgotten: u64,
    /// Total replay samples drawn
    pub total_sampled: u64,
    /// EMA of promotion rate (promoted / evaluated per cycle)
    pub ema_promotion_rate: f64,
    /// EMA of average importance at promotion
    pub ema_promoted_importance: f64,
    /// EMA of average long-term strength
    pub ema_long_term_strength: f64,
    /// EMA of short-term utilisation
    pub ema_short_term_util: f64,
    /// EMA of long-term utilisation
    pub ema_long_term_util: f64,
    /// Best (highest) importance score seen at promotion
    pub best_importance: f64,
    /// Average TD-error across all stored experiences (EMA)
    pub ema_avg_td_error: f64,
}

impl Default for ConsolidationStats {
    fn default() -> Self {
        Self {
            total_cycles: 0,
            total_stored: 0,
            total_promoted: 0,
            total_evicted: 0,
            total_forgotten: 0,
            total_sampled: 0,
            ema_promotion_rate: 0.0,
            ema_promoted_importance: 0.0,
            ema_long_term_strength: 1.0,
            ema_short_term_util: 0.0,
            ema_long_term_util: 0.0,
            best_importance: 0.0,
            ema_avg_td_error: 0.0,
        }
    }
}

impl ConsolidationStats {
    /// Average promotions per cycle
    pub fn avg_promotions_per_cycle(&self) -> f64 {
        if self.total_cycles == 0 {
            return 0.0;
        }
        self.total_promoted as f64 / self.total_cycles as f64
    }

    /// Average evictions per cycle
    pub fn avg_evictions_per_cycle(&self) -> f64 {
        if self.total_cycles == 0 {
            return 0.0;
        }
        self.total_evicted as f64 / self.total_cycles as f64
    }

    /// Churn rate: (evicted + forgotten) / total stored
    pub fn churn_rate(&self) -> f64 {
        if self.total_stored == 0 {
            return 0.0;
        }
        (self.total_evicted + self.total_forgotten) as f64 / self.total_stored as f64
    }

    /// Retention rate: promoted / total stored
    pub fn retention_rate(&self) -> f64 {
        if self.total_stored == 0 {
            return 0.0;
        }
        self.total_promoted as f64 / self.total_stored as f64
    }
}

// ---------------------------------------------------------------------------
// Simple PRNG (xoshiro-style, lightweight)
// ---------------------------------------------------------------------------

struct Rng {
    s: [u64; 2],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s0 = seed.wrapping_add(0x9E3779B97F4A7C15);
        let s1 = s0.wrapping_mul(0xBF58476D1CE4E5B9);
        Self {
            s: [s0 | 1, s1 | 1],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s0 = self.s[0];
        let mut s1 = self.s[1];
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s[0] = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s[1] = s1.rotate_left(37);
        result
    }

    /// Uniform [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n)
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Consolidation Engine
// ---------------------------------------------------------------------------

// Memory consolidation engine.
//
// Manages short-term and long-term experience buffers with importance-weighted
// promotion, exponential forgetting, and prioritised replay sampling.
// ---------------------------------------------------------------------------
// Free-standing helpers (avoid borrow-checker conflicts when used inside
// mutable iteration over `self.long_term` / `self.short_term`).
// ---------------------------------------------------------------------------

/// Compute importance score for an experience using config weights.
fn compute_importance_static(config: &ConsolidationConfig, exp: &Experience) -> f64 {
    let wsum = config.td_error_weight + config.reward_weight + config.novelty_weight;
    if wsum.abs() < 1e-15 {
        return 0.0;
    }
    let td_component = exp.td_error.abs().tanh();
    let reward_component = exp.reward.abs().tanh();
    let novelty_component = exp.novelty.clamp(0.0, 1.0);

    let raw = config.td_error_weight * td_component
        + config.reward_weight * reward_component
        + config.novelty_weight * novelty_component;

    (raw / wsum).clamp(0.0, 1.0)
}

/// Compute memory strength after decay: strength = exp(-λ * age), λ = ln(2) / half_life.
fn decay_strength_static(config: &ConsolidationConfig, age: u64) -> f64 {
    let lambda = (2.0_f64).ln() / config.forgetting_half_life;
    (-lambda * age as f64).exp()
}

/// Same as `decay_strength_static` but takes the half-life directly.
fn decay_strength_with_half_life(half_life: f64, age: u64) -> f64 {
    let lambda = (2.0_f64).ln() / half_life;
    (-lambda * age as f64).exp()
}

pub struct Consolidation {
    config: ConsolidationConfig,
    /// Short-term buffer (recent, not yet consolidated)
    short_term: Vec<Experience>,
    /// Long-term buffer (consolidated, persistent)
    long_term: Vec<Experience>,
    /// Monotonically increasing ID counter
    next_id: u64,
    /// Consolidation cycle counter
    cycle: u64,
    /// EMA initialised flag
    ema_initialized: bool,
    /// Sliding window of recent consolidation summaries
    recent: VecDeque<ConsolidationSummary>,
    /// Running statistics
    stats: ConsolidationStats,
    /// PRNG for sampling
    rng: Rng,
    /// Maximum priority in the long-term buffer (for normalisation)
    max_priority: f64,
}

impl Default for Consolidation {
    fn default() -> Self {
        Self::new()
    }
}

impl Consolidation {
    /// Create a new consolidation engine with default configuration
    pub fn new() -> Self {
        Self {
            config: ConsolidationConfig::default(),
            short_term: Vec::new(),
            long_term: Vec::new(),
            next_id: 0,
            cycle: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ConsolidationStats::default(),
            rng: Rng::new(42),
            max_priority: 1.0,
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: ConsolidationConfig) -> Result<Self> {
        if config.short_term_capacity == 0 {
            return Err(Error::InvalidInput(
                "short_term_capacity must be > 0".into(),
            ));
        }
        if config.long_term_capacity == 0 {
            return Err(Error::InvalidInput("long_term_capacity must be > 0".into()));
        }
        if config.forgetting_half_life <= 0.0 {
            return Err(Error::InvalidInput(
                "forgetting_half_life must be > 0".into(),
            ));
        }
        if config.consolidation_fraction <= 0.0 || config.consolidation_fraction > 1.0 {
            return Err(Error::InvalidInput(
                "consolidation_fraction must be in (0, 1]".into(),
            ));
        }
        if config.priority_exponent < 0.0 || config.priority_exponent > 1.0 {
            return Err(Error::InvalidInput(
                "priority_exponent must be in [0, 1]".into(),
            ));
        }
        if config.is_correction_exponent < 0.0 || config.is_correction_exponent > 1.0 {
            return Err(Error::InvalidInput(
                "is_correction_exponent must be in [0, 1]".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        let wsum = config.novelty_weight + config.td_error_weight + config.reward_weight;
        if wsum.abs() < 1e-15 {
            return Err(Error::InvalidInput(
                "importance weights must sum to a positive value".into(),
            ));
        }
        Ok(Self {
            config,
            short_term: Vec::new(),
            long_term: Vec::new(),
            next_id: 0,
            cycle: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ConsolidationStats::default(),
            rng: Rng::new(42),
            max_priority: 1.0,
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Importance scoring
    // -----------------------------------------------------------------------

    /// Compute importance score for an experience.
    ///
    /// Importance = weighted combination of TD-error magnitude, reward
    /// magnitude, and novelty score, all normalised to [0, 1].
    pub fn compute_importance(&self, exp: &Experience) -> f64 {
        compute_importance_static(&self.config, exp)
    }

    /// Compute memory strength after decay.
    ///
    /// Uses an exponential forgetting curve: strength = exp(-λ * age)
    /// where λ = ln(2) / half_life.
    pub fn decay_strength(&self, age: u64) -> f64 {
        decay_strength_static(&self.config, age)
    }

    // -----------------------------------------------------------------------
    // Experience storage
    // -----------------------------------------------------------------------

    /// Store a new experience in the short-term buffer.
    ///
    /// Automatically computes importance and assigns priority. If the buffer
    /// is full, the lowest-priority experience is evicted.
    pub fn store(&mut self, mut exp: Experience) -> u64 {
        exp.id = self.next_id;
        self.next_id += 1;
        exp.importance = self.compute_importance(&exp);
        exp.priority =
            (exp.importance + self.config.priority_epsilon).powf(self.config.priority_exponent);
        exp.strength = 1.0;
        exp.age = 0;

        // Evict lowest-priority if at capacity
        if self.short_term.len() >= self.config.short_term_capacity {
            if let Some((min_idx, _)) =
                self.short_term.iter().enumerate().min_by(|(_, a), (_, b)| {
                    a.priority
                        .partial_cmp(&b.priority)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                self.short_term.swap_remove(min_idx);
                self.stats.total_evicted += 1;
            }
        }

        let id = exp.id;
        self.stats.total_stored += 1;
        self.short_term.push(exp);
        id
    }

    /// Store a raw transition (convenience wrapper)
    pub fn store_transition(
        &mut self,
        state: Vec<f64>,
        action: Vec<f64>,
        reward: f64,
        next_state: Vec<f64>,
        terminal: bool,
    ) -> u64 {
        let exp = Experience::new(0, state, action, reward, next_state, terminal);
        self.store(exp)
    }

    /// Update the TD-error for an experience in the long-term buffer
    pub fn update_td_error(&mut self, id: u64, td_error: f64) {
        let cfg = self.config.clone();
        for exp in self.long_term.iter_mut() {
            if exp.id == id {
                exp.td_error = td_error.abs();
                exp.importance = compute_importance_static(&cfg, exp);
                exp.priority = (exp.importance + cfg.priority_epsilon).powf(cfg.priority_exponent);
                if exp.priority > self.max_priority {
                    self.max_priority = exp.priority;
                }
                return;
            }
        }
        // Also check short-term
        for exp in self.short_term.iter_mut() {
            if exp.id == id {
                exp.td_error = td_error.abs();
                exp.importance = compute_importance_static(&cfg, exp);
                exp.priority = (exp.importance + cfg.priority_epsilon).powf(cfg.priority_exponent);
                return;
            }
        }
    }

    /// Update the novelty score for an experience
    pub fn update_novelty(&mut self, id: u64, novelty: f64) {
        let clamped = novelty.clamp(0.0, 1.0);
        let cfg = self.config.clone();
        for exp in self.short_term.iter_mut().chain(self.long_term.iter_mut()) {
            if exp.id == id {
                exp.novelty = clamped;
                exp.importance = compute_importance_static(&cfg, exp);
                exp.priority = (exp.importance + cfg.priority_epsilon).powf(cfg.priority_exponent);
                return;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Consolidation cycle
    // -----------------------------------------------------------------------

    /// Run a consolidation cycle.
    ///
    /// This:
    /// 1. Ages all short-term memories and evicts those exceeding max age
    /// 2. Evaluates a fraction of short-term memories for promotion
    /// 3. Promotes those above the importance threshold to long-term
    /// 4. Applies forgetting decay to all long-term memories
    /// 5. Removes long-term memories that have decayed below a strength floor
    /// 6. Updates statistics
    pub fn consolidate(&mut self) -> ConsolidationSummary {
        self.cycle += 1;

        // 1. Age short-term memories
        for exp in self.short_term.iter_mut() {
            exp.age += 1;
        }

        // Evict aged-out short-term memories
        let before_evict = self.short_term.len();
        self.short_term
            .retain(|exp| exp.age <= self.config.max_short_term_age);
        let evicted_age = before_evict - self.short_term.len();
        self.stats.total_evicted += evicted_age as u64;

        // 2. Evaluate a fraction of short-term buffer for promotion
        let eval_count =
            (self.short_term.len() as f64 * self.config.consolidation_fraction).ceil() as usize;
        let eval_count = eval_count.min(self.short_term.len());

        // Sort by importance descending to evaluate the most important first
        self.short_term.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut promoted_count = 0usize;
        let mut promoted_importance_sum = 0.0f64;
        let mut to_promote: Vec<usize> = Vec::new();

        for i in 0..eval_count {
            let exp = &self.short_term[i];
            if exp.importance >= self.config.promotion_threshold {
                to_promote.push(i);
            }
        }

        // Promote (process in reverse order to keep indices valid during removal)
        to_promote.sort_unstable();
        for &idx in to_promote.iter().rev() {
            let mut exp = self.short_term.remove(idx);
            exp.promoted = true;
            exp.age = 0; // Reset age for long-term tracking
            exp.strength = 1.0;

            promoted_importance_sum += exp.importance;

            if exp.priority > self.max_priority {
                self.max_priority = exp.priority;
            }

            // Evict lowest-priority long-term if at capacity
            if self.long_term.len() >= self.config.long_term_capacity {
                if let Some((min_idx, _)) =
                    self.long_term.iter().enumerate().min_by(|(_, a), (_, b)| {
                        a.priority
                            .partial_cmp(&b.priority)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    // Only evict if the new experience is more important
                    if exp.priority > self.long_term[min_idx].priority {
                        self.long_term.swap_remove(min_idx);
                        self.stats.total_evicted += 1;
                    } else {
                        // Can't promote — put it back as evaluated
                        continue;
                    }
                }
            }

            self.long_term.push(exp);
            promoted_count += 1;
        }

        self.stats.total_promoted += promoted_count as u64;

        let avg_promoted_importance = if promoted_count > 0 {
            promoted_importance_sum / promoted_count as f64
        } else {
            0.0
        };

        // 4. Apply forgetting decay to long-term memories
        let half_life = self.config.forgetting_half_life;
        let priority_epsilon = self.config.priority_epsilon;
        let priority_exponent = self.config.priority_exponent;
        for exp in self.long_term.iter_mut() {
            exp.age += 1;
            exp.strength = decay_strength_with_half_life(half_life, exp.age);
            // Update priority based on decayed strength
            exp.priority =
                (exp.importance * exp.strength + priority_epsilon).powf(priority_exponent);
        }

        // 5. Remove long-term memories with strength below a floor
        let strength_floor = 0.01;
        let before_forget = self.long_term.len();
        self.long_term.retain(|exp| exp.strength >= strength_floor);
        let forgotten = before_forget - self.long_term.len();
        self.stats.total_forgotten += forgotten as u64;

        // Update max_priority
        self.max_priority = self
            .long_term
            .iter()
            .map(|e| e.priority)
            .fold(self.config.priority_epsilon, f64::max);

        // Compute summary metrics
        let avg_long_term_strength = if self.long_term.is_empty() {
            0.0
        } else {
            self.long_term.iter().map(|e| e.strength).sum::<f64>() / self.long_term.len() as f64
        };

        let short_term_util = self.short_term.len() as f64 / self.config.short_term_capacity as f64;
        let long_term_util = self.long_term.len() as f64 / self.config.long_term_capacity as f64;

        // 6. Update statistics
        self.stats.total_cycles += 1;

        let promotion_rate = if eval_count > 0 {
            promoted_count as f64 / eval_count as f64
        } else {
            0.0
        };

        let avg_td = if self.long_term.is_empty() {
            0.0
        } else {
            self.long_term.iter().map(|e| e.td_error).sum::<f64>() / self.long_term.len() as f64
        };

        if avg_promoted_importance > self.stats.best_importance {
            self.stats.best_importance = avg_promoted_importance;
        }

        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_promotion_rate = promotion_rate;
            self.stats.ema_promoted_importance = avg_promoted_importance;
            self.stats.ema_long_term_strength = avg_long_term_strength;
            self.stats.ema_short_term_util = short_term_util;
            self.stats.ema_long_term_util = long_term_util;
            self.stats.ema_avg_td_error = avg_td;
            self.ema_initialized = true;
        } else {
            self.stats.ema_promotion_rate =
                alpha * promotion_rate + (1.0 - alpha) * self.stats.ema_promotion_rate;
            self.stats.ema_promoted_importance = alpha * avg_promoted_importance
                + (1.0 - alpha) * self.stats.ema_promoted_importance;
            self.stats.ema_long_term_strength =
                alpha * avg_long_term_strength + (1.0 - alpha) * self.stats.ema_long_term_strength;
            self.stats.ema_short_term_util =
                alpha * short_term_util + (1.0 - alpha) * self.stats.ema_short_term_util;
            self.stats.ema_long_term_util =
                alpha * long_term_util + (1.0 - alpha) * self.stats.ema_long_term_util;
            self.stats.ema_avg_td_error =
                alpha * avg_td + (1.0 - alpha) * self.stats.ema_avg_td_error;
        }

        let summary = ConsolidationSummary {
            evaluated: eval_count,
            promoted: promoted_count,
            evicted: evicted_age,
            forgotten,
            avg_promoted_importance,
            avg_long_term_strength,
            short_term_utilisation: short_term_util,
            long_term_utilisation: long_term_util,
            cycle: self.cycle,
        };

        // Sliding window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(summary.clone());

        summary
    }

    // -----------------------------------------------------------------------
    // Prioritised replay sampling
    // -----------------------------------------------------------------------

    /// Sample a mini-batch from the long-term buffer using prioritised replay.
    ///
    /// Returns indices, importance-sampling weights, and cloned experiences.
    /// Returns an error if the buffer is empty or batch_size > buffer size.
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        if self.long_term.is_empty() {
            return Err(Error::InvalidState(
                "cannot sample from empty long-term buffer".into(),
            ));
        }
        let actual_batch = batch_size.min(self.long_term.len());
        if actual_batch == 0 {
            return Err(Error::InvalidInput("batch_size must be > 0".into()));
        }

        let n = self.long_term.len();

        // Compute sampling probabilities from priorities
        let total_priority: f64 = self.long_term.iter().map(|e| e.priority).sum();

        let mut indices = Vec::with_capacity(actual_batch);
        let mut is_weights = Vec::with_capacity(actual_batch);
        let mut experiences = Vec::with_capacity(actual_batch);

        // Rank-based sampling with replacement avoidance (best-effort)
        let mut selected = vec![false; n];
        let mut attempts = 0;
        let max_attempts = actual_batch * 10;

        while indices.len() < actual_batch && attempts < max_attempts {
            attempts += 1;
            // Sample proportional to priority
            let threshold = self.rng.next_f64() * total_priority;
            let mut cumulative = 0.0;
            let mut idx = 0;
            for (i, exp) in self.long_term.iter().enumerate() {
                cumulative += exp.priority;
                if cumulative >= threshold {
                    idx = i;
                    break;
                }
            }
            if idx >= n {
                idx = n - 1;
            }
            if selected[idx] {
                continue; // Avoid duplicates
            }
            selected[idx] = true;

            // Compute IS weight: (1 / (N * P(i)))^β
            let prob = self.long_term[idx].priority / total_priority.max(1e-15);
            let weight =
                (1.0 / (n as f64 * prob.max(1e-15))).powf(self.config.is_correction_exponent);

            indices.push(idx);
            is_weights.push(weight);
            experiences.push(self.long_term[idx].clone());
            self.long_term[idx].replay_count += 1;
        }

        // If we couldn't fill the batch (too many collisions), fill randomly
        if indices.len() < actual_batch {
            for i in 0..n {
                if indices.len() >= actual_batch {
                    break;
                }
                if !selected[i] {
                    selected[i] = true;
                    let prob = self.long_term[i].priority / total_priority.max(1e-15);
                    let weight = (1.0 / (n as f64 * prob.max(1e-15)))
                        .powf(self.config.is_correction_exponent);
                    indices.push(i);
                    is_weights.push(weight);
                    experiences.push(self.long_term[i].clone());
                    self.long_term[i].replay_count += 1;
                }
            }
        }

        // Normalise IS weights so the maximum weight is 1.0
        let max_weight = is_weights.iter().cloned().fold(0.0_f64, f64::max);
        if max_weight > 1e-15 {
            for w in is_weights.iter_mut() {
                *w /= max_weight;
            }
        }

        self.stats.total_sampled += indices.len() as u64;

        Ok(ReplayBatch {
            indices,
            is_weights,
            experiences,
        })
    }

    /// Sample uniformly (non-prioritised) from the long-term buffer
    pub fn sample_uniform(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        if self.long_term.is_empty() {
            return Err(Error::InvalidState(
                "cannot sample from empty long-term buffer".into(),
            ));
        }
        let actual_batch = batch_size.min(self.long_term.len());
        if actual_batch == 0 {
            return Err(Error::InvalidInput("batch_size must be > 0".into()));
        }

        let n = self.long_term.len();
        let mut indices = Vec::with_capacity(actual_batch);
        let mut selected = vec![false; n];

        while indices.len() < actual_batch {
            let idx = self.rng.next_usize(n);
            if !selected[idx] {
                selected[idx] = true;
                indices.push(idx);
            }
        }

        let is_weights = vec![1.0; actual_batch];
        let experiences: Vec<Experience> = indices
            .iter()
            .map(|&i| {
                self.long_term[i].replay_count += 1;
                self.long_term[i].clone()
            })
            .collect();

        self.stats.total_sampled += actual_batch as u64;

        Ok(ReplayBatch {
            indices,
            is_weights,
            experiences,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Short-term buffer size
    pub fn short_term_size(&self) -> usize {
        self.short_term.len()
    }

    /// Long-term buffer size
    pub fn long_term_size(&self) -> usize {
        self.long_term.len()
    }

    /// Total buffer size (short + long)
    pub fn total_size(&self) -> usize {
        self.short_term.len() + self.long_term.len()
    }

    /// Short-term buffer utilisation (0.0–1.0)
    pub fn short_term_utilisation(&self) -> f64 {
        self.short_term.len() as f64 / self.config.short_term_capacity as f64
    }

    /// Long-term buffer utilisation (0.0–1.0)
    pub fn long_term_utilisation(&self) -> f64 {
        self.long_term.len() as f64 / self.config.long_term_capacity as f64
    }

    /// Current cycle number
    pub fn current_cycle(&self) -> u64 {
        self.cycle
    }

    /// Running statistics
    pub fn stats(&self) -> &ConsolidationStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// Recent consolidation summaries (sliding window)
    pub fn recent_summaries(&self) -> &VecDeque<ConsolidationSummary> {
        &self.recent
    }

    /// Get an experience from the long-term buffer by ID
    pub fn get_long_term(&self, id: u64) -> Option<&Experience> {
        self.long_term.iter().find(|e| e.id == id)
    }

    /// Get an experience from the short-term buffer by ID
    pub fn get_short_term(&self, id: u64) -> Option<&Experience> {
        self.short_term.iter().find(|e| e.id == id)
    }

    /// All long-term experiences (read-only)
    pub fn long_term_buffer(&self) -> &[Experience] {
        &self.long_term
    }

    /// All short-term experiences (read-only)
    pub fn short_term_buffer(&self) -> &[Experience] {
        &self.short_term
    }

    /// EMA-smoothed promotion rate
    pub fn smoothed_promotion_rate(&self) -> f64 {
        self.stats.ema_promotion_rate
    }

    /// EMA-smoothed average long-term strength
    pub fn smoothed_long_term_strength(&self) -> f64 {
        self.stats.ema_long_term_strength
    }

    /// EMA-smoothed short-term utilisation
    pub fn smoothed_short_term_util(&self) -> f64 {
        self.stats.ema_short_term_util
    }

    /// EMA-smoothed long-term utilisation
    pub fn smoothed_long_term_util(&self) -> f64 {
        self.stats.ema_long_term_util
    }

    /// Windowed average promotion rate
    pub fn windowed_promotion_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|s| {
                if s.evaluated > 0 {
                    s.promoted as f64 / s.evaluated as f64
                } else {
                    0.0
                }
            })
            .sum();
        sum / self.recent.len() as f64
    }

    /// Windowed average long-term strength
    pub fn windowed_avg_strength(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.avg_long_term_strength).sum();
        sum / self.recent.len() as f64
    }

    /// Whether long-term memory strength is declining (second half < first half)
    pub fn is_strength_declining(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|s| s.avg_long_term_strength)
            .sum::<f64>()
            / mid as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|s| s.avg_long_term_strength)
            .sum::<f64>()
            / (n - mid) as f64;
        second_avg < first_avg * 0.95
    }

    /// Reset all state (clear buffers and statistics)
    pub fn reset(&mut self) {
        self.short_term.clear();
        self.long_term.clear();
        self.next_id = 0;
        self.cycle = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = ConsolidationStats::default();
        self.max_priority = 1.0;
        self.rng = Rng::new(42);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn simple_exp(reward: f64, td_error: f64, novelty: f64) -> Experience {
        Experience::new(
            0,
            vec![0.1, 0.2, 0.3],
            vec![1.0],
            reward,
            vec![0.2, 0.3, 0.4],
            false,
        )
        .with_td_error(td_error)
        .with_novelty(novelty)
    }

    fn high_importance_exp() -> Experience {
        simple_exp(5.0, 2.0, 0.9)
    }

    fn low_importance_exp() -> Experience {
        simple_exp(0.01, 0.001, 0.05)
    }

    fn small_config() -> ConsolidationConfig {
        ConsolidationConfig {
            short_term_capacity: 20,
            long_term_capacity: 50,
            forgetting_half_life: 10.0,
            promotion_threshold: 0.3,
            consolidation_fraction: 0.50,
            priority_exponent: 0.6,
            is_correction_exponent: 0.4,
            novelty_weight: 0.25,
            td_error_weight: 0.50,
            reward_weight: 0.25,
            priority_epsilon: 1e-6,
            ema_decay: 0.2,
            window_size: 10,
            max_short_term_age: 10,
        }
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let c = Consolidation::new();
        assert_eq!(c.short_term_size(), 0);
        assert_eq!(c.long_term_size(), 0);
        assert_eq!(c.current_cycle(), 0);
        assert!(c.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let c = Consolidation::with_config(small_config());
        assert!(c.is_ok());
    }

    #[test]
    fn test_invalid_short_term_capacity() {
        let mut cfg = small_config();
        cfg.short_term_capacity = 0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_long_term_capacity() {
        let mut cfg = small_config();
        cfg.long_term_capacity = 0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_half_life() {
        let mut cfg = small_config();
        cfg.forgetting_half_life = 0.0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_consolidation_fraction_zero() {
        let mut cfg = small_config();
        cfg.consolidation_fraction = 0.0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_consolidation_fraction_over_one() {
        let mut cfg = small_config();
        cfg.consolidation_fraction = 1.5;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_priority_exponent_negative() {
        let mut cfg = small_config();
        cfg.priority_exponent = -0.1;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let mut cfg = small_config();
        cfg.ema_decay = 0.0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let mut cfg = small_config();
        cfg.ema_decay = 1.0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_window_size() {
        let mut cfg = small_config();
        cfg.window_size = 0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_zero_weights() {
        let mut cfg = small_config();
        cfg.novelty_weight = 0.0;
        cfg.td_error_weight = 0.0;
        cfg.reward_weight = 0.0;
        assert!(Consolidation::with_config(cfg).is_err());
    }

    // -- Importance scoring --

    #[test]
    fn test_importance_high() {
        let c = Consolidation::new();
        let exp = high_importance_exp();
        let imp = c.compute_importance(&exp);
        assert!(imp > 0.5);
        assert!(imp <= 1.0);
    }

    #[test]
    fn test_importance_low() {
        let c = Consolidation::new();
        let exp = low_importance_exp();
        let imp = c.compute_importance(&exp);
        assert!(imp < 0.3);
        assert!(imp >= 0.0);
    }

    #[test]
    fn test_importance_bounded() {
        let c = Consolidation::new();
        let exp = simple_exp(100.0, 100.0, 1.0);
        let imp = c.compute_importance(&exp);
        assert!((0.0..=1.0).contains(&imp));
    }

    #[test]
    fn test_importance_zero_inputs() {
        let c = Consolidation::new();
        let exp = simple_exp(0.0, 0.0, 0.0);
        let imp = c.compute_importance(&exp);
        assert!((imp - 0.0).abs() < 1e-10);
    }

    // -- Forgetting curve --

    #[test]
    fn test_decay_strength_at_zero_age() {
        let c = Consolidation::new();
        assert!((c.decay_strength(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_strength_at_half_life() {
        let c = Consolidation::with_config(ConsolidationConfig {
            forgetting_half_life: 10.0,
            ..small_config()
        })
        .unwrap();
        let strength = c.decay_strength(10);
        assert!((strength - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_strength_decreases_with_age() {
        let c = Consolidation::new();
        let s1 = c.decay_strength(5);
        let s2 = c.decay_strength(10);
        let s3 = c.decay_strength(50);
        assert!(s1 > s2);
        assert!(s2 > s3);
        assert!(s3 > 0.0);
    }

    #[test]
    fn test_decay_strength_always_positive() {
        let c = Consolidation::new();
        for age in 0..1000 {
            assert!(c.decay_strength(age) > 0.0);
        }
    }

    // -- Storage --

    #[test]
    fn test_store_experience() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(high_importance_exp());
        assert_eq!(c.short_term_size(), 1);
        assert!(c.get_short_term(id).is_some());
    }

    #[test]
    fn test_store_assigns_id() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id1 = c.store(high_importance_exp());
        let id2 = c.store(low_importance_exp());
        assert_ne!(id1, id2);
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_store_computes_importance() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(high_importance_exp());
        let exp = c.get_short_term(id).unwrap();
        assert!(exp.importance > 0.0);
        assert!(exp.priority > 0.0);
        assert_eq!(exp.age, 0);
        assert!((exp.strength - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_store_transition_convenience() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store_transition(vec![1.0, 2.0], vec![0.5], 1.0, vec![1.1, 2.1], false);
        assert_eq!(c.short_term_size(), 1);
        assert!(c.get_short_term(id).is_some());
    }

    #[test]
    fn test_store_evicts_at_capacity() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        // Capacity is 20; fill it
        for _ in 0..20 {
            c.store(low_importance_exp());
        }
        assert_eq!(c.short_term_size(), 20);

        // Store one more — should evict the lowest priority
        c.store(high_importance_exp());
        assert_eq!(c.short_term_size(), 20);
    }

    #[test]
    fn test_store_with_tag() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let exp = high_importance_exp().with_tag("momentum");
        let id = c.store(exp);
        assert_eq!(c.get_short_term(id).unwrap().tag, "momentum");
    }

    #[test]
    fn test_stats_total_stored() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        c.store(high_importance_exp());
        c.store(low_importance_exp());
        assert_eq!(c.stats().total_stored, 2);
    }

    // -- Update TD error --

    #[test]
    fn test_update_td_error() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(simple_exp(1.0, 0.1, 0.5));
        c.update_td_error(id, 5.0);
        let exp = c.get_short_term(id).unwrap();
        assert!((exp.td_error - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_update_td_error_negative_becomes_abs() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(simple_exp(1.0, 0.1, 0.5));
        c.update_td_error(id, -3.0);
        let exp = c.get_short_term(id).unwrap();
        assert!((exp.td_error - 3.0).abs() < 1e-10);
    }

    // -- Update novelty --

    #[test]
    fn test_update_novelty() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(simple_exp(1.0, 0.1, 0.2));
        c.update_novelty(id, 0.9);
        let exp = c.get_short_term(id).unwrap();
        assert!((exp.novelty - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_update_novelty_clamped() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(simple_exp(1.0, 0.1, 0.5));
        c.update_novelty(id, 5.0);
        let exp = c.get_short_term(id).unwrap();
        assert!((exp.novelty - 1.0).abs() < 1e-10);
    }

    // -- Consolidation cycle --

    #[test]
    fn test_consolidate_empty() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let summary = c.consolidate();
        assert_eq!(summary.evaluated, 0);
        assert_eq!(summary.promoted, 0);
        assert_eq!(summary.cycle, 1);
    }

    #[test]
    fn test_consolidate_promotes_high_importance() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        c.store(high_importance_exp());
        c.store(high_importance_exp());
        c.store(high_importance_exp());

        let summary = c.consolidate();
        assert!(summary.promoted > 0);
        assert!(c.long_term_size() > 0);
    }

    #[test]
    fn test_consolidate_does_not_promote_low_importance() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        c.store(low_importance_exp());
        c.store(low_importance_exp());

        let summary = c.consolidate();
        assert_eq!(summary.promoted, 0);
        assert_eq!(c.long_term_size(), 0);
    }

    #[test]
    fn test_consolidate_ages_short_term() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        let id = c.store(high_importance_exp());
        c.consolidate();
        // If promoted, it's in long-term; otherwise age incremented
        let in_st = c.get_short_term(id);
        let in_lt = c.get_long_term(id);
        if let Some(exp) = in_st {
            assert!(exp.age >= 1);
        }
        if let Some(exp) = in_lt {
            // Promoted and then aged by 1 in long-term
            assert!(exp.age >= 1);
        }
    }

    #[test]
    fn test_consolidate_evicts_aged_short_term() {
        let mut cfg = small_config();
        cfg.max_short_term_age = 2;
        cfg.promotion_threshold = 999.0; // prevent promotion
        let mut c = Consolidation::with_config(cfg).unwrap();

        c.store(low_importance_exp());
        assert_eq!(c.short_term_size(), 1);

        c.consolidate(); // age = 1
        assert_eq!(c.short_term_size(), 1);

        c.consolidate(); // age = 2
        assert_eq!(c.short_term_size(), 1);

        c.consolidate(); // age = 3 > max_short_term_age = 2 → evicted
        assert_eq!(c.short_term_size(), 0);
    }

    #[test]
    fn test_consolidate_applies_decay() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        // Store and immediately consolidate to promote
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert!(c.long_term_size() > 0);

        // Run many cycles to decay
        for _ in 0..50 {
            c.consolidate();
        }

        // Long-term memories should have decayed strength
        for exp in c.long_term_buffer() {
            assert!(exp.strength < 1.0);
        }
    }

    #[test]
    fn test_consolidate_forgets_very_old_memories() {
        let mut cfg = small_config();
        cfg.forgetting_half_life = 5.0; // Very short half-life
        cfg.promotion_threshold = 0.1; // Easy to promote
        let mut c = Consolidation::with_config(cfg).unwrap();

        // Store and promote
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert!(c.long_term_size() > 0);

        // Run many cycles — memories should eventually be forgotten
        for _ in 0..500 {
            c.consolidate();
        }

        // With half-life 5 and 500 cycles, strength ≈ 2^(-500/5) ≈ 0 → forgotten
        // Some might remain if they were recently promoted from later stores
        // but the original ones should be gone
        let total_forgotten = c.stats().total_forgotten;
        assert!(total_forgotten > 0);
    }

    #[test]
    fn test_consolidate_cycle_counter() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        c.consolidate();
        c.consolidate();
        c.consolidate();
        assert_eq!(c.current_cycle(), 3);
    }

    #[test]
    fn test_consolidate_updates_stats() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        assert_eq!(c.stats().total_cycles, 1);
        assert!(c.stats().total_promoted > 0);
    }

    #[test]
    fn test_consolidate_summary_utilisation() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        let summary = c.consolidate();
        // 10 stored in capacity 20 → 50% short-term (minus promoted)
        assert!(summary.short_term_utilisation >= 0.0);
        assert!(summary.short_term_utilisation <= 1.0);
        assert!(summary.long_term_utilisation >= 0.0);
    }

    // -- Replay sampling --

    #[test]
    fn test_sample_empty_buffer_error() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        assert!(c.sample(5).is_err());
    }

    #[test]
    fn test_sample_from_long_term() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        c.consolidate(); // Promote to long-term
        assert!(c.long_term_size() > 0);

        let batch = c.sample(3);
        assert!(batch.is_ok());
        let batch = batch.unwrap();
        assert_eq!(batch.experiences.len(), 3);
        assert_eq!(batch.indices.len(), 3);
        assert_eq!(batch.is_weights.len(), 3);
    }

    #[test]
    fn test_sample_batch_capped_at_buffer_size() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..3 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        let lt_size = c.long_term_size();
        assert!(lt_size > 0);

        // Request more than buffer size
        let batch = c.sample(100).unwrap();
        assert!(batch.experiences.len() <= lt_size);
    }

    #[test]
    fn test_sample_is_weights_normalised() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        let batch = c.sample(5).unwrap();
        let max_w = batch.is_weights.iter().cloned().fold(0.0_f64, f64::max);
        assert!((max_w - 1.0).abs() < 1e-10);
        for &w in &batch.is_weights {
            assert!((0.0..=1.0 + 1e-10).contains(&w));
        }
    }

    #[test]
    fn test_sample_increments_replay_count() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        c.sample(3).unwrap();

        let total_replays: u64 = c.long_term_buffer().iter().map(|e| e.replay_count).sum();
        assert!(total_replays >= 3);
    }

    #[test]
    fn test_sample_updates_total_sampled() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        c.sample(3).unwrap();
        assert!(c.stats().total_sampled >= 3);
    }

    // -- Uniform sampling --

    #[test]
    fn test_sample_uniform_empty_error() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        assert!(c.sample_uniform(5).is_err());
    }

    #[test]
    fn test_sample_uniform() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        let batch = c.sample_uniform(3).unwrap();
        assert_eq!(batch.experiences.len(), 3);
        // Uniform sampling has all IS weights = 1.0
        for &w in &batch.is_weights {
            assert!((w - 1.0).abs() < 1e-10);
        }
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let c = Consolidation::new();
        assert_eq!(c.stats().total_cycles, 0);
        assert_eq!(c.stats().total_stored, 0);
        assert_eq!(c.stats().total_promoted, 0);
    }

    #[test]
    fn test_stats_avg_promotions_per_cycle() {
        let stats = ConsolidationStats {
            total_cycles: 10,
            total_promoted: 25,
            ..Default::default()
        };
        assert!((stats.avg_promotions_per_cycle() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_avg_promotions_zero_cycles() {
        let stats = ConsolidationStats::default();
        assert_eq!(stats.avg_promotions_per_cycle(), 0.0);
    }

    #[test]
    fn test_stats_churn_rate() {
        let stats = ConsolidationStats {
            total_stored: 100,
            total_evicted: 10,
            total_forgotten: 5,
            ..Default::default()
        };
        assert!((stats.churn_rate() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_stats_churn_rate_zero_stored() {
        let stats = ConsolidationStats::default();
        assert_eq!(stats.churn_rate(), 0.0);
    }

    #[test]
    fn test_stats_retention_rate() {
        let stats = ConsolidationStats {
            total_stored: 100,
            total_promoted: 30,
            ..Default::default()
        };
        assert!((stats.retention_rate() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_stats_retention_rate_zero() {
        let stats = ConsolidationStats::default();
        assert_eq!(stats.retention_rate(), 0.0);
    }

    // -- EMA --

    #[test]
    fn test_ema_initializes_first_cycle() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert!(c.smoothed_promotion_rate() >= 0.0);
        assert!(c.smoothed_long_term_strength() >= 0.0);
    }

    #[test]
    fn test_ema_blends_subsequent() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        let strength1 = c.smoothed_long_term_strength();

        // Many cycles without new data → strength should decay
        for _ in 0..20 {
            c.consolidate();
        }
        let strength2 = c.smoothed_long_term_strength();
        assert!(strength2 < strength1);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_summaries_stored() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        c.consolidate();
        assert_eq!(c.recent_summaries().len(), 1);
    }

    #[test]
    fn test_recent_summaries_windowed() {
        let mut cfg = small_config();
        cfg.window_size = 3;
        let mut c = Consolidation::with_config(cfg).unwrap();

        for _ in 0..10 {
            c.consolidate();
        }
        assert!(c.recent_summaries().len() <= 3);
    }

    #[test]
    fn test_windowed_promotion_rate() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert!(c.windowed_promotion_rate() >= 0.0);
    }

    #[test]
    fn test_windowed_avg_strength() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert!(c.windowed_avg_strength() >= 0.0);
    }

    #[test]
    fn test_windowed_empty() {
        let c = Consolidation::new();
        assert_eq!(c.windowed_promotion_rate(), 0.0);
        assert_eq!(c.windowed_avg_strength(), 0.0);
    }

    // -- Trend detection --

    #[test]
    fn test_is_strength_declining_insufficient_data() {
        let c = Consolidation::new();
        assert!(!c.is_strength_declining());
    }

    // -- Utilisation --

    #[test]
    fn test_short_term_utilisation() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(low_importance_exp());
        }
        assert!((c.short_term_utilisation() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_long_term_utilisation_initially_zero() {
        let c = Consolidation::with_config(small_config()).unwrap();
        assert!((c.long_term_utilisation() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_size() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        assert_eq!(c.total_size(), c.short_term_size() + c.long_term_size());
    }

    // -- Experience builder --

    #[test]
    fn test_experience_builder() {
        let exp = Experience::new(0, vec![1.0], vec![0.5], 2.0, vec![1.1], true)
            .with_td_error(-3.0)
            .with_novelty(0.8)
            .with_tag("test_strategy");

        assert!((exp.td_error - 3.0).abs() < 1e-10); // abs
        assert!((exp.novelty - 0.8).abs() < 1e-10);
        assert_eq!(exp.tag, "test_strategy");
        assert!(exp.terminal);
    }

    #[test]
    fn test_experience_new_defaults() {
        let exp = Experience::new(42, vec![1.0], vec![0.5], 1.0, vec![1.1], false);
        assert_eq!(exp.id, 42);
        assert_eq!(exp.td_error, 0.0);
        assert_eq!(exp.novelty, 0.0);
        assert_eq!(exp.importance, 0.0);
        assert_eq!(exp.priority, 1.0);
        assert_eq!(exp.age, 0);
        assert!((exp.strength - 1.0).abs() < 1e-10);
        assert_eq!(exp.replay_count, 0);
        assert!(!exp.promoted);
        assert!(exp.tag.is_empty());
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        c.consolidate();
        c.consolidate();

        assert!(c.short_term_size() > 0 || c.long_term_size() > 0);
        assert!(c.current_cycle() > 0);
        assert!(c.stats().total_cycles > 0);

        c.reset();

        assert_eq!(c.short_term_size(), 0);
        assert_eq!(c.long_term_size(), 0);
        assert_eq!(c.current_cycle(), 0);
        assert_eq!(c.stats().total_cycles, 0);
        assert_eq!(c.stats().total_stored, 0);
        assert!(c.recent_summaries().is_empty());
    }

    // -- Integration-style test --

    #[test]
    fn test_full_lifecycle() {
        let mut c = Consolidation::with_config(small_config()).unwrap();

        // Phase 1: accumulate experiences
        for i in 0..15 {
            let reward = if i % 3 == 0 { 5.0 } else { 0.1 };
            let td = if i % 2 == 0 { 2.0 } else { 0.01 };
            let novelty = if i % 5 == 0 { 0.9 } else { 0.1 };
            let exp = simple_exp(reward, td, novelty);
            c.store(exp);
        }

        assert_eq!(c.short_term_size(), 15);
        assert_eq!(c.long_term_size(), 0);

        // Phase 2: consolidation — promotes high-importance ones
        let summary = c.consolidate();
        assert!(summary.evaluated > 0);
        assert!(summary.promoted > 0 || summary.evaluated > 0);

        // Phase 3: replay from long-term
        if c.long_term_size() > 0 {
            let batch = c.sample(3.min(c.long_term_size())).unwrap();
            assert!(!batch.experiences.is_empty());
            assert_eq!(batch.is_weights.len(), batch.experiences.len());

            // Update TD errors from replay
            for exp in &batch.experiences {
                c.update_td_error(exp.id, 0.5);
            }
        }

        // Phase 4: many consolidation cycles — forgetting kicks in
        for _ in 0..100 {
            c.consolidate();
        }

        // Stats should reflect the lifecycle
        assert!(c.stats().total_cycles > 100);
        assert!(c.stats().total_stored == 15);
        assert!(c.stats().total_promoted > 0);
        assert!(c.smoothed_promotion_rate() >= 0.0);
        assert!(c.smoothed_long_term_strength() >= 0.0);

        // Windowed metrics should be available
        assert!(!c.recent_summaries().is_empty());
    }

    // -- Consolidation summary fields --

    #[test]
    fn test_consolidation_summary_fields() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..10 {
            c.store(high_importance_exp());
        }
        let summary = c.consolidate();

        assert!(summary.evaluated > 0);
        assert!(summary.avg_promoted_importance >= 0.0);
        assert!(summary.avg_long_term_strength >= 0.0);
        assert!(summary.short_term_utilisation >= 0.0);
        assert!(summary.long_term_utilisation >= 0.0);
        assert_eq!(summary.cycle, 1);
    }

    // -- Long-term capacity enforcement --

    #[test]
    fn test_long_term_capacity_enforced() {
        let mut cfg = small_config();
        cfg.long_term_capacity = 5;
        cfg.promotion_threshold = 0.1; // Easy to promote
        let mut c = Consolidation::with_config(cfg).unwrap();

        for _ in 0..20 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        assert!(c.long_term_size() <= 5);
    }

    // -- Best importance tracking --

    #[test]
    fn test_best_importance_tracked() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        if c.stats().total_promoted > 0 {
            assert!(c.stats().best_importance > 0.0);
        }
    }

    // -- Promoted flag --

    #[test]
    fn test_promoted_flag_set() {
        let mut c = Consolidation::with_config(small_config()).unwrap();
        for _ in 0..5 {
            c.store(high_importance_exp());
        }
        c.consolidate();

        for exp in c.long_term_buffer() {
            assert!(exp.promoted);
        }
    }
}
