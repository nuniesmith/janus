//! Engineered Heterogeneity — Crowding Resistance Through Per-Instance Parameterization
//!
//! This module implements the engineered heterogeneity mechanism described in
//! the JANUS specification (Section IV, Table 1). Each JANUS deployment receives
//! a unique [`HeterogeneityProfile`] that perturbs brain-region parameters away
//! from their defaults, producing idiosyncratic order flow that reduces
//! cross-instance correlation and mitigates co-impact costs.
//!
//! # Theory
//!
//! Strategy crowding — the convergence of multiple algorithmic trading systems
//! on similar positions — amplifies co-impact costs (Bucci et al., 2020) and
//! creates correlated drawdowns (Khandani & Lo, 2011). Two key results motivate
//! JANUS's defense:
//!
//! - **Wagner (2011):** Rational agents should choose heterogeneous portfolios
//!   and forgo diversification benefits to avoid joint liquidation risk.
//! - **DeMiguel et al. (2021):** Trading diversification — institutions
//!   exploiting different characteristics — increases capacity by 45%.
//!
//! # Design
//!
//! Heterogeneity is implemented across six brain regions:
//!
//! | Region         | Mechanism                                      | Effect on Order Flow                    |
//! |----------------|------------------------------------------------|-----------------------------------------|
//! | Thalamus       | Modality attention weights randomized           | Different signal weighting per instance |
//! | Hypothalamus   | Risk setpoints drawn from distributions         | Different position sizing / tolerance   |
//! | Basal Ganglia  | Dopamine sensitivity & discount factors varied  | Different action selection thresholds   |
//! | Prefrontal     | LTN axiom weights & hedge params varied         | Different constraint priorities          |
//! | Hippocampus    | Replay priorities & consolidation varied        | Different learning from same experience |
//! | Cerebellum     | Execution timing models varied                  | Different order placement patterns      |
//!
//! # Usage
//!
//! ```rust,no_run
//! use janus_neuromorphic::integration::heterogeneity::{
//!     HeterogeneityConfig, HeterogeneityProfile,
//! };
//!
//! let config = HeterogeneityConfig::default();
//! let profile = HeterogeneityProfile::from_instance_id("prod-janus-42", &config);
//!
//! // Use region-specific params to configure each brain region
//! let thalamus_params = &profile.thalamus;
//! let basal_ganglia_params = &profile.basal_ganglia;
//! ```
//!
//! # Determinism
//!
//! Profiles are generated deterministically from the `instance_id` via a
//! seeded PRNG (`SmallRng`, which uses Xoshiro256++ on 64-bit platforms).
//! The same `instance_id` always produces the same profile, ensuring
//! reproducibility across restarts while maintaining diversity across
//! deployments.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration — controls the *range* of heterogeneity per region
// ---------------------------------------------------------------------------

/// Master configuration governing the strength and bounds of engineered
/// heterogeneity across all brain regions.
///
/// Each field specifies the perturbation range as a fraction of the default
/// value (e.g., `thalamus_weight_range = 0.3` means weights vary ±30% from
/// their defaults).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneityConfig {
    /// Whether heterogeneity is enabled at all.
    pub enabled: bool,

    // -- Thalamus ----------------------------------------------------------
    /// Perturbation range for modality attention weights (fraction of default).
    pub thalamus_weight_range: f64,
    /// Perturbation range for cross-attention temperature.
    pub thalamus_temperature_range: f64,

    // -- Hypothalamus ------------------------------------------------------
    /// Perturbation range for risk appetite setpoint.
    pub hypothalamus_risk_setpoint_range: f64,
    /// Perturbation range for position sizing aggressiveness.
    pub hypothalamus_position_sizing_range: f64,
    /// Perturbation range for drawdown threshold.
    pub hypothalamus_drawdown_range: f64,

    // -- Basal Ganglia -----------------------------------------------------
    /// Perturbation range for dopamine sensitivity.
    pub basal_ganglia_dopamine_range: f64,
    /// Perturbation range for Go/NoGo threshold.
    pub basal_ganglia_threshold_range: f64,
    /// Perturbation range for temporal discount factor.
    pub basal_ganglia_discount_range: f64,
    /// Perturbation range for action selection temperature.
    pub basal_ganglia_temperature_range: f64,

    // -- Prefrontal --------------------------------------------------------
    /// Perturbation range for LTN axiom weights.
    pub prefrontal_axiom_weight_range: f64,
    /// Perturbation range for fuzzy logic violation threshold.
    pub prefrontal_violation_threshold_range: f64,

    // -- Hippocampus -------------------------------------------------------
    /// Perturbation range for replay priority exponent (alpha).
    pub hippocampus_alpha_range: f64,
    /// Perturbation range for importance sampling beta.
    pub hippocampus_beta_range: f64,
    /// Perturbation range for consolidation threshold.
    pub hippocampus_consolidation_range: f64,

    // -- Cerebellum --------------------------------------------------------
    /// Perturbation range for Almgren-Chriss risk aversion (lambda).
    pub cerebellum_lambda_range: f64,
    /// Perturbation range for temporary impact coefficient (eta).
    pub cerebellum_eta_range: f64,
    /// Perturbation range for PID controller gains.
    pub cerebellum_pid_range: f64,
}

impl Default for HeterogeneityConfig {
    fn default() -> Self {
        Self {
            enabled: true,

            // Thalamus: moderate variation in signal weighting
            thalamus_weight_range: 0.25,
            thalamus_temperature_range: 0.20,

            // Hypothalamus: moderate variation in risk appetite
            hypothalamus_risk_setpoint_range: 0.20,
            hypothalamus_position_sizing_range: 0.25,
            hypothalamus_drawdown_range: 0.15,

            // Basal Ganglia: significant variation in action selection
            basal_ganglia_dopamine_range: 0.30,
            basal_ganglia_threshold_range: 0.20,
            basal_ganglia_discount_range: 0.15,
            basal_ganglia_temperature_range: 0.25,

            // Prefrontal: moderate variation in constraint priorities
            prefrontal_axiom_weight_range: 0.20,
            prefrontal_violation_threshold_range: 0.15,

            // Hippocampus: moderate variation in learning dynamics
            hippocampus_alpha_range: 0.25,
            hippocampus_beta_range: 0.15,
            hippocampus_consolidation_range: 0.20,

            // Cerebellum: moderate variation in execution parameters
            cerebellum_lambda_range: 0.25,
            cerebellum_eta_range: 0.20,
            cerebellum_pid_range: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-region heterogeneity parameters
// ---------------------------------------------------------------------------

/// Thalamus heterogeneity: modality attention weights and gating parameters.
///
/// Controls how this instance weights different data modalities (order book,
/// price, volume, sentiment), producing different signal emphasis per instance.
#[derive(Debug, Clone)]
pub struct ThalamicHeterogeneity {
    /// Weight multiplier for order book modality (1.0 = default).
    pub orderbook_weight: f64,
    /// Weight multiplier for price modality.
    pub price_weight: f64,
    /// Weight multiplier for volume modality.
    pub volume_weight: f64,
    /// Weight multiplier for sentiment modality.
    pub sentiment_weight: f64,
    /// Cross-attention temperature scaling.
    pub attention_temperature: f64,
    /// EMA decay for attention smoothing.
    pub ema_decay: f64,
}

impl Default for ThalamicHeterogeneity {
    fn default() -> Self {
        Self {
            orderbook_weight: 1.0,
            price_weight: 1.0,
            volume_weight: 1.0,
            sentiment_weight: 1.0,
            attention_temperature: 1.0,
            ema_decay: 0.1,
        }
    }
}

/// Hypothalamic heterogeneity: risk setpoints and homeostatic parameters.
///
/// Controls the "personality" of each instance's risk tolerance, making some
/// instances naturally more aggressive and others more conservative.
#[derive(Debug, Clone)]
pub struct HypothalamicHeterogeneity {
    /// Risk appetite setpoint (0.0 = maximally conservative, 1.0 = maximally aggressive).
    pub risk_appetite_setpoint: f64,
    /// Position sizing aggressiveness multiplier.
    pub position_sizing_multiplier: f64,
    /// Maximum drawdown threshold before protective action.
    pub drawdown_threshold: f64,
    /// Profit drive intensity.
    pub profit_drive: f64,
    /// Safety drive intensity.
    pub safety_drive: f64,
    /// Kelly fraction multiplier (fractional Kelly scaling).
    pub kelly_fraction: f64,
}

impl Default for HypothalamicHeterogeneity {
    fn default() -> Self {
        Self {
            risk_appetite_setpoint: 0.5,
            position_sizing_multiplier: 1.0,
            drawdown_threshold: 0.10,
            profit_drive: 0.5,
            safety_drive: 0.8,
            kelly_fraction: 0.5,
        }
    }
}

/// Basal ganglia heterogeneity: dopamine sensitivity and action selection.
///
/// Controls how aggressively each instance responds to reward signals and how
/// it balances Go/NoGo pathway competition.
#[derive(Debug, Clone)]
pub struct BasalGangliaHeterogeneity {
    /// Dopamine sensitivity for Go pathway (higher = more reward-responsive).
    pub dopamine_sensitivity: f64,
    /// Base threshold for Go signal action initiation.
    pub go_threshold: f64,
    /// Inhibition weight (lambda) for NoGo pathway.
    pub nogo_inhibition_weight: f64,
    /// Temporal discount factor (gamma) for reward evaluation.
    pub discount_factor: f64,
    /// Action selection temperature (lower = more exploitative).
    pub selection_temperature: f64,
    /// Learning rate for value updates.
    pub learning_rate: f64,
    /// Signal decay rate.
    pub decay_rate: f64,
}

impl Default for BasalGangliaHeterogeneity {
    fn default() -> Self {
        Self {
            dopamine_sensitivity: 1.0,
            go_threshold: 0.5,
            nogo_inhibition_weight: 1.0,
            discount_factor: 0.99,
            selection_temperature: 1.0,
            learning_rate: 0.1,
            decay_rate: 0.95,
        }
    }
}

/// Prefrontal heterogeneity: LTN axiom weights and constraint priorities.
///
/// Controls which constraints each instance prioritizes, producing different
/// compliance "personalities" (e.g., one instance might weight risk limits
/// more heavily while another emphasizes position limits).
#[derive(Debug, Clone)]
pub struct PrefrontalHeterogeneity {
    /// Weight multiplier for risk limit axioms.
    pub risk_limit_weight: f64,
    /// Weight multiplier for position limit axioms.
    pub position_limit_weight: f64,
    /// Weight multiplier for impact/slippage constraints.
    pub impact_constraint_weight: f64,
    /// Weight multiplier for wash sale constraints.
    pub wash_sale_weight: f64,
    /// Fuzzy logic violation threshold.
    pub violation_threshold: f64,
    /// Hedge concentration exponent (for `very` hedge).
    pub hedge_concentration: f64,
}

impl Default for PrefrontalHeterogeneity {
    fn default() -> Self {
        Self {
            risk_limit_weight: 1.0,
            position_limit_weight: 1.0,
            impact_constraint_weight: 1.0,
            wash_sale_weight: 1.0,
            violation_threshold: 0.5,
            hedge_concentration: 2.0,
        }
    }
}

/// Hippocampal heterogeneity: replay priorities and consolidation dynamics.
///
/// Controls how each instance learns from experience, biasing replay toward
/// different types of events and consolidating at different rates.
#[derive(Debug, Clone)]
pub struct HippocampalHeterogeneity {
    /// Priority exponent (alpha) for prioritized experience replay.
    /// Higher = more aggressive prioritization by TD error.
    pub replay_alpha: f64,
    /// Importance sampling correction (beta initial value).
    pub replay_beta_initial: f64,
    /// Consolidation threshold (minimum replay score for promotion).
    pub consolidation_threshold: f64,
    /// Emotional salience multiplier for consolidation priority.
    pub emotional_salience_weight: f64,
    /// SWR replay rate multiplier.
    pub replay_rate_multiplier: f64,
    /// Schema learning rate for neocortical consolidation.
    pub schema_learning_rate: f64,
}

impl Default for HippocampalHeterogeneity {
    fn default() -> Self {
        Self {
            replay_alpha: 0.6,
            replay_beta_initial: 0.4,
            consolidation_threshold: 0.5,
            emotional_salience_weight: 1.0,
            replay_rate_multiplier: 1.0,
            schema_learning_rate: 0.01,
        }
    }
}

/// Cerebellar heterogeneity: execution timing and market impact models.
///
/// Controls how each instance executes trades, varying the aggressiveness of
/// optimal execution trajectories and error correction dynamics.
#[derive(Debug, Clone)]
pub struct CerebellarHeterogeneity {
    /// Almgren-Chriss risk aversion parameter (lambda).
    pub ac_lambda: f64,
    /// Temporary market impact coefficient (eta).
    pub ac_eta: f64,
    /// Permanent market impact coefficient (gamma).
    pub ac_gamma: f64,
    /// PID proportional gain multiplier.
    pub pid_kp_multiplier: f64,
    /// PID integral gain multiplier.
    pub pid_ki_multiplier: f64,
    /// PID derivative gain multiplier.
    pub pid_kd_multiplier: f64,
}

impl Default for CerebellarHeterogeneity {
    fn default() -> Self {
        Self {
            ac_lambda: 1.0,
            ac_eta: 0.01,
            ac_gamma: 0.005,
            pid_kp_multiplier: 1.0,
            pid_ki_multiplier: 1.0,
            pid_kd_multiplier: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Composite profile
// ---------------------------------------------------------------------------

/// A complete heterogeneity profile for a single JANUS instance.
///
/// Generated deterministically from the `instance_id` using a seeded PRNG,
/// ensuring reproducibility across restarts while maintaining diversity across
/// deployments. Each brain region receives its own parameter perturbations.
#[derive(Debug, Clone)]
pub struct HeterogeneityProfile {
    /// The instance identifier used to generate this profile.
    pub instance_id: String,

    /// 64-bit seed derived from the instance ID (for reproducibility).
    pub seed: u64,

    /// Thalamus: modality attention weights and gating.
    pub thalamus: ThalamicHeterogeneity,

    /// Hypothalamus: risk appetite and homeostasis setpoints.
    pub hypothalamus: HypothalamicHeterogeneity,

    /// Basal Ganglia: dopamine sensitivity and action selection.
    pub basal_ganglia: BasalGangliaHeterogeneity,

    /// Prefrontal: LTN axiom weights and constraint priorities.
    pub prefrontal: PrefrontalHeterogeneity,

    /// Hippocampus: replay priorities and consolidation dynamics.
    pub hippocampus: HippocampalHeterogeneity,

    /// Cerebellum: execution timing and impact model parameters.
    pub cerebellum: CerebellarHeterogeneity,
}

impl HeterogeneityProfile {
    /// Generate a heterogeneity profile from an instance ID.
    ///
    /// Uses a deterministic hash of the `instance_id` to seed a PRNG, then
    /// draws each parameter from a range centered on the default value.
    ///
    /// The same `instance_id` always produces the same profile.
    pub fn from_instance_id(instance_id: &str, config: &HeterogeneityConfig) -> Self {
        let seed = Self::hash_instance_id(instance_id);
        let mut rng = SmallRng::seed_from_u64(seed);

        if !config.enabled {
            return Self {
                instance_id: instance_id.to_string(),
                seed,
                thalamus: ThalamicHeterogeneity::default(),
                hypothalamus: HypothalamicHeterogeneity::default(),
                basal_ganglia: BasalGangliaHeterogeneity::default(),
                prefrontal: PrefrontalHeterogeneity::default(),
                hippocampus: HippocampalHeterogeneity::default(),
                cerebellum: CerebellarHeterogeneity::default(),
            };
        }

        let thalamus = Self::gen_thalamus(&mut rng, config);
        let hypothalamus = Self::gen_hypothalamus(&mut rng, config);
        let basal_ganglia = Self::gen_basal_ganglia(&mut rng, config);
        let prefrontal = Self::gen_prefrontal(&mut rng, config);
        let hippocampus = Self::gen_hippocampus(&mut rng, config);
        let cerebellum = Self::gen_cerebellum(&mut rng, config);

        Self {
            instance_id: instance_id.to_string(),
            seed,
            thalamus,
            hypothalamus,
            basal_ganglia,
            prefrontal,
            hippocampus,
            cerebellum,
        }
    }

    /// Compute the pairwise distance between two profiles.
    ///
    /// Returns a normalized distance in [0, 1] where 0 = identical profiles
    /// and 1 = maximally different. This metric is used for cross-instance
    /// correlation monitoring.
    pub fn distance(&self, other: &Self) -> f64 {
        let v1 = self.to_feature_vector();
        let v2 = other.to_feature_vector();

        assert_eq!(v1.len(), v2.len());

        let sq_sum: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let d = (sq_sum / v1.len() as f64).sqrt();

        // Normalize: parameters are roughly in [0, 2] range, so max distance ≈ 2.0
        (d / 2.0).min(1.0)
    }

    /// Convert the profile to a flat feature vector for distance computation.
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            // Thalamus (6)
            self.thalamus.orderbook_weight,
            self.thalamus.price_weight,
            self.thalamus.volume_weight,
            self.thalamus.sentiment_weight,
            self.thalamus.attention_temperature,
            self.thalamus.ema_decay * 10.0, // scale up for distance computation
            // Hypothalamus (6)
            self.hypothalamus.risk_appetite_setpoint,
            self.hypothalamus.position_sizing_multiplier,
            self.hypothalamus.drawdown_threshold * 10.0,
            self.hypothalamus.profit_drive,
            self.hypothalamus.safety_drive,
            self.hypothalamus.kelly_fraction,
            // Basal Ganglia (7)
            self.basal_ganglia.dopamine_sensitivity,
            self.basal_ganglia.go_threshold,
            self.basal_ganglia.nogo_inhibition_weight,
            self.basal_ganglia.discount_factor,
            self.basal_ganglia.selection_temperature,
            self.basal_ganglia.learning_rate * 10.0,
            self.basal_ganglia.decay_rate,
            // Prefrontal (6)
            self.prefrontal.risk_limit_weight,
            self.prefrontal.position_limit_weight,
            self.prefrontal.impact_constraint_weight,
            self.prefrontal.wash_sale_weight,
            self.prefrontal.violation_threshold,
            self.prefrontal.hedge_concentration,
            // Hippocampus (6)
            self.hippocampus.replay_alpha,
            self.hippocampus.replay_beta_initial,
            self.hippocampus.consolidation_threshold,
            self.hippocampus.emotional_salience_weight,
            self.hippocampus.replay_rate_multiplier,
            self.hippocampus.schema_learning_rate * 100.0,
            // Cerebellum (6)
            self.cerebellum.ac_lambda,
            self.cerebellum.ac_eta * 100.0,
            self.cerebellum.ac_gamma * 200.0,
            self.cerebellum.pid_kp_multiplier,
            self.cerebellum.pid_ki_multiplier,
            self.cerebellum.pid_kd_multiplier,
        ]
    }

    /// Produce a human-readable summary of the profile's deviations from defaults.
    pub fn summary(&self) -> String {
        let defaults = HeterogeneityProfile {
            instance_id: String::new(),
            seed: 0,
            thalamus: ThalamicHeterogeneity::default(),
            hypothalamus: HypothalamicHeterogeneity::default(),
            basal_ganglia: BasalGangliaHeterogeneity::default(),
            prefrontal: PrefrontalHeterogeneity::default(),
            hippocampus: HippocampalHeterogeneity::default(),
            cerebellum: CerebellarHeterogeneity::default(),
        };

        let mut lines = vec![format!(
            "Heterogeneity Profile: {} (seed: {})",
            self.instance_id, self.seed
        )];
        lines.push("=".repeat(60));

        // Thalamus
        lines.push("Thalamus (Modality Attention):".to_string());
        lines.push(format!(
            "  orderbook_weight:     {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.thalamus.orderbook_weight,
            defaults.thalamus.orderbook_weight,
            pct_delta(
                self.thalamus.orderbook_weight,
                defaults.thalamus.orderbook_weight
            )
        ));
        lines.push(format!(
            "  price_weight:         {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.thalamus.price_weight,
            defaults.thalamus.price_weight,
            pct_delta(self.thalamus.price_weight, defaults.thalamus.price_weight)
        ));
        lines.push(format!(
            "  volume_weight:        {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.thalamus.volume_weight,
            defaults.thalamus.volume_weight,
            pct_delta(self.thalamus.volume_weight, defaults.thalamus.volume_weight)
        ));
        lines.push(format!(
            "  sentiment_weight:     {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.thalamus.sentiment_weight,
            defaults.thalamus.sentiment_weight,
            pct_delta(
                self.thalamus.sentiment_weight,
                defaults.thalamus.sentiment_weight
            )
        ));
        lines.push(format!(
            "  attention_temperature:{:.3} (default: {:.3})",
            self.thalamus.attention_temperature, defaults.thalamus.attention_temperature,
        ));

        // Hypothalamus
        lines.push(String::new());
        lines.push("Hypothalamus (Risk Setpoints):".to_string());
        lines.push(format!(
            "  risk_appetite:        {:.3} (default: {:.3})",
            self.hypothalamus.risk_appetite_setpoint, defaults.hypothalamus.risk_appetite_setpoint,
        ));
        lines.push(format!(
            "  position_sizing:      {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.hypothalamus.position_sizing_multiplier,
            defaults.hypothalamus.position_sizing_multiplier,
            pct_delta(
                self.hypothalamus.position_sizing_multiplier,
                defaults.hypothalamus.position_sizing_multiplier,
            )
        ));
        lines.push(format!(
            "  drawdown_threshold:   {:.3} (default: {:.3})",
            self.hypothalamus.drawdown_threshold, defaults.hypothalamus.drawdown_threshold,
        ));

        // Basal Ganglia
        lines.push(String::new());
        lines.push("Basal Ganglia (Action Selection):".to_string());
        lines.push(format!(
            "  dopamine_sensitivity: {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.basal_ganglia.dopamine_sensitivity,
            defaults.basal_ganglia.dopamine_sensitivity,
            pct_delta(
                self.basal_ganglia.dopamine_sensitivity,
                defaults.basal_ganglia.dopamine_sensitivity,
            )
        ));
        lines.push(format!(
            "  go_threshold:         {:.3} (default: {:.3})",
            self.basal_ganglia.go_threshold, defaults.basal_ganglia.go_threshold,
        ));
        lines.push(format!(
            "  discount_factor:      {:.4} (default: {:.4})",
            self.basal_ganglia.discount_factor, defaults.basal_ganglia.discount_factor,
        ));
        lines.push(format!(
            "  selection_temperature:{:.3} (default: {:.3})",
            self.basal_ganglia.selection_temperature, defaults.basal_ganglia.selection_temperature,
        ));

        // Prefrontal
        lines.push(String::new());
        lines.push("Prefrontal (Constraint Priorities):".to_string());
        lines.push(format!(
            "  risk_limit_weight:    {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.prefrontal.risk_limit_weight,
            defaults.prefrontal.risk_limit_weight,
            pct_delta(
                self.prefrontal.risk_limit_weight,
                defaults.prefrontal.risk_limit_weight,
            )
        ));
        lines.push(format!(
            "  position_limit_weight:{:.3} (default: {:.3}, delta: {:+.1}%)",
            self.prefrontal.position_limit_weight,
            defaults.prefrontal.position_limit_weight,
            pct_delta(
                self.prefrontal.position_limit_weight,
                defaults.prefrontal.position_limit_weight,
            )
        ));

        // Hippocampus
        lines.push(String::new());
        lines.push("Hippocampus (Learning Dynamics):".to_string());
        lines.push(format!(
            "  replay_alpha:         {:.3} (default: {:.3})",
            self.hippocampus.replay_alpha, defaults.hippocampus.replay_alpha,
        ));
        lines.push(format!(
            "  replay_beta_initial:  {:.3} (default: {:.3})",
            self.hippocampus.replay_beta_initial, defaults.hippocampus.replay_beta_initial,
        ));
        lines.push(format!(
            "  consolidation_thresh: {:.3} (default: {:.3})",
            self.hippocampus.consolidation_threshold, defaults.hippocampus.consolidation_threshold,
        ));

        // Cerebellum
        lines.push(String::new());
        lines.push("Cerebellum (Execution Timing):".to_string());
        lines.push(format!(
            "  ac_lambda:            {:.3} (default: {:.3}, delta: {:+.1}%)",
            self.cerebellum.ac_lambda,
            defaults.cerebellum.ac_lambda,
            pct_delta(self.cerebellum.ac_lambda, defaults.cerebellum.ac_lambda)
        ));
        lines.push(format!(
            "  ac_eta:               {:.4} (default: {:.4})",
            self.cerebellum.ac_eta, defaults.cerebellum.ac_eta,
        ));
        lines.push(format!(
            "  pid_kp_multiplier:    {:.3} (default: {:.3})",
            self.cerebellum.pid_kp_multiplier, defaults.cerebellum.pid_kp_multiplier,
        ));

        lines.join("\n")
    }

    // -------------------------------------------------------------------
    // Private: instance ID hashing
    // -------------------------------------------------------------------

    /// Hash an instance ID string to a 64-bit seed using FNV-1a.
    fn hash_instance_id(id: &str) -> u64 {
        // FNV-1a hash — simple, deterministic, no dependencies
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in id.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    // -------------------------------------------------------------------
    // Private: per-region generation
    // -------------------------------------------------------------------

    fn gen_thalamus(rng: &mut SmallRng, config: &HeterogeneityConfig) -> ThalamicHeterogeneity {
        let defaults = ThalamicHeterogeneity::default();
        let wr = config.thalamus_weight_range;
        let tr = config.thalamus_temperature_range;

        ThalamicHeterogeneity {
            orderbook_weight: perturb(rng, defaults.orderbook_weight, wr, 0.3, 2.0),
            price_weight: perturb(rng, defaults.price_weight, wr, 0.3, 2.0),
            volume_weight: perturb(rng, defaults.volume_weight, wr, 0.3, 2.0),
            sentiment_weight: perturb(rng, defaults.sentiment_weight, wr, 0.2, 2.0),
            attention_temperature: perturb(rng, defaults.attention_temperature, tr, 0.5, 2.0),
            ema_decay: perturb(rng, defaults.ema_decay, 0.15, 0.01, 0.5),
        }
    }

    fn gen_hypothalamus(
        rng: &mut SmallRng,
        config: &HeterogeneityConfig,
    ) -> HypothalamicHeterogeneity {
        let defaults = HypothalamicHeterogeneity::default();
        let rr = config.hypothalamus_risk_setpoint_range;
        let pr = config.hypothalamus_position_sizing_range;
        let dr = config.hypothalamus_drawdown_range;

        HypothalamicHeterogeneity {
            risk_appetite_setpoint: perturb(rng, defaults.risk_appetite_setpoint, rr, 0.1, 0.9),
            position_sizing_multiplier: perturb(
                rng,
                defaults.position_sizing_multiplier,
                pr,
                0.5,
                1.5,
            ),
            drawdown_threshold: perturb(rng, defaults.drawdown_threshold, dr, 0.05, 0.20),
            profit_drive: perturb(rng, defaults.profit_drive, 0.15, 0.2, 0.8),
            safety_drive: perturb(rng, defaults.safety_drive, 0.15, 0.5, 1.0),
            kelly_fraction: perturb(rng, defaults.kelly_fraction, 0.20, 0.2, 0.8),
        }
    }

    fn gen_basal_ganglia(
        rng: &mut SmallRng,
        config: &HeterogeneityConfig,
    ) -> BasalGangliaHeterogeneity {
        let defaults = BasalGangliaHeterogeneity::default();
        let dr = config.basal_ganglia_dopamine_range;
        let tr = config.basal_ganglia_threshold_range;
        let disc = config.basal_ganglia_discount_range;
        let temp = config.basal_ganglia_temperature_range;

        BasalGangliaHeterogeneity {
            dopamine_sensitivity: perturb(rng, defaults.dopamine_sensitivity, dr, 0.4, 1.8),
            go_threshold: perturb(rng, defaults.go_threshold, tr, 0.2, 0.8),
            nogo_inhibition_weight: perturb(rng, defaults.nogo_inhibition_weight, 0.20, 0.5, 1.5),
            discount_factor: perturb(rng, defaults.discount_factor, disc, 0.90, 0.999),
            selection_temperature: perturb(rng, defaults.selection_temperature, temp, 0.3, 2.0),
            learning_rate: perturb(rng, defaults.learning_rate, 0.20, 0.01, 0.3),
            decay_rate: perturb(rng, defaults.decay_rate, 0.10, 0.80, 0.99),
        }
    }

    fn gen_prefrontal(rng: &mut SmallRng, config: &HeterogeneityConfig) -> PrefrontalHeterogeneity {
        let defaults = PrefrontalHeterogeneity::default();
        let aw = config.prefrontal_axiom_weight_range;
        let vt = config.prefrontal_violation_threshold_range;

        PrefrontalHeterogeneity {
            risk_limit_weight: perturb(rng, defaults.risk_limit_weight, aw, 0.5, 1.5),
            position_limit_weight: perturb(rng, defaults.position_limit_weight, aw, 0.5, 1.5),
            impact_constraint_weight: perturb(rng, defaults.impact_constraint_weight, aw, 0.5, 1.5),
            wash_sale_weight: perturb(rng, defaults.wash_sale_weight, aw, 0.5, 1.5),
            violation_threshold: perturb(rng, defaults.violation_threshold, vt, 0.3, 0.7),
            hedge_concentration: perturb(rng, defaults.hedge_concentration, 0.15, 1.5, 2.5),
        }
    }

    fn gen_hippocampus(
        rng: &mut SmallRng,
        config: &HeterogeneityConfig,
    ) -> HippocampalHeterogeneity {
        let defaults = HippocampalHeterogeneity::default();
        let ar = config.hippocampus_alpha_range;
        let br = config.hippocampus_beta_range;
        let cr = config.hippocampus_consolidation_range;

        HippocampalHeterogeneity {
            replay_alpha: perturb(rng, defaults.replay_alpha, ar, 0.2, 1.0),
            replay_beta_initial: perturb(rng, defaults.replay_beta_initial, br, 0.2, 0.7),
            consolidation_threshold: perturb(rng, defaults.consolidation_threshold, cr, 0.2, 0.8),
            emotional_salience_weight: perturb(
                rng,
                defaults.emotional_salience_weight,
                0.20,
                0.5,
                1.5,
            ),
            replay_rate_multiplier: perturb(rng, defaults.replay_rate_multiplier, 0.20, 0.5, 1.5),
            schema_learning_rate: perturb(rng, defaults.schema_learning_rate, 0.25, 0.001, 0.05),
        }
    }

    fn gen_cerebellum(rng: &mut SmallRng, config: &HeterogeneityConfig) -> CerebellarHeterogeneity {
        let defaults = CerebellarHeterogeneity::default();
        let lr = config.cerebellum_lambda_range;
        let er = config.cerebellum_eta_range;
        let pr = config.cerebellum_pid_range;

        CerebellarHeterogeneity {
            ac_lambda: perturb(rng, defaults.ac_lambda, lr, 0.3, 2.0),
            ac_eta: perturb(rng, defaults.ac_eta, er, 0.001, 0.05),
            ac_gamma: perturb(rng, defaults.ac_gamma, 0.15, 0.001, 0.02),
            pid_kp_multiplier: perturb(rng, defaults.pid_kp_multiplier, pr, 0.5, 1.5),
            pid_ki_multiplier: perturb(rng, defaults.pid_ki_multiplier, pr, 0.5, 1.5),
            pid_kd_multiplier: perturb(rng, defaults.pid_kd_multiplier, pr, 0.5, 1.5),
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-Instance Correlation Monitor
// ---------------------------------------------------------------------------

/// Monitors correlation across multiple JANUS instances to detect emergent
/// crowding even when heterogeneity is enabled.
///
/// This implements the "active monitoring of cross-instance correlation"
/// mitigation described in the limitations section. The monitor tracks:
///
/// - Pairwise profile distances between registered instances
/// - Order flow correlation (when fed external data)
/// - Alerts when correlation exceeds thresholds
#[derive(Debug, Clone)]
pub struct CrossInstanceMonitor {
    /// Registered instance profiles.
    profiles: HashMap<String, HeterogeneityProfile>,
    /// Pairwise distance cache (key = sorted pair of instance IDs).
    distance_cache: HashMap<(String, String), f64>,
    /// Per-instance flow history (ring buffer of signed order flow values).
    flow_histories: HashMap<String, Vec<f64>>,
    /// Configuration.
    config: MonitorConfig,
}

/// Configuration for the cross-instance monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Minimum profile distance below which a warning is raised.
    pub min_distance_threshold: f64,
    /// Maximum acceptable order flow correlation.
    pub max_flow_correlation: f64,
    /// Window size for flow correlation estimation.
    pub correlation_window: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            min_distance_threshold: 0.05,
            max_flow_correlation: 0.60,
            correlation_window: 100,
        }
    }
}

/// Alert from the cross-instance monitor.
#[derive(Debug, Clone)]
pub struct CrowdingAlert {
    /// Instance ID pair triggering the alert.
    pub instance_a: String,
    pub instance_b: String,
    /// Alert severity (0.0 = informational, 1.0 = critical).
    pub severity: f64,
    /// Human-readable reason.
    pub reason: String,
    /// Profile distance between the pair.
    pub profile_distance: f64,
    /// Estimated order flow correlation (if available).
    pub flow_correlation: Option<f64>,
}

impl CrossInstanceMonitor {
    /// Create a new monitor with default configuration.
    pub fn new() -> Self {
        Self::with_config(MonitorConfig::default())
    }

    /// Create a monitor with custom configuration.
    pub fn with_config(config: MonitorConfig) -> Self {
        Self {
            profiles: HashMap::new(),
            distance_cache: HashMap::new(),
            flow_histories: HashMap::new(),
            config,
        }
    }

    /// Register a new instance profile.
    pub fn register(&mut self, profile: HeterogeneityProfile) {
        let id = profile.instance_id.clone();

        // Compute distances to all existing profiles
        for (existing_id, existing_profile) in &self.profiles {
            let dist = profile.distance(existing_profile);
            let key = Self::pair_key(&id, existing_id);
            self.distance_cache.insert(key, dist);
        }

        self.profiles.insert(id, profile);
    }

    /// Remove an instance from monitoring.
    pub fn deregister(&mut self, instance_id: &str) {
        self.profiles.remove(instance_id);
        self.distance_cache
            .retain(|(a, b), _| a != instance_id && b != instance_id);
        self.flow_histories.remove(instance_id);
    }

    /// Record an order flow observation for a specific instance.
    ///
    /// `signed_flow` is the net signed order flow (positive = buy, negative = sell)
    /// for a single tick. Flow values are stored per-instance and used to compute
    /// pairwise Pearson correlation between instances in [`check_alerts`].
    ///
    /// All instances should record flow at each tick so that their histories
    /// remain aligned (same index = same timestamp).
    pub fn record_flow(&mut self, instance_id: &str, signed_flow: f64) {
        if !self.profiles.contains_key(instance_id) {
            return;
        }
        let history = self
            .flow_histories
            .entry(instance_id.to_string())
            .or_default();
        history.push(signed_flow);
        if history.len() > self.config.correlation_window {
            history.remove(0);
        }
    }

    /// Check all registered instances for crowding alerts.
    ///
    /// Returns alerts for two conditions:
    /// 1. **Profile distance** below `min_distance_threshold` (static risk)
    /// 2. **Flow correlation** above `max_flow_correlation` (dynamic risk)
    pub fn check_alerts(&self) -> Vec<CrowdingAlert> {
        let mut alerts = Vec::new();

        for ((id_a, id_b), &dist) in &self.distance_cache {
            // Compute flow correlation for this pair if both have sufficient history
            let flow_corr = self.pairwise_flow_correlation(id_a, id_b);

            // Alert 1: Profile distance too small
            if dist < self.config.min_distance_threshold {
                let severity = 1.0 - (dist / self.config.min_distance_threshold);
                alerts.push(CrowdingAlert {
                    instance_a: id_a.clone(),
                    instance_b: id_b.clone(),
                    severity: severity.clamp(0.0, 1.0),
                    reason: format!(
                        "Profile distance ({:.4}) below threshold ({:.4})",
                        dist, self.config.min_distance_threshold
                    ),
                    profile_distance: dist,
                    flow_correlation: flow_corr,
                });
            }

            // Alert 2: Flow correlation too high
            if let Some(corr) = flow_corr {
                if corr > self.config.max_flow_correlation {
                    let severity = (corr - self.config.max_flow_correlation)
                        / (1.0 - self.config.max_flow_correlation);
                    alerts.push(CrowdingAlert {
                        instance_a: id_a.clone(),
                        instance_b: id_b.clone(),
                        severity: severity.clamp(0.0, 1.0),
                        reason: format!(
                            "Flow correlation ({:.4}) exceeds threshold ({:.4})",
                            corr, self.config.max_flow_correlation
                        ),
                        profile_distance: dist,
                        flow_correlation: Some(corr),
                    });
                }
            }
        }

        alerts
    }

    /// Compute Pearson correlation between the flow histories of two instances.
    ///
    /// Returns `None` if either instance has insufficient data (< 2 observations).
    fn pairwise_flow_correlation(&self, id_a: &str, id_b: &str) -> Option<f64> {
        let hist_a = self.flow_histories.get(id_a)?;
        let hist_b = self.flow_histories.get(id_b)?;

        // Use the overlapping tail of both histories (aligned by most recent)
        let n = hist_a.len().min(hist_b.len());
        if n < 2 {
            return None;
        }

        let a_slice = &hist_a[hist_a.len() - n..];
        let b_slice = &hist_b[hist_b.len() - n..];

        let mean_a: f64 = a_slice.iter().sum::<f64>() / n as f64;
        let mean_b: f64 = b_slice.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for i in 0..n {
            let da = a_slice[i] - mean_a;
            let db = b_slice[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-12 {
            return None;
        }

        Some(cov / denom)
    }

    /// Get the minimum pairwise distance among all registered instances.
    pub fn min_pairwise_distance(&self) -> Option<f64> {
        self.distance_cache.values().cloned().reduce(f64::min)
    }

    /// Get the mean pairwise distance among all registered instances.
    pub fn mean_pairwise_distance(&self) -> Option<f64> {
        if self.distance_cache.is_empty() {
            return None;
        }
        let sum: f64 = self.distance_cache.values().sum();
        Some(sum / self.distance_cache.len() as f64)
    }

    /// Number of registered instances.
    pub fn instance_count(&self) -> usize {
        self.profiles.len()
    }

    /// Get a registered profile by instance ID.
    pub fn profile(&self, instance_id: &str) -> Option<&HeterogeneityProfile> {
        self.profiles.get(instance_id)
    }

    // -- Private helpers ---------------------------------------------------

    fn pair_key(a: &str, b: &str) -> (String, String) {
        if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }
}

impl Default for CrossInstanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Re-randomization support
// ---------------------------------------------------------------------------

/// Re-randomize a profile's parameters while preserving the instance ID.
///
/// This implements periodic re-randomization as a crowding mitigation:
/// over time, evolutionary dynamics can cause heterogeneous populations to
/// converge. Periodic re-seeding combats this drift.
///
/// The `epoch` parameter is combined with the instance ID to produce a new
/// seed, so re-randomization is deterministic for a given (instance_id, epoch).
pub fn rerandomize_profile(
    profile: &HeterogeneityProfile,
    epoch: u64,
    config: &HeterogeneityConfig,
) -> HeterogeneityProfile {
    let composite_id = format!("{}::epoch={}", profile.instance_id, epoch);
    HeterogeneityProfile::from_instance_id(&composite_id, config)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Perturb a value by a random fraction within `[-range, +range]`, clamped
/// to `[min_val, max_val]`.
fn perturb(rng: &mut SmallRng, default: f64, range: f64, min_val: f64, max_val: f64) -> f64 {
    let delta: f64 = rng.random_range(-range..=range);
    let perturbed = default * (1.0 + delta);
    perturbed.clamp(min_val, max_val)
}

/// Compute percentage delta between actual and reference.
fn pct_delta(actual: f64, reference: f64) -> f64 {
    if reference.abs() < 1e-12 {
        0.0
    } else {
        (actual - reference) / reference * 100.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_profile_generation() {
        let config = HeterogeneityConfig::default();
        let p1 = HeterogeneityProfile::from_instance_id("test-instance-42", &config);
        let p2 = HeterogeneityProfile::from_instance_id("test-instance-42", &config);

        // Same ID → same seed
        assert_eq!(p1.seed, p2.seed);

        // Same ID → same parameters
        assert_eq!(p1.thalamus.orderbook_weight, p2.thalamus.orderbook_weight);
        assert_eq!(
            p1.basal_ganglia.dopamine_sensitivity,
            p2.basal_ganglia.dopamine_sensitivity
        );
        assert_eq!(
            p1.hypothalamus.risk_appetite_setpoint,
            p2.hypothalamus.risk_appetite_setpoint
        );
        assert_eq!(p1.cerebellum.ac_lambda, p2.cerebellum.ac_lambda);
    }

    #[test]
    fn test_different_ids_produce_different_profiles() {
        let config = HeterogeneityConfig::default();
        let p1 = HeterogeneityProfile::from_instance_id("alpha", &config);
        let p2 = HeterogeneityProfile::from_instance_id("beta", &config);

        assert_ne!(p1.seed, p2.seed);

        // Very unlikely all parameters are the same with different seeds
        let dist = p1.distance(&p2);
        assert!(
            dist > 0.0,
            "different IDs should produce different profiles"
        );
    }

    #[test]
    fn test_disabled_heterogeneity_returns_defaults() {
        let config = HeterogeneityConfig {
            enabled: false,
            ..Default::default()
        };
        let profile = HeterogeneityProfile::from_instance_id("any-id", &config);

        let defaults = ThalamicHeterogeneity::default();
        assert_eq!(profile.thalamus.orderbook_weight, defaults.orderbook_weight);
        assert_eq!(profile.thalamus.price_weight, defaults.price_weight);
    }

    #[test]
    fn test_parameters_within_bounds() {
        let config = HeterogeneityConfig::default();

        // Generate many profiles and check bounds
        for i in 0..100 {
            let profile = HeterogeneityProfile::from_instance_id(&format!("inst-{}", i), &config);

            // Thalamus weights should be positive
            assert!(profile.thalamus.orderbook_weight > 0.0);
            assert!(profile.thalamus.price_weight > 0.0);
            assert!(profile.thalamus.volume_weight > 0.0);
            assert!(profile.thalamus.sentiment_weight > 0.0);

            // Hypothalamus risk should be in [0.1, 0.9]
            assert!(profile.hypothalamus.risk_appetite_setpoint >= 0.1);
            assert!(profile.hypothalamus.risk_appetite_setpoint <= 0.9);

            // BG discount factor should be in [0.90, 0.999]
            assert!(profile.basal_ganglia.discount_factor >= 0.90);
            assert!(profile.basal_ganglia.discount_factor <= 0.999);

            // Hippocampus alpha should be in [0.2, 1.0]
            assert!(profile.hippocampus.replay_alpha >= 0.2);
            assert!(profile.hippocampus.replay_alpha <= 1.0);
        }
    }

    #[test]
    fn test_profile_distance_self_is_zero() {
        let config = HeterogeneityConfig::default();
        let p = HeterogeneityProfile::from_instance_id("self-test", &config);
        assert_eq!(p.distance(&p), 0.0);
    }

    #[test]
    fn test_profile_distance_symmetry() {
        let config = HeterogeneityConfig::default();
        let p1 = HeterogeneityProfile::from_instance_id("a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("b", &config);

        let d12 = p1.distance(&p2);
        let d21 = p2.distance(&p1);
        assert!((d12 - d21).abs() < 1e-12, "distance should be symmetric");
    }

    #[test]
    fn test_sufficient_diversity_across_100_instances() {
        let config = HeterogeneityConfig::default();
        let profiles: Vec<_> = (0..100)
            .map(|i| HeterogeneityProfile::from_instance_id(&format!("prod-{}", i), &config))
            .collect();

        // Compute min pairwise distance
        let mut min_dist = f64::MAX;
        let mut total_dist = 0.0;
        let mut count = 0u64;

        for i in 0..profiles.len() {
            for j in (i + 1)..profiles.len() {
                let d = profiles[i].distance(&profiles[j]);
                min_dist = min_dist.min(d);
                total_dist += d;
                count += 1;
            }
        }

        let mean_dist = total_dist / count as f64;

        // With 100 instances and default ranges, we expect reasonable diversity
        assert!(
            min_dist > 0.01,
            "minimum pairwise distance ({:.4}) should be > 0.01",
            min_dist
        );
        assert!(
            mean_dist > 0.05,
            "mean pairwise distance ({:.4}) should be > 0.05",
            mean_dist
        );
    }

    #[test]
    fn test_cross_instance_monitor() {
        let config = HeterogeneityConfig::default();
        let mut monitor = CrossInstanceMonitor::new();

        let p1 = HeterogeneityProfile::from_instance_id("inst-1", &config);
        let p2 = HeterogeneityProfile::from_instance_id("inst-2", &config);
        let p3 = HeterogeneityProfile::from_instance_id("inst-3", &config);

        monitor.register(p1);
        monitor.register(p2);
        monitor.register(p3);

        assert_eq!(monitor.instance_count(), 3);

        // All distances should be non-zero
        let min_dist = monitor.min_pairwise_distance().unwrap();
        assert!(min_dist > 0.0);

        let mean_dist = monitor.mean_pairwise_distance().unwrap();
        assert!(mean_dist > 0.0);
        assert!(mean_dist >= min_dist);
    }

    #[test]
    fn test_monitor_identical_profiles_trigger_alert() {
        let config = HeterogeneityConfig {
            enabled: false, // disabled = all defaults = identical
            ..Default::default()
        };
        let monitor_config = MonitorConfig {
            min_distance_threshold: 0.01,
            ..Default::default()
        };

        let mut monitor = CrossInstanceMonitor::with_config(monitor_config);

        let p1 = HeterogeneityProfile::from_instance_id("clone-a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("clone-b", &config);

        monitor.register(p1);
        monitor.register(p2);

        let alerts = monitor.check_alerts();
        assert!(
            !alerts.is_empty(),
            "identical profiles should trigger a crowding alert"
        );
        assert!(alerts[0].severity > 0.0);
    }

    #[test]
    fn test_rerandomize_produces_different_profile() {
        let config = HeterogeneityConfig::default();
        let original = HeterogeneityProfile::from_instance_id("stable-inst", &config);
        let rerand = rerandomize_profile(&original, 1, &config);

        // Same instance ID but different seed (due to epoch)
        assert_eq!(rerand.instance_id, "stable-inst::epoch=1");
        assert_ne!(rerand.seed, original.seed);

        // Parameters should differ
        let dist = original.distance(&rerand);
        assert!(dist > 0.0, "re-randomized profile should differ");
    }

    #[test]
    fn test_rerandomize_is_deterministic() {
        let config = HeterogeneityConfig::default();
        let original = HeterogeneityProfile::from_instance_id("stable-inst", &config);

        let r1 = rerandomize_profile(&original, 5, &config);
        let r2 = rerandomize_profile(&original, 5, &config);

        assert_eq!(r1.seed, r2.seed);
        assert_eq!(r1.thalamus.orderbook_weight, r2.thalamus.orderbook_weight);
    }

    #[test]
    fn test_feature_vector_length() {
        let config = HeterogeneityConfig::default();
        let p = HeterogeneityProfile::from_instance_id("test", &config);
        let v = p.to_feature_vector();

        // 6 + 6 + 7 + 6 + 6 + 6 = 37
        assert_eq!(v.len(), 37, "feature vector should have 37 dimensions");
    }

    #[test]
    fn test_summary_contains_all_regions() {
        let config = HeterogeneityConfig::default();
        let p = HeterogeneityProfile::from_instance_id("summary-test", &config);
        let summary = p.summary();

        assert!(summary.contains("Thalamus"));
        assert!(summary.contains("Hypothalamus"));
        assert!(summary.contains("Basal Ganglia"));
        assert!(summary.contains("Prefrontal"));
        assert!(summary.contains("Hippocampus"));
        assert!(summary.contains("Cerebellum"));
    }

    #[test]
    fn test_monitor_deregister() {
        let config = HeterogeneityConfig::default();
        let mut monitor = CrossInstanceMonitor::new();

        let p1 = HeterogeneityProfile::from_instance_id("inst-a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("inst-b", &config);

        monitor.register(p1);
        monitor.register(p2);
        assert_eq!(monitor.instance_count(), 2);

        monitor.deregister("inst-a");
        assert_eq!(monitor.instance_count(), 1);
        assert!(monitor.profile("inst-a").is_none());
        assert!(monitor.profile("inst-b").is_some());

        // Distance cache should be cleaned
        assert!(monitor.min_pairwise_distance().is_none());
    }

    #[test]
    fn test_hash_consistency() {
        // Ensure the hash function is stable across runs
        let h1 = HeterogeneityProfile::hash_instance_id("test");
        let h2 = HeterogeneityProfile::hash_instance_id("test");
        assert_eq!(h1, h2);

        let h3 = HeterogeneityProfile::hash_instance_id("test2");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_perturb_function() {
        let mut rng = SmallRng::seed_from_u64(42);

        for _ in 0..1000 {
            let val = perturb(&mut rng, 1.0, 0.3, 0.5, 1.5);
            assert!(val >= 0.5, "perturbed value {:.4} below min", val);
            assert!(val <= 1.5, "perturbed value {:.4} above max", val);
        }
    }

    #[test]
    fn test_flow_correlation_high_correlation() {
        let config = HeterogeneityConfig::default();
        let monitor_config = MonitorConfig {
            max_flow_correlation: 0.80,
            correlation_window: 50,
            ..Default::default()
        };
        let mut monitor = CrossInstanceMonitor::with_config(monitor_config);

        let p1 = HeterogeneityProfile::from_instance_id("corr-a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("corr-b", &config);
        monitor.register(p1);
        monitor.register(p2);

        // Feed identical flow to both instances → correlation ≈ 1.0
        for i in 0..20 {
            let flow = (i as f64).sin();
            monitor.record_flow("corr-a", flow);
            monitor.record_flow("corr-b", flow);
        }

        let alerts = monitor.check_alerts();
        let corr_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.reason.contains("Flow correlation"))
            .collect();
        assert!(
            !corr_alerts.is_empty(),
            "identical flows should trigger a correlation alert"
        );
        assert!(corr_alerts[0].flow_correlation.unwrap() > 0.99);
    }

    #[test]
    fn test_flow_correlation_uncorrelated() {
        let config = HeterogeneityConfig::default();
        let monitor_config = MonitorConfig {
            max_flow_correlation: 0.80,
            correlation_window: 200,
            ..Default::default()
        };
        let mut monitor = CrossInstanceMonitor::with_config(monitor_config);

        let p1 = HeterogeneityProfile::from_instance_id("uncorr-a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("uncorr-b", &config);
        monitor.register(p1);
        monitor.register(p2);

        // Feed opposite flows → correlation ≈ -1.0
        for i in 0..100 {
            let flow = (i as f64 * 0.3).sin();
            monitor.record_flow("uncorr-a", flow);
            monitor.record_flow("uncorr-b", -flow);
        }

        let alerts = monitor.check_alerts();
        let corr_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.reason.contains("Flow correlation"))
            .collect();
        assert!(
            corr_alerts.is_empty(),
            "negatively correlated flows should not trigger correlation alert"
        );
    }

    #[test]
    fn test_flow_correlation_insufficient_data() {
        let config = HeterogeneityConfig::default();
        let mut monitor = CrossInstanceMonitor::new();

        let p1 = HeterogeneityProfile::from_instance_id("short-a", &config);
        let p2 = HeterogeneityProfile::from_instance_id("short-b", &config);
        monitor.register(p1);
        monitor.register(p2);

        // Only record one observation — insufficient for correlation
        monitor.record_flow("short-a", 1.0);

        let alerts = monitor.check_alerts();
        // Should have no correlation-based alerts
        let corr_alerts: Vec<_> = alerts
            .iter()
            .filter(|a| a.reason.contains("Flow correlation"))
            .collect();
        assert!(corr_alerts.is_empty());
    }

    #[test]
    fn test_record_flow_unregistered_instance_ignored() {
        let mut monitor = CrossInstanceMonitor::new();
        // Should not panic or store data for unregistered instances
        monitor.record_flow("ghost", 1.0);
        assert_eq!(monitor.instance_count(), 0);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = HeterogeneityConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deser: HeterogeneityConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.enabled, config.enabled);
        assert_eq!(deser.thalamus_weight_range, config.thalamus_weight_range);
        assert_eq!(
            deser.basal_ganglia_dopamine_range,
            config.basal_ganglia_dopamine_range
        );
    }

    #[test]
    fn test_monitor_config_serde_roundtrip() {
        let config = MonitorConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deser: MonitorConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.min_distance_threshold, config.min_distance_threshold);
        assert_eq!(deser.max_flow_correlation, config.max_flow_correlation);
        assert_eq!(deser.correlation_window, config.correlation_window);
    }
}
