//! Action inhibition (indirect pathway)
//!
//! Part of the Basal Ganglia region
//! Component: praxeological
//!
//! The NoGo pathway in the basal ganglia is responsible for inhibiting
//! actions that should not be taken. In trading:
//! - Prevents impulsive trades
//! - Blocks actions during high-risk conditions
//! - Implements stop-loss and risk limit enforcement
//! - Provides veto power over Go signals

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Reasons for action inhibition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InhibitionReason {
    /// Risk threshold exceeded
    RiskThreshold,
    /// Position limit reached
    PositionLimit,
    /// Loss limit triggered
    LossLimit,
    /// Volatility too high
    HighVolatility,
    /// Low liquidity conditions
    LowLiquidity,
    /// Cooling-off period active
    CoolingOff,
    /// Correlation risk
    CorrelationRisk,
    /// Drawdown protection
    DrawdownProtection,
    /// Time-based restriction
    TimeRestriction,
    /// External halt signal
    ExternalHalt,
    /// Pattern-based inhibition (learned bad patterns)
    LearnedPattern,
    /// Custom inhibition
    Custom(String),
}

impl InhibitionReason {
    /// Get severity weight for this reason (higher = more severe)
    pub fn severity(&self) -> f64 {
        match self {
            Self::ExternalHalt => 1.0,
            Self::LossLimit => 0.95,
            Self::DrawdownProtection => 0.9,
            Self::RiskThreshold => 0.8,
            Self::PositionLimit => 0.7,
            Self::HighVolatility => 0.6,
            Self::LowLiquidity => 0.6,
            Self::CorrelationRisk => 0.5,
            Self::CoolingOff => 0.4,
            Self::TimeRestriction => 0.3,
            Self::LearnedPattern => 0.5,
            Self::Custom(_) => 0.5,
        }
    }

    /// Whether this inhibition can be overridden
    pub fn is_overridable(&self) -> bool {
        !matches!(
            self,
            Self::ExternalHalt | Self::LossLimit | Self::DrawdownProtection
        )
    }
}

/// Configuration for NoGo signal generation
#[derive(Debug, Clone)]
pub struct NoGoConfig {
    /// Base inhibition threshold (0.0 - 1.0)
    pub base_threshold: f64,
    /// Risk-adjusted threshold multiplier
    pub risk_multiplier: f64,
    /// Cooling-off period duration in milliseconds
    pub cooling_off_ms: u64,
    /// Maximum consecutive losses before forced inhibition
    pub max_consecutive_losses: u32,
    /// Drawdown threshold for protection (percentage)
    pub drawdown_threshold: f64,
    /// Volatility threshold for inhibition
    pub volatility_threshold: f64,
    /// History window size for pattern learning
    pub history_window: usize,
    /// Decay rate for learned inhibitions
    pub learning_decay: f64,
    /// Enable adaptive thresholds
    pub adaptive_threshold: bool,
}

impl Default for NoGoConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.5,
            risk_multiplier: 1.5,
            cooling_off_ms: 5000,
            max_consecutive_losses: 3,
            drawdown_threshold: 0.1,
            volatility_threshold: 0.03,
            history_window: 100,
            learning_decay: 0.95,
            adaptive_threshold: true,
        }
    }
}

/// Context for evaluating inhibition
#[derive(Debug, Clone, Default)]
pub struct InhibitionContext {
    /// Current portfolio risk level (0.0 - 1.0)
    pub risk_level: f64,
    /// Current volatility
    pub volatility: f64,
    /// Current drawdown from peak
    pub drawdown: f64,
    /// Consecutive losing trades
    pub consecutive_losses: u32,
    /// Current position exposure
    pub position_exposure: f64,
    /// Maximum allowed exposure
    pub max_exposure: f64,
    /// Liquidity score (0.0 - 1.0, higher = more liquid)
    pub liquidity_score: f64,
    /// Correlation with existing positions
    pub correlation_with_portfolio: f64,
    /// External halt flag
    pub external_halt: bool,
    /// Time-based trading allowed
    pub time_allowed: bool,
    /// Action type being evaluated
    pub action_type: String,
    /// Symbol being traded
    pub symbol: String,
}

/// Result of NoGo evaluation
#[derive(Debug, Clone)]
pub struct NoGoDecision {
    /// Whether to inhibit the action
    pub inhibit: bool,
    /// Overall inhibition strength (0.0 - 1.0)
    pub strength: f64,
    /// Primary reason for inhibition (if any)
    pub primary_reason: Option<InhibitionReason>,
    /// All active inhibition reasons
    pub all_reasons: Vec<(InhibitionReason, f64)>,
    /// Confidence in this decision
    pub confidence: f64,
    /// Timestamp of decision
    pub timestamp: Instant,
    /// Whether override is possible
    pub can_override: bool,
}

impl NoGoDecision {
    /// Create a non-inhibiting decision
    pub fn allow() -> Self {
        Self {
            inhibit: false,
            strength: 0.0,
            primary_reason: None,
            all_reasons: Vec::new(),
            confidence: 1.0,
            timestamp: Instant::now(),
            can_override: true,
        }
    }

    /// Create an inhibiting decision
    pub fn inhibit(reason: InhibitionReason, strength: f64) -> Self {
        let can_override = reason.is_overridable();
        Self {
            inhibit: true,
            strength,
            primary_reason: Some(reason.clone()),
            all_reasons: vec![(reason, strength)],
            confidence: strength,
            timestamp: Instant::now(),
            can_override,
        }
    }
}

/// Historical action outcome for learning
#[derive(Debug, Clone)]
struct ActionOutcome {
    #[allow(dead_code)]
    action_type: String,
    #[allow(dead_code)]
    symbol: String,
    was_profitable: bool,
    #[allow(dead_code)]
    context_hash: u64,
    #[allow(dead_code)]
    timestamp: Instant,
}

/// Statistics for inhibition tracking
#[derive(Debug, Clone, Default)]
pub struct InhibitionStats {
    /// Total evaluations
    pub total_evaluations: u64,
    /// Total inhibitions
    pub total_inhibitions: u64,
    /// Inhibitions by reason
    pub by_reason: HashMap<String, u64>,
    /// Overrides performed
    pub overrides: u64,
    /// Correct inhibitions (prevented losses)
    pub correct_inhibitions: u64,
    /// Missed opportunities (inhibited profitable actions)
    pub missed_opportunities: u64,
}

/// Action inhibition signal (indirect pathway)
///
/// Implements the NoGo pathway from basal ganglia, responsible for
/// preventing actions that should not be taken based on risk,
/// learned patterns, and current market conditions.
pub struct NoGoSignal {
    config: NoGoConfig,
    /// Learned inhibition weights per context
    learned_inhibitions: HashMap<u64, f64>,
    /// Recent action history
    action_history: VecDeque<ActionOutcome>,
    /// Active cooling-off periods
    cooling_off: HashMap<String, Instant>,
    /// Current adaptive threshold
    adaptive_threshold: f64,
    /// Statistics
    stats: InhibitionStats,
    /// Last evaluation result
    last_decision: Option<NoGoDecision>,
}

impl Default for NoGoSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGoSignal {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(NoGoConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: NoGoConfig) -> Self {
        let base_threshold = config.base_threshold;
        Self {
            config,
            learned_inhibitions: HashMap::new(),
            action_history: VecDeque::new(),
            cooling_off: HashMap::new(),
            adaptive_threshold: base_threshold,
            stats: InhibitionStats::default(),
            last_decision: None,
        }
    }

    /// Main processing function - for compatibility
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Evaluate whether to inhibit an action
    pub fn evaluate(&mut self, context: &InhibitionContext) -> NoGoDecision {
        self.stats.total_evaluations += 1;

        let mut reasons = Vec::new();

        // Check external halt (highest priority)
        if context.external_halt {
            return self
                .record_decision(NoGoDecision::inhibit(InhibitionReason::ExternalHalt, 1.0));
        }

        // Check time restrictions
        if !context.time_allowed {
            reasons.push((InhibitionReason::TimeRestriction, 0.8));
        }

        // Check drawdown protection
        if context.drawdown > self.config.drawdown_threshold {
            let strength = (context.drawdown / self.config.drawdown_threshold).min(1.0);
            reasons.push((InhibitionReason::DrawdownProtection, strength));
        }

        // Check consecutive losses
        if context.consecutive_losses >= self.config.max_consecutive_losses {
            reasons.push((InhibitionReason::LossLimit, 0.9));
        }

        // Check risk threshold
        let risk_threshold = self.adaptive_threshold * self.config.risk_multiplier;
        if context.risk_level > risk_threshold {
            let strength = (context.risk_level / risk_threshold).min(1.0);
            reasons.push((InhibitionReason::RiskThreshold, strength));
        }

        // Check position limits
        if context.position_exposure >= context.max_exposure {
            reasons.push((InhibitionReason::PositionLimit, 0.9));
        }

        // Check volatility
        if context.volatility > self.config.volatility_threshold {
            let strength = (context.volatility / self.config.volatility_threshold - 1.0)
                .min(1.0)
                .max(0.0);
            if strength > 0.3 {
                reasons.push((InhibitionReason::HighVolatility, strength));
            }
        }

        // Check liquidity
        if context.liquidity_score < 0.3 {
            let strength = 1.0 - context.liquidity_score;
            reasons.push((InhibitionReason::LowLiquidity, strength));
        }

        // Check correlation risk
        if context.correlation_with_portfolio.abs() > 0.8 {
            reasons.push((
                InhibitionReason::CorrelationRisk,
                context.correlation_with_portfolio.abs(),
            ));
        }

        // Check cooling-off period
        if self.is_cooling_off(&context.symbol) {
            reasons.push((InhibitionReason::CoolingOff, 0.7));
        }

        // Check learned patterns
        let context_hash = self.hash_context(context);
        if let Some(&learned_strength) = self.learned_inhibitions.get(&context_hash) {
            if learned_strength > 0.3 {
                reasons.push((InhibitionReason::LearnedPattern, learned_strength));
            }
        }

        // Calculate overall inhibition decision
        self.make_decision(reasons)
    }

    /// Quick check if action should be immediately blocked
    pub fn should_block(&self, context: &InhibitionContext) -> bool {
        context.external_halt
            || context.drawdown > self.config.drawdown_threshold
            || context.consecutive_losses >= self.config.max_consecutive_losses
            || self.is_cooling_off(&context.symbol)
    }

    /// Start cooling-off period for a symbol
    pub fn start_cooling_off(&mut self, symbol: &str) {
        self.cooling_off.insert(symbol.to_string(), Instant::now());
    }

    /// Check if symbol is in cooling-off period
    pub fn is_cooling_off(&self, symbol: &str) -> bool {
        if let Some(&start) = self.cooling_off.get(symbol) {
            start.elapsed() < Duration::from_millis(self.config.cooling_off_ms)
        } else {
            false
        }
    }

    /// Clear cooling-off for a symbol
    pub fn clear_cooling_off(&mut self, symbol: &str) {
        self.cooling_off.remove(symbol);
    }

    /// Update with action outcome for learning
    pub fn update_with_outcome(
        &mut self,
        context: &InhibitionContext,
        was_profitable: bool,
        was_inhibited: bool,
    ) {
        let context_hash = self.hash_context(context);

        // Record outcome
        let outcome = ActionOutcome {
            action_type: context.action_type.clone(),
            symbol: context.symbol.clone(),
            was_profitable,
            context_hash,
            timestamp: Instant::now(),
        };

        self.action_history.push_back(outcome);
        while self.action_history.len() > self.config.history_window {
            self.action_history.pop_front();
        }

        // Update learned inhibitions
        let current = self
            .learned_inhibitions
            .get(&context_hash)
            .copied()
            .unwrap_or(0.0);

        let adjustment = if was_inhibited {
            if was_profitable {
                // We inhibited a profitable action - reduce inhibition
                self.stats.missed_opportunities += 1;
                -0.1
            } else {
                // We correctly inhibited a losing action
                self.stats.correct_inhibitions += 1;
                0.05
            }
        } else if was_profitable {
            // Allowed action was profitable - reduce inhibition
            -0.02
        } else {
            // Allowed action was unprofitable - increase inhibition
            0.08
        };

        let new_value = (current + adjustment).clamp(0.0, 1.0);
        self.learned_inhibitions.insert(context_hash, new_value);

        // Apply decay to all learned inhibitions
        self.decay_learned_inhibitions();

        // Update adaptive threshold
        if self.config.adaptive_threshold {
            self.update_adaptive_threshold();
        }
    }

    /// Get current inhibition strength for a context
    pub fn get_inhibition_strength(&self, context: &InhibitionContext) -> f64 {
        let context_hash = self.hash_context(context);
        self.learned_inhibitions
            .get(&context_hash)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get statistics
    pub fn stats(&self) -> &InhibitionStats {
        &self.stats
    }

    /// Get last decision
    pub fn last_decision(&self) -> Option<&NoGoDecision> {
        self.last_decision.as_ref()
    }

    /// Get inhibition rate
    pub fn inhibition_rate(&self) -> f64 {
        if self.stats.total_evaluations == 0 {
            0.0
        } else {
            self.stats.total_inhibitions as f64 / self.stats.total_evaluations as f64
        }
    }

    /// Get accuracy (correct inhibitions / total inhibitions)
    pub fn accuracy(&self) -> f64 {
        if self.stats.total_inhibitions == 0 {
            0.0
        } else {
            self.stats.correct_inhibitions as f64 / self.stats.total_inhibitions as f64
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.learned_inhibitions.clear();
        self.action_history.clear();
        self.cooling_off.clear();
        self.adaptive_threshold = self.config.base_threshold;
        self.stats = InhibitionStats::default();
        self.last_decision = None;
    }

    /// Override an inhibition (for manual intervention)
    pub fn override_inhibition(&mut self, _reason: &str) -> Result<()> {
        if let Some(decision) = &self.last_decision {
            if !decision.can_override {
                return Err(Error::InvalidInput(
                    "This inhibition cannot be overridden".into(),
                ));
            }
        }
        self.stats.overrides += 1;
        Ok(())
    }

    // --- Private methods ---

    /// Make decision based on collected reasons
    fn make_decision(&mut self, reasons: Vec<(InhibitionReason, f64)>) -> NoGoDecision {
        if reasons.is_empty() {
            return self.record_decision(NoGoDecision::allow());
        }

        // Find the strongest reason
        let (primary_reason, max_strength) = reasons
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, s)| (r.clone(), *s))
            .unwrap();

        // Calculate weighted overall strength
        let total_strength: f64 = reasons.iter().map(|(_, s)| s).sum();
        let avg_strength = total_strength / reasons.len() as f64;
        let overall_strength = (max_strength * 0.6 + avg_strength * 0.4).min(1.0);

        // Determine if we should inhibit
        let inhibit = overall_strength > self.adaptive_threshold;
        let can_override = reasons.iter().all(|(r, _)| r.is_overridable());

        let decision = NoGoDecision {
            inhibit,
            strength: overall_strength,
            primary_reason: if inhibit { Some(primary_reason) } else { None },
            all_reasons: reasons,
            confidence: overall_strength,
            timestamp: Instant::now(),
            can_override,
        };

        self.record_decision(decision)
    }

    /// Record decision and update stats
    fn record_decision(&mut self, decision: NoGoDecision) -> NoGoDecision {
        if decision.inhibit {
            self.stats.total_inhibitions += 1;
            if let Some(ref reason) = decision.primary_reason {
                let reason_str = format!("{:?}", reason);
                *self.stats.by_reason.entry(reason_str).or_insert(0) += 1;
            }
        }

        self.last_decision = Some(decision.clone());
        decision
    }

    /// Hash context for learned inhibition lookup
    fn hash_context(&self, context: &InhibitionContext) -> u64 {
        // Simple hash combining key context elements
        let mut hash: u64 = 0;

        // Discretize risk level (0-10)
        let risk_bucket = (context.risk_level * 10.0) as u64;
        hash = hash.wrapping_mul(31).wrapping_add(risk_bucket);

        // Discretize volatility (0-10)
        let vol_bucket = (context.volatility * 100.0).min(10.0) as u64;
        hash = hash.wrapping_mul(31).wrapping_add(vol_bucket);

        // Hash action type
        for c in context.action_type.chars() {
            hash = hash.wrapping_mul(31).wrapping_add(c as u64);
        }

        // Hash symbol (first few chars)
        for c in context.symbol.chars().take(4) {
            hash = hash.wrapping_mul(31).wrapping_add(c as u64);
        }

        hash
    }

    /// Decay learned inhibitions over time
    fn decay_learned_inhibitions(&mut self) {
        let decay = self.config.learning_decay;
        self.learned_inhibitions
            .values_mut()
            .for_each(|v| *v *= decay);

        // Remove very small values
        self.learned_inhibitions.retain(|_, v| *v > 0.01);
    }

    /// Update adaptive threshold based on recent performance
    fn update_adaptive_threshold(&mut self) {
        if self.action_history.len() < 10 {
            return;
        }

        // Calculate recent success rate
        let recent: Vec<_> = self.action_history.iter().rev().take(20).collect();
        let profitable = recent.iter().filter(|o| o.was_profitable).count();
        let success_rate = profitable as f64 / recent.len() as f64;

        // Adjust threshold based on performance
        // Lower threshold (more inhibition) if success rate is low
        // Higher threshold (less inhibition) if success rate is high
        let target = self.config.base_threshold;
        let adjustment = (success_rate - 0.5) * 0.1;

        self.adaptive_threshold = (target + adjustment).clamp(0.3, 0.8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_context() -> InhibitionContext {
        InhibitionContext {
            risk_level: 0.3,
            volatility: 0.01,
            drawdown: 0.02,
            consecutive_losses: 0,
            position_exposure: 0.5,
            max_exposure: 1.0,
            liquidity_score: 0.8,
            correlation_with_portfolio: 0.3,
            external_halt: false,
            time_allowed: true,
            action_type: "buy".to_string(),
            symbol: "BTC-USD".to_string(),
        }
    }

    #[test]
    fn test_basic() {
        let instance = NoGoSignal::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_allow_normal_action() {
        let mut nogo = NoGoSignal::new();
        let context = default_context();

        let decision = nogo.evaluate(&context);
        assert!(!decision.inhibit, "Normal context should not be inhibited");
    }

    #[test]
    fn test_external_halt() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();
        context.external_halt = true;

        let decision = nogo.evaluate(&context);
        assert!(decision.inhibit);
        assert_eq!(
            decision.primary_reason,
            Some(InhibitionReason::ExternalHalt)
        );
        assert!(!decision.can_override);
    }

    #[test]
    fn test_drawdown_protection() {
        let config = NoGoConfig {
            drawdown_threshold: 0.05,
            ..Default::default()
        };
        let mut nogo = NoGoSignal::with_config(config);

        let mut context = default_context();
        context.drawdown = 0.08;

        let decision = nogo.evaluate(&context);
        assert!(decision.inhibit);
        assert_eq!(
            decision.primary_reason,
            Some(InhibitionReason::DrawdownProtection)
        );
    }

    #[test]
    fn test_consecutive_losses() {
        let config = NoGoConfig {
            max_consecutive_losses: 3,
            ..Default::default()
        };
        let mut nogo = NoGoSignal::with_config(config);

        let mut context = default_context();
        context.consecutive_losses = 4;

        let decision = nogo.evaluate(&context);
        assert!(decision.inhibit);
        assert!(
            decision
                .all_reasons
                .iter()
                .any(|(r, _)| *r == InhibitionReason::LossLimit)
        );
    }

    #[test]
    fn test_high_volatility() {
        let config = NoGoConfig {
            volatility_threshold: 0.02,
            ..Default::default()
        };
        let mut nogo = NoGoSignal::with_config(config);

        let mut context = default_context();
        context.volatility = 0.05;

        let decision = nogo.evaluate(&context);
        assert!(
            decision
                .all_reasons
                .iter()
                .any(|(r, _)| *r == InhibitionReason::HighVolatility)
        );
    }

    #[test]
    fn test_cooling_off() {
        let mut nogo = NoGoSignal::new();

        nogo.start_cooling_off("BTC-USD");
        assert!(nogo.is_cooling_off("BTC-USD"));
        assert!(!nogo.is_cooling_off("ETH-USD"));

        nogo.clear_cooling_off("BTC-USD");
        assert!(!nogo.is_cooling_off("BTC-USD"));
    }

    #[test]
    fn test_learning_from_outcomes() {
        let mut nogo = NoGoSignal::new();
        let context = default_context();

        // Initially no learned inhibition
        let initial_strength = nogo.get_inhibition_strength(&context);
        assert!(initial_strength < 0.01);

        // Update with unprofitable outcome (should increase inhibition)
        nogo.update_with_outcome(&context, false, false);
        let after_loss = nogo.get_inhibition_strength(&context);
        assert!(after_loss > initial_strength);

        // Update with profitable outcome (should decrease)
        nogo.update_with_outcome(&context, true, false);
        // May still be higher than initial due to learning rate differences
    }

    #[test]
    fn test_inhibition_stats() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();

        // Normal evaluation
        nogo.evaluate(&context);
        assert_eq!(nogo.stats().total_evaluations, 1);

        // Force inhibition
        context.external_halt = true;
        nogo.evaluate(&context);
        assert_eq!(nogo.stats().total_evaluations, 2);
        assert_eq!(nogo.stats().total_inhibitions, 1);
    }

    #[test]
    fn test_inhibition_rate() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();

        // 2 normal evaluations
        nogo.evaluate(&context);
        nogo.evaluate(&context);

        // 1 inhibited
        context.external_halt = true;
        nogo.evaluate(&context);

        let rate = nogo.inhibition_rate();
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut nogo = NoGoSignal::new();
        let context = default_context();

        nogo.evaluate(&context);
        nogo.start_cooling_off("BTC-USD");
        nogo.update_with_outcome(&context, false, false);

        nogo.reset();

        assert_eq!(nogo.stats().total_evaluations, 0);
        assert!(!nogo.is_cooling_off("BTC-USD"));
        assert!(nogo.get_inhibition_strength(&context) < 0.01);
    }

    #[test]
    fn test_override_not_allowed() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();
        context.external_halt = true;

        nogo.evaluate(&context);

        let result = nogo.override_inhibition("Manual override");
        assert!(result.is_err());
    }

    #[test]
    fn test_override_allowed() {
        let config = NoGoConfig {
            volatility_threshold: 0.01,
            ..Default::default()
        };
        let mut nogo = NoGoSignal::with_config(config);

        let mut context = default_context();
        context.volatility = 0.05;

        let decision = nogo.evaluate(&context);
        if decision.can_override {
            let result = nogo.override_inhibition("Manual override");
            assert!(result.is_ok());
            assert_eq!(nogo.stats().overrides, 1);
        }
    }

    #[test]
    fn test_severity_ordering() {
        assert!(
            InhibitionReason::ExternalHalt.severity() > InhibitionReason::CoolingOff.severity()
        );
        assert!(
            InhibitionReason::LossLimit.severity() > InhibitionReason::TimeRestriction.severity()
        );
    }

    #[test]
    fn test_position_limit() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();
        context.position_exposure = 1.0;
        context.max_exposure = 1.0;

        let decision = nogo.evaluate(&context);
        assert!(
            decision
                .all_reasons
                .iter()
                .any(|(r, _)| *r == InhibitionReason::PositionLimit)
        );
    }

    #[test]
    fn test_low_liquidity() {
        let mut nogo = NoGoSignal::new();
        let mut context = default_context();
        context.liquidity_score = 0.1;

        let decision = nogo.evaluate(&context);
        assert!(
            decision
                .all_reasons
                .iter()
                .any(|(r, _)| *r == InhibitionReason::LowLiquidity)
        );
    }
}
