//! Relevance scoring
//!
//! Part of the Thalamus region - assesses signal importance and relevance
//! to current trading context and goals.

use crate::common::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Relevance dimension for multi-factor scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelevanceDimension {
    /// Temporal relevance - how recent/timely
    Temporal,
    /// Contextual relevance - fits current market context
    Contextual,
    /// Strategic relevance - aligns with trading strategy
    Strategic,
    /// Risk relevance - important for risk management
    Risk,
    /// Opportunity relevance - potential profit opportunity
    Opportunity,
    /// Informational relevance - valuable information content
    Informational,
    /// Correlation relevance - relates to watched assets
    Correlation,
}

impl RelevanceDimension {
    /// Get default weight for this dimension
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::Temporal => 1.0,
            Self::Contextual => 1.2,
            Self::Strategic => 1.5,
            Self::Risk => 2.0,
            Self::Opportunity => 1.3,
            Self::Informational => 0.8,
            Self::Correlation => 0.7,
        }
    }

    /// Get all dimensions
    pub fn all() -> Vec<Self> {
        vec![
            Self::Temporal,
            Self::Contextual,
            Self::Strategic,
            Self::Risk,
            Self::Opportunity,
            Self::Informational,
            Self::Correlation,
        ]
    }
}

/// Score for a single dimension
#[derive(Debug, Clone)]
pub struct DimensionScore {
    /// The dimension being scored
    pub dimension: RelevanceDimension,
    /// Raw score (0.0 - 1.0)
    pub raw_score: f64,
    /// Weight applied
    pub weight: f64,
    /// Weighted score
    pub weighted_score: f64,
    /// Confidence in this score
    pub confidence: f64,
    /// Explanation for the score
    pub explanation: Option<String>,
}

impl DimensionScore {
    /// Create a new dimension score
    pub fn new(dimension: RelevanceDimension, raw_score: f64, weight: f64) -> Self {
        Self {
            dimension,
            raw_score: raw_score.clamp(0.0, 1.0),
            weight,
            weighted_score: raw_score.clamp(0.0, 1.0) * weight,
            confidence: 1.0,
            explanation: None,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set explanation
    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }
}

/// Complete relevance assessment
#[derive(Debug, Clone)]
pub struct RelevanceAssessment {
    /// Signal or entity being assessed
    pub subject_id: String,
    /// Individual dimension scores
    pub dimension_scores: Vec<DimensionScore>,
    /// Overall relevance score (0.0 - 1.0)
    pub overall_score: f64,
    /// Overall confidence
    pub overall_confidence: f64,
    /// Relevance category
    pub category: RelevanceCategory,
    /// Assessment timestamp
    pub timestamp: u64,
    /// Context used for assessment
    pub context: Option<String>,
}

impl RelevanceAssessment {
    /// Get score for a specific dimension
    pub fn dimension_score(&self, dimension: RelevanceDimension) -> Option<f64> {
        self.dimension_scores
            .iter()
            .find(|s| s.dimension == dimension)
            .map(|s| s.raw_score)
    }

    /// Check if passes minimum threshold
    pub fn passes_threshold(&self, threshold: f64) -> bool {
        self.overall_score >= threshold
    }

    /// Get dominant dimension (highest score)
    pub fn dominant_dimension(&self) -> Option<&DimensionScore> {
        self.dimension_scores.iter().max_by(|a, b| {
            a.weighted_score
                .partial_cmp(&b.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get weakest dimension (lowest score)
    pub fn weakest_dimension(&self) -> Option<&DimensionScore> {
        self.dimension_scores.iter().min_by(|a, b| {
            a.weighted_score
                .partial_cmp(&b.weighted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Relevance category based on score
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelevanceCategory {
    /// Critical relevance - must process immediately
    Critical,
    /// High relevance - prioritize processing
    High,
    /// Medium relevance - normal processing
    Medium,
    /// Low relevance - process if resources available
    Low,
    /// Negligible - can be safely ignored
    Negligible,
}

impl RelevanceCategory {
    /// Create from score
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 0.9 => Self::Critical,
            s if s >= 0.7 => Self::High,
            s if s >= 0.4 => Self::Medium,
            s if s >= 0.2 => Self::Low,
            _ => Self::Negligible,
        }
    }

    /// Get minimum score for this category
    pub fn min_score(&self) -> f64 {
        match self {
            Self::Critical => 0.9,
            Self::High => 0.7,
            Self::Medium => 0.4,
            Self::Low => 0.2,
            Self::Negligible => 0.0,
        }
    }

    /// Get processing priority (higher = more important)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Critical => 5,
            Self::High => 4,
            Self::Medium => 3,
            Self::Low => 2,
            Self::Negligible => 1,
        }
    }
}

/// Context information for relevance scoring
#[derive(Debug, Clone, Default)]
pub struct ScoringContext {
    /// Current market regime
    pub market_regime: Option<String>,
    /// Active trading symbols
    pub active_symbols: Vec<String>,
    /// Current positions
    pub current_positions: Vec<String>,
    /// Risk level (0.0 - 1.0)
    pub risk_level: f64,
    /// Trading mode (e.g., "aggressive", "conservative")
    pub trading_mode: Option<String>,
    /// Time of day factor (market hours importance)
    pub market_hours_factor: f64,
    /// Custom context values
    pub custom: HashMap<String, f64>,
}

impl ScoringContext {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            market_hours_factor: 1.0,
            ..Default::default()
        }
    }

    /// Set market regime
    pub fn with_market_regime(mut self, regime: impl Into<String>) -> Self {
        self.market_regime = Some(regime.into());
        self
    }

    /// Set active symbols
    pub fn with_active_symbols(mut self, symbols: Vec<String>) -> Self {
        self.active_symbols = symbols;
        self
    }

    /// Set risk level
    pub fn with_risk_level(mut self, level: f64) -> Self {
        self.risk_level = level.clamp(0.0, 1.0);
        self
    }

    /// Add custom context value
    pub fn with_custom(mut self, key: impl Into<String>, value: f64) -> Self {
        self.custom.insert(key.into(), value);
        self
    }

    /// Check if a symbol is active
    pub fn is_symbol_active(&self, symbol: &str) -> bool {
        self.active_symbols.iter().any(|s| s == symbol)
    }

    /// Check if symbol has position
    pub fn has_position(&self, symbol: &str) -> bool {
        self.current_positions.iter().any(|s| s == symbol)
    }
}

/// Signal input for relevance scoring
#[derive(Debug, Clone)]
pub struct SignalInput {
    /// Unique identifier
    pub id: String,
    /// Signal type
    pub signal_type: String,
    /// Associated symbol
    pub symbol: Option<String>,
    /// Signal age in milliseconds
    pub age_ms: u64,
    /// Signal strength (0.0 - 1.0)
    pub strength: f64,
    /// Source reliability (0.0 - 1.0)
    pub source_reliability: f64,
    /// Information content estimate
    pub information_content: f64,
    /// Risk impact estimate (-1.0 to 1.0)
    pub risk_impact: f64,
    /// Opportunity score (0.0 - 1.0)
    pub opportunity_score: f64,
    /// Correlation with portfolio
    pub portfolio_correlation: f64,
    /// Custom attributes
    pub attributes: HashMap<String, f64>,
}

impl SignalInput {
    /// Create a new signal input
    pub fn new(id: impl Into<String>, signal_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            signal_type: signal_type.into(),
            symbol: None,
            age_ms: 0,
            strength: 0.5,
            source_reliability: 0.8,
            information_content: 0.5,
            risk_impact: 0.0,
            opportunity_score: 0.5,
            portfolio_correlation: 0.0,
            attributes: HashMap::new(),
        }
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Set age
    pub fn with_age(mut self, age_ms: u64) -> Self {
        self.age_ms = age_ms;
        self
    }

    /// Set strength
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set source reliability
    pub fn with_reliability(mut self, reliability: f64) -> Self {
        self.source_reliability = reliability.clamp(0.0, 1.0);
        self
    }

    /// Set risk impact
    pub fn with_risk_impact(mut self, impact: f64) -> Self {
        self.risk_impact = impact.clamp(-1.0, 1.0);
        self
    }

    /// Set opportunity score
    pub fn with_opportunity(mut self, score: f64) -> Self {
        self.opportunity_score = score.clamp(0.0, 1.0);
        self
    }
}

/// Relevance scoring configuration
#[derive(Debug, Clone)]
pub struct RelevanceConfig {
    /// Dimension weights
    pub weights: HashMap<RelevanceDimension, f64>,
    /// Signal age decay half-life in milliseconds
    pub age_half_life_ms: u64,
    /// Minimum score threshold
    pub min_threshold: f64,
    /// Enable adaptive weighting
    pub adaptive_weights: bool,
    /// Cache assessments for this duration (ms)
    pub cache_duration_ms: u64,
    /// Default reliability for unknown sources
    pub default_reliability: f64,
}

impl Default for RelevanceConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        for dim in RelevanceDimension::all() {
            weights.insert(dim, dim.default_weight());
        }

        Self {
            weights,
            age_half_life_ms: 5000, // 5 seconds half-life
            min_threshold: 0.1,
            adaptive_weights: true,
            cache_duration_ms: 1000,
            default_reliability: 0.5,
        }
    }
}

/// Relevance scorer statistics
#[derive(Debug, Clone, Default)]
pub struct RelevanceStats {
    /// Total assessments made
    pub total_assessments: u64,
    /// Assessments by category
    pub by_category: HashMap<String, u64>,
    /// Average score
    pub avg_score: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Assessments that passed threshold
    pub passed_threshold: u64,
    /// Average assessment latency in microseconds
    pub avg_latency_us: f64,
    /// Score distribution histogram
    pub score_histogram: [u64; 10],
}

/// Relevance scoring system
pub struct Relevance {
    /// Configuration
    config: RelevanceConfig,
    /// Current context
    context: Arc<RwLock<ScoringContext>>,
    /// Statistics
    stats: Arc<RwLock<RelevanceStats>>,
    /// Assessment cache
    cache: Arc<RwLock<HashMap<String, (RelevanceAssessment, u64)>>>,
    /// Signal type weights (learned)
    type_weights: Arc<RwLock<HashMap<String, f64>>>,
}

impl Default for Relevance {
    fn default() -> Self {
        Self::new()
    }
}

impl Relevance {
    /// Create a new relevance scorer
    pub fn new() -> Self {
        Self::with_config(RelevanceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: RelevanceConfig) -> Self {
        Self {
            config,
            context: Arc::new(RwLock::new(ScoringContext::new())),
            stats: Arc::new(RwLock::new(RelevanceStats::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            type_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Update the scoring context
    pub async fn set_context(&self, context: ScoringContext) {
        *self.context.write().await = context;
    }

    /// Get current context
    pub async fn get_context(&self) -> ScoringContext {
        self.context.read().await.clone()
    }

    /// Score a signal's relevance
    pub async fn score(&self, input: &SignalInput) -> RelevanceAssessment {
        let start = std::time::Instant::now();

        // Check cache
        if let Some(cached) = self.get_cached(&input.id).await {
            return cached;
        }

        let context = self.context.read().await;
        let mut dimension_scores = Vec::new();

        // Score each dimension
        dimension_scores.push(self.score_temporal(input));
        dimension_scores.push(self.score_contextual(input, &context));
        dimension_scores.push(self.score_strategic(input, &context));
        dimension_scores.push(self.score_risk(input, &context));
        dimension_scores.push(self.score_opportunity(input, &context));
        dimension_scores.push(self.score_informational(input));
        dimension_scores.push(self.score_correlation(input, &context));

        // Calculate overall score
        let total_weight: f64 = dimension_scores.iter().map(|s| s.weight).sum();
        let weighted_sum: f64 = dimension_scores.iter().map(|s| s.weighted_score).sum();
        let overall_score = if total_weight > 0.0 {
            (weighted_sum / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Calculate overall confidence
        let overall_confidence = dimension_scores
            .iter()
            .map(|s| s.confidence * s.weight)
            .sum::<f64>()
            / total_weight.max(1.0);

        let assessment = RelevanceAssessment {
            subject_id: input.id.clone(),
            dimension_scores,
            overall_score,
            overall_confidence,
            category: RelevanceCategory::from_score(overall_score),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            context: context.market_regime.clone(),
        };

        drop(context);

        // Update cache
        self.cache_assessment(&input.id, &assessment).await;

        // Update statistics
        let elapsed = start.elapsed().as_micros() as f64;
        self.update_stats(&assessment, elapsed).await;

        assessment
    }

    /// Score temporal dimension
    fn score_temporal(&self, input: &SignalInput) -> DimensionScore {
        // Exponential decay based on age
        let decay = (-0.693 * input.age_ms as f64 / self.config.age_half_life_ms as f64).exp();
        let score = decay * input.strength;

        DimensionScore::new(
            RelevanceDimension::Temporal,
            score,
            self.get_weight(RelevanceDimension::Temporal),
        )
        .with_explanation(format!("Age: {}ms, decay: {:.2}", input.age_ms, decay))
    }

    /// Score contextual dimension
    fn score_contextual(&self, input: &SignalInput, context: &ScoringContext) -> DimensionScore {
        let mut score = 0.5; // Base score

        // Boost if symbol is active
        if let Some(ref symbol) = input.symbol {
            if context.is_symbol_active(symbol) {
                score += 0.3;
            }
            if context.has_position(symbol) {
                score += 0.2;
            }
        }

        // Market hours factor
        score *= context.market_hours_factor;

        DimensionScore::new(
            RelevanceDimension::Contextual,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Contextual),
        )
    }

    /// Score strategic dimension
    fn score_strategic(&self, input: &SignalInput, context: &ScoringContext) -> DimensionScore {
        let mut score = input.strength;

        // Adjust based on trading mode
        if let Some(ref mode) = context.trading_mode {
            match mode.as_str() {
                "aggressive" => score *= 1.2,
                "conservative" => score *= 0.8,
                _ => {}
            }
        }

        DimensionScore::new(
            RelevanceDimension::Strategic,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Strategic),
        )
    }

    /// Score risk dimension
    fn score_risk(&self, input: &SignalInput, context: &ScoringContext) -> DimensionScore {
        // Risk signals are more relevant when risk level is high
        let risk_multiplier = 1.0 + context.risk_level;
        let score = input.risk_impact.abs() * risk_multiplier;

        DimensionScore::new(
            RelevanceDimension::Risk,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Risk),
        )
        .with_confidence(input.source_reliability)
    }

    /// Score opportunity dimension
    fn score_opportunity(&self, input: &SignalInput, context: &ScoringContext) -> DimensionScore {
        let mut score = input.opportunity_score;

        // Reduce opportunity score when risk is high
        score *= 1.0 - (context.risk_level * 0.5);

        DimensionScore::new(
            RelevanceDimension::Opportunity,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Opportunity),
        )
    }

    /// Score informational dimension
    fn score_informational(&self, input: &SignalInput) -> DimensionScore {
        let score = input.information_content * input.source_reliability;

        DimensionScore::new(
            RelevanceDimension::Informational,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Informational),
        )
        .with_confidence(input.source_reliability)
    }

    /// Score correlation dimension
    fn score_correlation(&self, input: &SignalInput, context: &ScoringContext) -> DimensionScore {
        let mut score = input.portfolio_correlation.abs();

        // Boost if symbol has position
        if let Some(ref symbol) = input.symbol {
            if context.has_position(symbol) {
                score = score.max(0.7);
            }
        }

        DimensionScore::new(
            RelevanceDimension::Correlation,
            score.clamp(0.0, 1.0),
            self.get_weight(RelevanceDimension::Correlation),
        )
    }

    /// Get weight for a dimension
    fn get_weight(&self, dimension: RelevanceDimension) -> f64 {
        self.config
            .weights
            .get(&dimension)
            .copied()
            .unwrap_or(dimension.default_weight())
    }

    /// Get cached assessment if valid
    async fn get_cached(&self, id: &str) -> Option<RelevanceAssessment> {
        let cache = self.cache.read().await;
        if let Some((assessment, timestamp)) = cache.get(id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            if now - timestamp < self.config.cache_duration_ms {
                return Some(assessment.clone());
            }
        }
        None
    }

    /// Cache an assessment
    async fn cache_assessment(&self, id: &str, assessment: &RelevanceAssessment) {
        let mut cache = self.cache.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        cache.insert(id.to_string(), (assessment.clone(), now));

        // Clean old entries
        let cutoff = now.saturating_sub(self.config.cache_duration_ms * 10);
        cache.retain(|_, (_, ts)| *ts > cutoff);
    }

    /// Update statistics
    async fn update_stats(&self, assessment: &RelevanceAssessment, latency_us: f64) {
        let mut stats = self.stats.write().await;

        stats.total_assessments += 1;

        // Update by category
        let cat_key = format!("{:?}", assessment.category);
        *stats.by_category.entry(cat_key).or_insert(0) += 1;

        // Update averages (EMA)
        let alpha = 0.1;
        stats.avg_score = stats.avg_score * (1.0 - alpha) + assessment.overall_score * alpha;
        stats.avg_confidence =
            stats.avg_confidence * (1.0 - alpha) + assessment.overall_confidence * alpha;
        stats.avg_latency_us = stats.avg_latency_us * (1.0 - alpha) + latency_us * alpha;

        // Update threshold counter
        if assessment.overall_score >= self.config.min_threshold {
            stats.passed_threshold += 1;
        }

        // Update histogram
        let bucket = (assessment.overall_score * 10.0).min(9.0) as usize;
        stats.score_histogram[bucket] += 1;
    }

    /// Score multiple signals in batch
    pub async fn score_batch(&self, inputs: &[SignalInput]) -> Vec<RelevanceAssessment> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.score(input).await);
        }
        results
    }

    /// Filter signals by minimum relevance
    pub async fn filter_relevant(
        &self,
        inputs: &[SignalInput],
        min_score: f64,
    ) -> Vec<(SignalInput, RelevanceAssessment)> {
        let mut relevant = Vec::new();
        for input in inputs {
            let assessment = self.score(input).await;
            if assessment.overall_score >= min_score {
                relevant.push((input.clone(), assessment));
            }
        }

        // Sort by relevance (highest first)
        relevant.sort_by(|a, b| {
            b.1.overall_score
                .partial_cmp(&a.1.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        relevant
    }

    /// Set weight for a dimension
    pub async fn set_dimension_weight(&self, dimension: RelevanceDimension, weight: f64) {
        // Note: In a real implementation, we'd need mutable config
        // For now, this is a placeholder for adaptive weight learning
        let mut type_weights = self.type_weights.write().await;
        type_weights.insert(format!("{:?}", dimension), weight);
    }

    /// Get statistics
    pub async fn stats(&self) -> RelevanceStats {
        self.stats.read().await.clone()
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &RelevanceConfig {
        &self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_category_from_score() {
        assert_eq!(
            RelevanceCategory::from_score(0.95),
            RelevanceCategory::Critical
        );
        assert_eq!(RelevanceCategory::from_score(0.75), RelevanceCategory::High);
        assert_eq!(
            RelevanceCategory::from_score(0.5),
            RelevanceCategory::Medium
        );
        assert_eq!(RelevanceCategory::from_score(0.25), RelevanceCategory::Low);
        assert_eq!(
            RelevanceCategory::from_score(0.1),
            RelevanceCategory::Negligible
        );
    }

    #[test]
    fn test_dimension_weights() {
        assert!(
            RelevanceDimension::Risk.default_weight()
                > RelevanceDimension::Informational.default_weight()
        );
    }

    #[test]
    fn test_signal_input_builder() {
        let input = SignalInput::new("sig_001", "price_update")
            .with_symbol("BTCUSD")
            .with_strength(0.8)
            .with_age(1000)
            .with_risk_impact(0.3);

        assert_eq!(input.id, "sig_001");
        assert_eq!(input.symbol, Some("BTCUSD".to_string()));
        assert!((input.strength - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_scoring_context() {
        let context = ScoringContext::new()
            .with_active_symbols(vec!["BTCUSD".to_string(), "ETHUSD".to_string()])
            .with_risk_level(0.5);

        assert!(context.is_symbol_active("BTCUSD"));
        assert!(!context.is_symbol_active("XRPUSD"));
    }

    #[tokio::test]
    async fn test_score_signal() {
        let scorer = Relevance::new();

        let input = SignalInput::new("test_sig", "price")
            .with_strength(0.7)
            .with_age(100);

        let assessment = scorer.score(&input).await;

        assert!(assessment.overall_score > 0.0);
        assert!(assessment.overall_score <= 1.0);
        assert!(!assessment.dimension_scores.is_empty());
    }

    #[tokio::test]
    async fn test_temporal_decay() {
        let scorer = Relevance::new();

        let fresh = SignalInput::new("fresh", "price")
            .with_strength(0.8)
            .with_age(0);

        let old = SignalInput::new("old", "price")
            .with_strength(0.8)
            .with_age(10000); // 10 seconds old

        let fresh_assessment = scorer.score(&fresh).await;
        let old_assessment = scorer.score(&old).await;

        // Fresh signal should have higher temporal score
        let fresh_temporal = fresh_assessment
            .dimension_score(RelevanceDimension::Temporal)
            .unwrap();
        let old_temporal = old_assessment
            .dimension_score(RelevanceDimension::Temporal)
            .unwrap();

        assert!(fresh_temporal > old_temporal);
    }

    #[tokio::test]
    async fn test_context_affects_score() {
        let scorer = Relevance::new();

        // Set context with active symbol
        let context = ScoringContext::new().with_active_symbols(vec!["BTCUSD".to_string()]);
        scorer.set_context(context).await;

        let active_signal = SignalInput::new("active", "price")
            .with_symbol("BTCUSD")
            .with_strength(0.5);

        let inactive_signal = SignalInput::new("inactive", "price")
            .with_symbol("XRPUSD")
            .with_strength(0.5);

        let active_assessment = scorer.score(&active_signal).await;
        let inactive_assessment = scorer.score(&inactive_signal).await;

        // Active symbol should have higher contextual score
        let active_contextual = active_assessment
            .dimension_score(RelevanceDimension::Contextual)
            .unwrap();
        let inactive_contextual = inactive_assessment
            .dimension_score(RelevanceDimension::Contextual)
            .unwrap();

        assert!(active_contextual > inactive_contextual);
    }

    #[tokio::test]
    async fn test_risk_relevance() {
        let scorer = Relevance::new();

        // High risk context
        let context = ScoringContext::new().with_risk_level(0.9);
        scorer.set_context(context).await;

        let risk_signal = SignalInput::new("risk", "alert").with_risk_impact(0.8);

        let assessment = scorer.score(&risk_signal).await;
        let risk_score = assessment
            .dimension_score(RelevanceDimension::Risk)
            .unwrap();

        // High risk impact in high risk context should be very relevant
        assert!(risk_score > 0.5);
    }

    #[tokio::test]
    async fn test_filter_relevant() {
        let scorer = Relevance::new();

        let inputs = vec![
            SignalInput::new("high", "price").with_strength(0.9),
            SignalInput::new("low", "price").with_strength(0.1),
            SignalInput::new("med", "price").with_strength(0.5),
        ];

        let relevant = scorer.filter_relevant(&inputs, 0.3).await;

        // Should be sorted by relevance
        assert!(!relevant.is_empty());
        if relevant.len() >= 2 {
            assert!(relevant[0].1.overall_score >= relevant[1].1.overall_score);
        }
    }

    #[tokio::test]
    async fn test_batch_scoring() {
        let scorer = Relevance::new();

        let inputs: Vec<SignalInput> = (0..5)
            .map(|i| SignalInput::new(format!("sig_{}", i), "price"))
            .collect();

        let assessments = scorer.score_batch(&inputs).await;

        assert_eq!(assessments.len(), 5);
    }

    #[tokio::test]
    async fn test_statistics() {
        let scorer = Relevance::new();

        for i in 0..10 {
            let input =
                SignalInput::new(format!("sig_{}", i), "price").with_strength((i as f64) / 10.0);
            scorer.score(&input).await;
        }

        let stats = scorer.stats().await;
        assert_eq!(stats.total_assessments, 10);
        assert!(stats.avg_score >= 0.0);
    }

    #[test]
    fn test_assessment_helpers() {
        let assessment = RelevanceAssessment {
            subject_id: "test".to_string(),
            dimension_scores: vec![
                DimensionScore::new(RelevanceDimension::Temporal, 0.9, 1.0),
                DimensionScore::new(RelevanceDimension::Risk, 0.3, 2.0),
            ],
            overall_score: 0.6,
            overall_confidence: 0.8,
            category: RelevanceCategory::Medium,
            timestamp: 0,
            context: None,
        };

        assert!(assessment.passes_threshold(0.5));
        assert!(!assessment.passes_threshold(0.7));

        let dominant = assessment.dominant_dimension().unwrap();
        // Risk has higher weighted score (0.3 * 2.0 = 0.6) vs Temporal (0.9 * 1.0 = 0.9)
        // Actually Temporal has higher weighted score
        assert_eq!(dominant.dimension, RelevanceDimension::Temporal);
    }

    #[test]
    fn test_process_compatibility() {
        let relevance = Relevance::new();
        assert!(relevance.process().is_ok());
    }
}
