//! Complete trade sequences
//!
//! Part of the Hippocampus region
//! Component: episodes
//!
//! This module tracks complete trade episodes from entry to exit, storing
//! contextual information for pattern recognition and learning.

use crate::common::{Error, Result};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradeDirection {
    Long,
    Short,
}

/// Trade outcome classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradeOutcome {
    /// Profitable trade
    Win,
    /// Loss trade
    Loss,
    /// Roughly breakeven (within threshold)
    Breakeven,
    /// Still open
    Open,
}

/// Exit reason for completed trades
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    /// Take profit target hit
    TakeProfit,
    /// Stop loss triggered
    StopLoss,
    /// Trailing stop triggered
    TrailingStop,
    /// Time-based exit
    TimeExpiry,
    /// Signal reversal
    SignalReversal,
    /// Manual exit
    Manual,
    /// Risk management forced exit
    RiskManagement,
    /// Market close
    MarketClose,
    /// Unknown/other
    Other,
}

/// Market regime during the trade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Trending up
    Bullish,
    /// Trending down
    Bearish,
    /// Range-bound
    Ranging,
    /// High volatility
    Volatile,
    /// Low volatility / quiet
    Quiet,
    /// Unknown
    Unknown,
}

/// Entry signal that triggered the trade
#[derive(Debug, Clone)]
pub struct EntrySignal {
    /// Signal source/strategy name
    pub source: String,
    /// Signal strength (0.0 - 1.0)
    pub strength: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Additional signal metadata
    pub metadata: HashMap<String, f64>,
}

impl Default for EntrySignal {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            strength: 0.5,
            confidence: 0.5,
            metadata: HashMap::new(),
        }
    }
}

/// Market context at a point in time
#[derive(Debug, Clone)]
pub struct MarketContext {
    /// Current price
    pub price: f64,
    /// Volatility measure (e.g., ATR %)
    pub volatility: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Spread as percentage
    pub spread_pct: f64,
    /// Market regime
    pub regime: MarketRegime,
    /// Time of day (0-23)
    pub hour_of_day: u8,
    /// Day of week (0=Monday, 6=Sunday)
    pub day_of_week: u8,
    /// Additional context data
    pub indicators: HashMap<String, f64>,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            price: 0.0,
            volatility: 0.0,
            relative_volume: 1.0,
            spread_pct: 0.0,
            regime: MarketRegime::Unknown,
            hour_of_day: 12,
            day_of_week: 0,
            indicators: HashMap::new(),
        }
    }
}

/// A complete trade episode from entry to exit
#[derive(Debug, Clone)]
pub struct TradeEpisode {
    /// Unique episode ID
    pub id: String,
    /// Symbol traded
    pub symbol: String,
    /// Trade direction
    pub direction: TradeDirection,
    /// Entry timestamp (unix millis)
    pub entry_time: u64,
    /// Exit timestamp (unix millis), None if still open
    pub exit_time: Option<u64>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price, None if still open
    pub exit_price: Option<f64>,
    /// Position size
    pub size: f64,
    /// Entry signal information
    pub entry_signal: EntrySignal,
    /// Market context at entry
    pub entry_context: MarketContext,
    /// Market context at exit
    pub exit_context: Option<MarketContext>,
    /// Exit reason
    pub exit_reason: Option<ExitReason>,
    /// Realized PnL (after fees)
    pub realized_pnl: Option<f64>,
    /// Return percentage
    pub return_pct: Option<f64>,
    /// Maximum favorable excursion (best unrealized profit)
    pub mfe: f64,
    /// Maximum adverse excursion (worst unrealized loss)
    pub mae: f64,
    /// Trade duration in seconds
    pub duration_secs: Option<u64>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Strategy that generated this trade
    pub strategy_id: String,
}

impl TradeEpisode {
    /// Create a new open trade episode
    pub fn new(
        id: &str,
        symbol: &str,
        direction: TradeDirection,
        entry_price: f64,
        size: f64,
        strategy_id: &str,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        Self {
            id: id.to_string(),
            symbol: symbol.to_string(),
            direction,
            entry_time: now,
            exit_time: None,
            entry_price,
            exit_price: None,
            size,
            entry_signal: EntrySignal::default(),
            entry_context: MarketContext::default(),
            exit_context: None,
            exit_reason: None,
            realized_pnl: None,
            return_pct: None,
            mfe: 0.0,
            mae: 0.0,
            duration_secs: None,
            tags: Vec::new(),
            strategy_id: strategy_id.to_string(),
        }
    }

    /// Close the trade episode
    pub fn close(&mut self, exit_price: f64, exit_reason: ExitReason, fees: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        self.exit_time = Some(now);
        self.exit_price = Some(exit_price);
        self.exit_reason = Some(exit_reason);
        self.duration_secs = Some((now - self.entry_time) / 1000);

        // Calculate PnL
        let gross_pnl = match self.direction {
            TradeDirection::Long => (exit_price - self.entry_price) * self.size,
            TradeDirection::Short => (self.entry_price - exit_price) * self.size,
        };

        self.realized_pnl = Some(gross_pnl - fees);
        self.return_pct = Some(
            (exit_price - self.entry_price) / self.entry_price
                * 100.0
                * if self.direction == TradeDirection::Short {
                    -1.0
                } else {
                    1.0
                },
        );
    }

    /// Update MFE/MAE with current price
    pub fn update_excursion(&mut self, current_price: f64) {
        let unrealized = match self.direction {
            TradeDirection::Long => (current_price - self.entry_price) * self.size,
            TradeDirection::Short => (self.entry_price - current_price) * self.size,
        };

        if unrealized > self.mfe {
            self.mfe = unrealized;
        }
        if unrealized < self.mae {
            self.mae = unrealized;
        }
    }

    /// Get trade outcome
    pub fn outcome(&self) -> TradeOutcome {
        match self.realized_pnl {
            None => TradeOutcome::Open,
            Some(pnl) => {
                let threshold = self.entry_price * self.size * 0.001; // 0.1% threshold
                if pnl > threshold {
                    TradeOutcome::Win
                } else if pnl < -threshold {
                    TradeOutcome::Loss
                } else {
                    TradeOutcome::Breakeven
                }
            }
        }
    }

    /// Check if trade is still open
    pub fn is_open(&self) -> bool {
        self.exit_time.is_none()
    }

    /// Get efficiency ratio (actual profit vs best possible)
    pub fn efficiency(&self) -> Option<f64> {
        if self.mfe <= 0.0 {
            return None;
        }
        self.realized_pnl.map(|pnl| pnl / self.mfe)
    }

    /// Get risk-reward ratio achieved
    pub fn risk_reward(&self) -> Option<f64> {
        if self.mae >= 0.0 {
            return None;
        }
        self.realized_pnl.map(|pnl| pnl / (-self.mae))
    }
}

/// Pattern identified in trade history
#[derive(Debug, Clone)]
pub struct TradePattern {
    /// Pattern name
    pub name: String,
    /// Episodes matching this pattern
    pub episode_ids: Vec<String>,
    /// Win rate for this pattern
    pub win_rate: f64,
    /// Average return for this pattern
    pub avg_return: f64,
    /// Sample size
    pub sample_size: usize,
    /// Pattern confidence
    pub confidence: f64,
    /// Pattern criteria
    pub criteria: HashMap<String, String>,
}

/// Statistics for a group of trades
#[derive(Debug, Clone, Default)]
pub struct TradeStatistics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub breakeven_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub avg_duration_secs: f64,
    pub avg_mfe: f64,
    pub avg_mae: f64,
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
}

/// Trade episode storage and analysis
pub struct TradeEpisodes {
    /// All episodes indexed by ID
    episodes: HashMap<String, TradeEpisode>,
    /// Open episodes indexed by symbol
    open_by_symbol: HashMap<String, Vec<String>>,
    /// Episodes indexed by strategy
    by_strategy: HashMap<String, Vec<String>>,
    /// Episodes indexed by outcome
    by_outcome: HashMap<TradeOutcome, Vec<String>>,
    /// Maximum episodes to retain
    max_episodes: usize,
    /// Episode counter for ID generation
    episode_counter: u64,
    /// Identified patterns
    patterns: Vec<TradePattern>,
}

impl Default for TradeEpisodes {
    fn default() -> Self {
        Self::new()
    }
}

impl TradeEpisodes {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            episodes: HashMap::new(),
            open_by_symbol: HashMap::new(),
            by_strategy: HashMap::new(),
            by_outcome: HashMap::new(),
            max_episodes: 10000,
            episode_counter: 0,
            patterns: Vec::new(),
        }
    }

    /// Create with custom capacity
    pub fn with_capacity(max_episodes: usize) -> Self {
        Self {
            episodes: HashMap::with_capacity(max_episodes),
            open_by_symbol: HashMap::new(),
            by_strategy: HashMap::new(),
            by_outcome: HashMap::new(),
            max_episodes,
            episode_counter: 0,
            patterns: Vec::new(),
        }
    }

    /// Generate a new episode ID
    fn generate_id(&mut self) -> String {
        self.episode_counter += 1;
        format!("ep_{}", self.episode_counter)
    }

    /// Record a new trade entry
    pub fn record_entry(
        &mut self,
        symbol: &str,
        direction: TradeDirection,
        entry_price: f64,
        size: f64,
        strategy_id: &str,
        entry_signal: Option<EntrySignal>,
        entry_context: Option<MarketContext>,
    ) -> Result<String> {
        if entry_price <= 0.0 {
            return Err(Error::InvalidInput(
                "Entry price must be positive".to_string(),
            ));
        }
        if size <= 0.0 {
            return Err(Error::InvalidInput("Size must be positive".to_string()));
        }

        let id = self.generate_id();
        let mut episode = TradeEpisode::new(&id, symbol, direction, entry_price, size, strategy_id);

        if let Some(signal) = entry_signal {
            episode.entry_signal = signal;
        }
        if let Some(context) = entry_context {
            episode.entry_context = context;
        }

        // Index the episode
        self.open_by_symbol
            .entry(symbol.to_string())
            .or_default()
            .push(id.clone());

        self.by_strategy
            .entry(strategy_id.to_string())
            .or_default()
            .push(id.clone());

        self.episodes.insert(id.clone(), episode);

        // Prune if needed
        self.prune_if_needed();

        Ok(id)
    }

    /// Record a trade exit
    pub fn record_exit(
        &mut self,
        episode_id: &str,
        exit_price: f64,
        exit_reason: ExitReason,
        fees: f64,
        exit_context: Option<MarketContext>,
    ) -> Result<()> {
        let episode = self
            .episodes
            .get_mut(episode_id)
            .ok_or_else(|| Error::NotFound(format!("Episode not found: {}", episode_id)))?;

        if !episode.is_open() {
            return Err(Error::InvalidState("Episode already closed".to_string()));
        }

        let symbol = episode.symbol.clone();

        episode.close(exit_price, exit_reason, fees);
        episode.exit_context = exit_context;

        // Update indices
        if let Some(open_list) = self.open_by_symbol.get_mut(&symbol) {
            open_list.retain(|id| id != episode_id);
        }

        let outcome = episode.outcome();
        self.by_outcome
            .entry(outcome)
            .or_default()
            .push(episode_id.to_string());

        Ok(())
    }

    /// Update excursion for open trades
    pub fn update_price(&mut self, symbol: &str, current_price: f64) {
        if let Some(open_ids) = self.open_by_symbol.get(symbol) {
            for id in open_ids {
                if let Some(episode) = self.episodes.get_mut(id) {
                    episode.update_excursion(current_price);
                }
            }
        }
    }

    /// Get an episode by ID
    pub fn get(&self, episode_id: &str) -> Option<&TradeEpisode> {
        self.episodes.get(episode_id)
    }

    /// Get all open episodes for a symbol
    pub fn get_open_for_symbol(&self, symbol: &str) -> Vec<&TradeEpisode> {
        self.open_by_symbol
            .get(symbol)
            .map(|ids| ids.iter().filter_map(|id| self.episodes.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all episodes for a strategy
    pub fn get_by_strategy(&self, strategy_id: &str) -> Vec<&TradeEpisode> {
        self.by_strategy
            .get(strategy_id)
            .map(|ids| ids.iter().filter_map(|id| self.episodes.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get recent episodes
    pub fn get_recent(&self, count: usize) -> Vec<&TradeEpisode> {
        let mut episodes: Vec<_> = self.episodes.values().collect();
        episodes.sort_by(|a, b| b.entry_time.cmp(&a.entry_time));
        episodes.into_iter().take(count).collect()
    }

    /// Get closed episodes only
    pub fn get_closed(&self) -> Vec<&TradeEpisode> {
        self.episodes.values().filter(|e| !e.is_open()).collect()
    }

    /// Calculate statistics for a set of episodes
    pub fn calculate_statistics(&self, episode_ids: &[String]) -> TradeStatistics {
        let episodes: Vec<_> = episode_ids
            .iter()
            .filter_map(|id| self.episodes.get(id))
            .filter(|e| !e.is_open())
            .collect();

        if episodes.is_empty() {
            return TradeStatistics::default();
        }

        let total = episodes.len();
        let mut wins = 0;
        let mut losses = 0;
        let mut breakeven = 0;
        let mut total_win_pnl = 0.0;
        let mut total_loss_pnl = 0.0;
        let mut total_pnl = 0.0;
        let mut total_duration = 0u64;
        let mut total_mfe = 0.0;
        let mut total_mae = 0.0;
        let mut returns = Vec::new();

        for episode in &episodes {
            let pnl = episode.realized_pnl.unwrap_or(0.0);
            total_pnl += pnl;
            total_mfe += episode.mfe;
            total_mae += episode.mae;
            total_duration += episode.duration_secs.unwrap_or(0);

            if let Some(ret) = episode.return_pct {
                returns.push(ret);
            }

            match episode.outcome() {
                TradeOutcome::Win => {
                    wins += 1;
                    total_win_pnl += pnl;
                }
                TradeOutcome::Loss => {
                    losses += 1;
                    total_loss_pnl += pnl.abs();
                }
                TradeOutcome::Breakeven => {
                    breakeven += 1;
                }
                TradeOutcome::Open => {}
            }
        }

        let win_rate = if total > 0 {
            wins as f64 / total as f64
        } else {
            0.0
        };

        let avg_win = if wins > 0 {
            total_win_pnl / wins as f64
        } else {
            0.0
        };

        let avg_loss = if losses > 0 {
            total_loss_pnl / losses as f64
        } else {
            0.0
        };

        let profit_factor = if total_loss_pnl > 0.0 {
            total_win_pnl / total_loss_pnl
        } else if total_win_pnl > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let expectancy = win_rate * avg_win - (1.0 - win_rate) * avg_loss;

        // Calculate Sharpe ratio (simplified)
        let sharpe_ratio = if !returns.is_empty() {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 =
                returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 { mean / std_dev } else { 0.0 }
        } else {
            0.0
        };

        TradeStatistics {
            total_trades: total,
            winning_trades: wins,
            losing_trades: losses,
            breakeven_trades: breakeven,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            expectancy,
            avg_duration_secs: total_duration as f64 / total as f64,
            avg_mfe: total_mfe / total as f64,
            avg_mae: total_mae / total as f64,
            total_pnl,
            sharpe_ratio,
        }
    }

    /// Calculate overall statistics
    pub fn overall_statistics(&self) -> TradeStatistics {
        let all_ids: Vec<_> = self.episodes.keys().cloned().collect();
        self.calculate_statistics(&all_ids)
    }

    /// Calculate statistics by strategy
    pub fn statistics_by_strategy(&self) -> HashMap<String, TradeStatistics> {
        self.by_strategy
            .iter()
            .map(|(strategy, ids)| (strategy.clone(), self.calculate_statistics(ids)))
            .collect()
    }

    /// Find similar episodes based on entry context
    pub fn find_similar(&self, context: &MarketContext, limit: usize) -> Vec<&TradeEpisode> {
        let mut scored: Vec<_> = self
            .episodes
            .values()
            .filter(|e| !e.is_open())
            .map(|e| {
                let score = self.context_similarity(&e.entry_context, context);
                (e, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(e, _)| e).collect()
    }

    /// Calculate similarity between two contexts
    fn context_similarity(&self, a: &MarketContext, b: &MarketContext) -> f64 {
        let mut score = 0.0;
        let mut weights = 0.0;

        // Regime match
        if a.regime == b.regime {
            score += 2.0;
        }
        weights += 2.0;

        // Volatility similarity
        let vol_diff = (a.volatility - b.volatility).abs();
        score += (1.0 - vol_diff.min(1.0)) * 1.5;
        weights += 1.5;

        // Volume similarity
        let vol_ratio = if b.relative_volume > 0.0 {
            (a.relative_volume / b.relative_volume).min(2.0)
        } else {
            1.0
        };
        score += (1.0 - (vol_ratio - 1.0).abs().min(1.0)) * 1.0;
        weights += 1.0;

        // Time of day similarity
        let hour_diff = ((a.hour_of_day as i32 - b.hour_of_day as i32).abs() % 12) as f64;
        score += (1.0 - hour_diff / 12.0) * 0.5;
        weights += 0.5;

        score / weights
    }

    /// Analyze patterns in trade history
    pub fn analyze_patterns(&mut self) -> Vec<TradePattern> {
        self.patterns.clear();

        // Pattern 1: Win rate by market regime
        for regime in [
            MarketRegime::Bullish,
            MarketRegime::Bearish,
            MarketRegime::Ranging,
            MarketRegime::Volatile,
            MarketRegime::Quiet,
        ] {
            let matching: Vec<_> = self
                .episodes
                .iter()
                .filter(|(_, e)| !e.is_open() && e.entry_context.regime == regime)
                .map(|(id, _)| id.clone())
                .collect();

            if matching.len() >= 10 {
                let stats = self.calculate_statistics(&matching);
                let mut criteria = HashMap::new();
                criteria.insert("regime".to_string(), format!("{:?}", regime));

                self.patterns.push(TradePattern {
                    name: format!("{:?} Regime Trades", regime),
                    episode_ids: matching.clone(),
                    win_rate: stats.win_rate,
                    avg_return: stats.expectancy,
                    sample_size: matching.len(),
                    confidence: (matching.len() as f64 / 100.0).min(1.0),
                    criteria,
                });
            }
        }

        // Pattern 2: Win rate by time of day
        for hour_range in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)] {
            let matching: Vec<_> = self
                .episodes
                .iter()
                .filter(|(_, e)| {
                    !e.is_open()
                        && e.entry_context.hour_of_day >= hour_range.0
                        && e.entry_context.hour_of_day < hour_range.1
                })
                .map(|(id, _)| id.clone())
                .collect();

            if matching.len() >= 10 {
                let stats = self.calculate_statistics(&matching);
                let mut criteria = HashMap::new();
                criteria.insert(
                    "time_range".to_string(),
                    format!("{:02}:00-{:02}:00", hour_range.0, hour_range.1),
                );

                self.patterns.push(TradePattern {
                    name: format!("Trades {:02}:00-{:02}:00", hour_range.0, hour_range.1),
                    episode_ids: matching.clone(),
                    win_rate: stats.win_rate,
                    avg_return: stats.expectancy,
                    sample_size: matching.len(),
                    confidence: (matching.len() as f64 / 100.0).min(1.0),
                    criteria,
                });
            }
        }

        // Pattern 3: High volatility vs low volatility
        let vol_median = {
            let mut vols: Vec<_> = self
                .episodes
                .values()
                .filter(|e| !e.is_open())
                .map(|e| e.entry_context.volatility)
                .collect();
            vols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if vols.is_empty() {
                0.0
            } else {
                vols[vols.len() / 2]
            }
        };

        for (name, is_high) in [("High Volatility", true), ("Low Volatility", false)] {
            let matching: Vec<_> = self
                .episodes
                .iter()
                .filter(|(_, e)| {
                    !e.is_open()
                        && if is_high {
                            e.entry_context.volatility >= vol_median
                        } else {
                            e.entry_context.volatility < vol_median
                        }
                })
                .map(|(id, _)| id.clone())
                .collect();

            if matching.len() >= 10 {
                let stats = self.calculate_statistics(&matching);
                let mut criteria = HashMap::new();
                criteria.insert("volatility".to_string(), name.to_string());

                self.patterns.push(TradePattern {
                    name: format!("{} Trades", name),
                    episode_ids: matching.clone(),
                    win_rate: stats.win_rate,
                    avg_return: stats.expectancy,
                    sample_size: matching.len(),
                    confidence: (matching.len() as f64 / 100.0).min(1.0),
                    criteria,
                });
            }
        }

        self.patterns.clone()
    }

    /// Get identified patterns
    pub fn get_patterns(&self) -> &[TradePattern] {
        &self.patterns
    }

    /// Get best performing patterns
    pub fn get_best_patterns(&self, limit: usize) -> Vec<&TradePattern> {
        let mut patterns: Vec<_> = self.patterns.iter().collect();
        patterns.sort_by(|a, b| {
            let score_a = a.win_rate * a.confidence;
            let score_b = b.win_rate * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns.into_iter().take(limit).collect()
    }

    /// Prune old episodes if over capacity
    fn prune_if_needed(&mut self) {
        if self.episodes.len() <= self.max_episodes {
            return;
        }

        // Keep open trades, prune oldest closed trades
        let mut closed: Vec<_> = self
            .episodes
            .iter()
            .filter(|(_, e)| !e.is_open())
            .map(|(id, e)| (id.clone(), e.entry_time))
            .collect();

        closed.sort_by_key(|(_, time)| *time);

        let to_remove = closed.len().saturating_sub(self.max_episodes / 2);
        for (id, _) in closed.into_iter().take(to_remove) {
            if let Some(episode) = self.episodes.remove(&id) {
                // Clean up indices
                if let Some(list) = self.by_strategy.get_mut(&episode.strategy_id) {
                    list.retain(|i| i != &id);
                }
                let outcome = episode.outcome();
                if let Some(list) = self.by_outcome.get_mut(&outcome) {
                    list.retain(|i| i != &id);
                }
            }
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    /// Get episode count
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get open trade count
    pub fn open_count(&self) -> usize {
        self.episodes.values().filter(|e| e.is_open()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_episode() -> TradeEpisodes {
        TradeEpisodes::new()
    }

    #[test]
    fn test_basic() {
        let instance = TradeEpisodes::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_record_entry() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "momentum",
                None,
                None,
            )
            .unwrap();

        assert!(!id.is_empty());
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes.open_count(), 1);
    }

    #[test]
    fn test_record_exit() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "momentum",
                None,
                None,
            )
            .unwrap();

        episodes
            .record_exit(&id, 51000.0, ExitReason::TakeProfit, 10.0, None)
            .unwrap();

        let episode = episodes.get(&id).unwrap();
        assert!(!episode.is_open());
        assert_eq!(episode.outcome(), TradeOutcome::Win);
        assert!(episode.realized_pnl.unwrap() > 0.0);
    }

    #[test]
    fn test_losing_trade() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "momentum",
                None,
                None,
            )
            .unwrap();

        episodes
            .record_exit(&id, 49000.0, ExitReason::StopLoss, 10.0, None)
            .unwrap();

        let episode = episodes.get(&id).unwrap();
        assert_eq!(episode.outcome(), TradeOutcome::Loss);
        assert!(episode.realized_pnl.unwrap() < 0.0);
    }

    #[test]
    fn test_short_trade() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Short,
                50000.0,
                1.0,
                "momentum",
                None,
                None,
            )
            .unwrap();

        // Short wins when price goes down
        episodes
            .record_exit(&id, 49000.0, ExitReason::TakeProfit, 10.0, None)
            .unwrap();

        let episode = episodes.get(&id).unwrap();
        assert_eq!(episode.outcome(), TradeOutcome::Win);
    }

    #[test]
    fn test_update_excursion() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "momentum",
                None,
                None,
            )
            .unwrap();

        // Simulate price movements
        episodes.update_price("BTC-USD", 51000.0); // +1000
        episodes.update_price("BTC-USD", 49000.0); // -1000
        episodes.update_price("BTC-USD", 52000.0); // +2000

        let episode = episodes.get(&id).unwrap();
        assert_eq!(episode.mfe, 2000.0);
        assert_eq!(episode.mae, -1000.0);
    }

    #[test]
    fn test_statistics() {
        let mut episodes = create_test_episode();

        // Create winning trades
        for _i in 0..5 {
            let id = episodes
                .record_entry(
                    "BTC-USD",
                    TradeDirection::Long,
                    50000.0,
                    1.0,
                    "test",
                    None,
                    None,
                )
                .unwrap();
            episodes
                .record_exit(&id, 51000.0, ExitReason::TakeProfit, 10.0, None)
                .unwrap();
        }

        // Create losing trades
        for _i in 0..5 {
            let id = episodes
                .record_entry(
                    "BTC-USD",
                    TradeDirection::Long,
                    50000.0,
                    1.0,
                    "test",
                    None,
                    None,
                )
                .unwrap();
            episodes
                .record_exit(&id, 49500.0, ExitReason::StopLoss, 10.0, None)
                .unwrap();
        }

        let stats = episodes.overall_statistics();
        assert_eq!(stats.total_trades, 10);
        assert_eq!(stats.winning_trades, 5);
        assert_eq!(stats.losing_trades, 5);
        assert!((stats.win_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_get_by_strategy() {
        let mut episodes = create_test_episode();

        episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "strategy_a",
                None,
                None,
            )
            .unwrap();

        episodes
            .record_entry(
                "ETH-USD",
                TradeDirection::Long,
                3000.0,
                10.0,
                "strategy_b",
                None,
                None,
            )
            .unwrap();

        let strategy_a = episodes.get_by_strategy("strategy_a");
        let strategy_b = episodes.get_by_strategy("strategy_b");

        assert_eq!(strategy_a.len(), 1);
        assert_eq!(strategy_b.len(), 1);
        assert_eq!(strategy_a[0].symbol, "BTC-USD");
        assert_eq!(strategy_b[0].symbol, "ETH-USD");
    }

    #[test]
    fn test_invalid_entry() {
        let mut episodes = create_test_episode();

        // Negative price should fail
        let result = episodes.record_entry(
            "BTC-USD",
            TradeDirection::Long,
            -1000.0,
            1.0,
            "test",
            None,
            None,
        );
        assert!(result.is_err());

        // Zero size should fail
        let result = episodes.record_entry(
            "BTC-USD",
            TradeDirection::Long,
            50000.0,
            0.0,
            "test",
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_double_close() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "test",
                None,
                None,
            )
            .unwrap();

        episodes
            .record_exit(&id, 51000.0, ExitReason::TakeProfit, 10.0, None)
            .unwrap();

        // Second close should fail
        let result = episodes.record_exit(&id, 52000.0, ExitReason::TakeProfit, 10.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_efficiency_ratio() {
        let mut episodes = create_test_episode();

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "test",
                None,
                None,
            )
            .unwrap();

        // MFE of 2000
        episodes.update_price("BTC-USD", 52000.0);

        // Exit at 1000 profit
        episodes
            .record_exit(&id, 51000.0, ExitReason::TakeProfit, 0.0, None)
            .unwrap();

        let episode = episodes.get(&id).unwrap();
        let efficiency = episode.efficiency().unwrap();
        assert!((efficiency - 0.5).abs() < 0.01); // 1000 / 2000 = 50%
    }

    #[test]
    fn test_entry_with_context() {
        let mut episodes = create_test_episode();

        let signal = EntrySignal {
            source: "RSI_oversold".to_string(),
            strength: 0.8,
            confidence: 0.9,
            metadata: HashMap::new(),
        };

        let context = MarketContext {
            price: 50000.0,
            volatility: 0.3,
            relative_volume: 1.5,
            spread_pct: 0.01,
            regime: MarketRegime::Bullish,
            hour_of_day: 14,
            day_of_week: 2,
            indicators: HashMap::new(),
        };

        let id = episodes
            .record_entry(
                "BTC-USD",
                TradeDirection::Long,
                50000.0,
                1.0,
                "test",
                Some(signal),
                Some(context),
            )
            .unwrap();

        let episode = episodes.get(&id).unwrap();
        assert_eq!(episode.entry_signal.source, "RSI_oversold");
        assert_eq!(episode.entry_context.regime, MarketRegime::Bullish);
    }
}
