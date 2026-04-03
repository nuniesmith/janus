//! # Strategy Gating
//!
//! Decides which strategies should run for a given asset based on:
//! - Current market regime (from `janus-regime`)
//! - Strategy affinity weights (from [`super::affinity`])
//! - Static configuration (TOML-loaded allow/deny lists)
//!
//! The gate acts as a filter in the signal pipeline: before a strategy
//! processes a tick, the gate checks whether it should run at all.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_strategies::gating::{StrategyGate, StrategyGatingConfig, AssetStrategyConfig};
//! use janus_strategies::affinity::StrategyAffinityTracker;
//!
//! let config = StrategyGatingConfig::default();
//! let tracker = StrategyAffinityTracker::new(10);
//! let gate = StrategyGate::new(config, tracker);
//!
//! // Check if a strategy should run
//! let regime = janus_regime::MarketRegime::MeanReverting;
//! if gate.should_run("BollingerSqueeze", "BTCUSD", &regime) {
//!     // run the strategy
//! }
//! ```

use crate::affinity::StrategyAffinityTracker;
use janus_regime::MarketRegime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level gating configuration, typically loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyGatingConfig {
    /// Global minimum affinity weight to enable a strategy.
    /// Strategies whose weight falls below this are disabled.
    #[serde(default = "default_min_weight")]
    pub min_weight: f64,

    /// Whether to allow strategies that have no recorded trades yet
    /// (i.e. untested strategies get the benefit of the doubt).
    #[serde(default = "default_allow_untested")]
    pub allow_untested: bool,

    /// Per-asset configuration overrides.
    #[serde(default)]
    pub assets: HashMap<String, AssetStrategyConfig>,
}

fn default_min_weight() -> f64 {
    0.3
}

fn default_allow_untested() -> bool {
    true
}

impl Default for StrategyGatingConfig {
    fn default() -> Self {
        Self {
            min_weight: default_min_weight(),
            allow_untested: default_allow_untested(),
            assets: HashMap::new(),
        }
    }
}

/// Per-asset strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AssetStrategyConfig {
    /// Strategies that are explicitly enabled for this asset.
    /// If non-empty, only these strategies are considered (allowlist).
    #[serde(default)]
    pub enabled_strategies: Vec<String>,

    /// Strategies that are explicitly disabled for this asset.
    /// These are never run, regardless of affinity or regime.
    #[serde(default)]
    pub disabled_strategies: Vec<String>,

    /// Strategies preferred for each regime category.
    /// Key is a regime label: `"TrendFollowing"`, `"MeanReverting"`, etc.
    #[serde(default)]
    pub preferred_regime_strategies: HashMap<String, Vec<String>>,
}

/// The strategy gate — central decision point for "should this strategy run?"
#[derive(Debug, Clone)]
pub struct StrategyGate {
    config: StrategyGatingConfig,
    tracker: StrategyAffinityTracker,
}

impl StrategyGate {
    /// Create a new gate from a config and an affinity tracker.
    pub fn new(config: StrategyGatingConfig, tracker: StrategyAffinityTracker) -> Self {
        Self { config, tracker }
    }

    /// Replace the affinity tracker (e.g. after loading state from Redis).
    pub fn set_tracker(&mut self, tracker: StrategyAffinityTracker) {
        self.tracker = tracker;
    }

    /// Get a mutable reference to the affinity tracker (e.g. to record trade results).
    pub fn tracker_mut(&mut self) -> &mut StrategyAffinityTracker {
        &mut self.tracker
    }

    /// Get a shared reference to the affinity tracker.
    pub fn tracker(&self) -> &StrategyAffinityTracker {
        &self.tracker
    }

    /// Get a shared reference to the gating config.
    pub fn config(&self) -> &StrategyGatingConfig {
        &self.config
    }

    /// Decide whether `strategy` should run on `asset` given the current `regime`.
    ///
    /// The decision pipeline:
    /// 1. Check the asset's disabled list → block if present
    /// 2. Check the asset's enabled list → block if non-empty and strategy absent
    /// 3. Check regime compatibility → block if regime has preferred strategies
    ///    and this strategy isn't among them
    /// 4. Check affinity weight → block if below threshold
    pub fn should_run(&self, strategy: &str, asset: &str, regime: &MarketRegime) -> bool {
        // Step 1: Explicit deny list
        if self.is_explicitly_disabled(strategy, asset) {
            return false;
        }

        // Step 2: Explicit allow list (if set, acts as allowlist)
        if !self.passes_allowlist(strategy, asset) {
            return false;
        }

        // Step 3: Regime compatibility
        if !self.is_regime_compatible(strategy, asset, regime) {
            return false;
        }

        // Step 4: Affinity weight
        self.passes_affinity_check(strategy, asset)
    }

    /// Return the list of strategies that should run on `asset` given `regime`,
    /// drawn from `all_strategies`.
    pub fn enabled_strategies<'a>(
        &self,
        asset: &str,
        regime: &MarketRegime,
        all_strategies: &'a [String],
    ) -> Vec<&'a str> {
        all_strategies
            .iter()
            .filter(|s| self.should_run(s, asset, regime))
            .map(String::as_str)
            .collect()
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal checks
    // ────────────────────────────────────────────────────────────────────

    /// Check if the strategy is on the asset's disabled list.
    fn is_explicitly_disabled(&self, strategy: &str, asset: &str) -> bool {
        self.config.assets.get(asset).is_some_and(|cfg| {
            cfg.disabled_strategies
                .iter()
                .any(|s| s.eq_ignore_ascii_case(strategy))
        })
    }

    /// Check allowlist. Returns `true` if:
    /// - No allowlist is configured for this asset (everything allowed), OR
    /// - The strategy is on the allowlist.
    fn passes_allowlist(&self, strategy: &str, asset: &str) -> bool {
        match self.config.assets.get(asset) {
            None => true,
            Some(cfg) => {
                if cfg.enabled_strategies.is_empty() {
                    true // no allowlist → everything allowed
                } else {
                    cfg.enabled_strategies
                        .iter()
                        .any(|s| s.eq_ignore_ascii_case(strategy))
                }
            }
        }
    }

    /// Check if the strategy is compatible with the current regime.
    ///
    /// If the asset has preferred strategies for this regime category, only
    /// those strategies pass. If no preferences are set, all strategies pass.
    fn is_regime_compatible(&self, strategy: &str, asset: &str, regime: &MarketRegime) -> bool {
        let asset_config = match self.config.assets.get(asset) {
            Some(cfg) => cfg,
            None => return true, // no config → all compatible
        };

        let regime_key = regime_to_key(regime);

        match asset_config.preferred_regime_strategies.get(regime_key) {
            None => true, // no preferences for this regime → all compatible
            Some(preferred) => {
                if preferred.is_empty() {
                    true
                } else {
                    preferred.iter().any(|s| s.eq_ignore_ascii_case(strategy))
                }
            }
        }
    }

    /// Check affinity weight against the minimum threshold.
    fn passes_affinity_check(&self, strategy: &str, asset: &str) -> bool {
        let weight = self.tracker.weight_for(strategy, asset);

        // If weight is exactly 0.5, the strategy has no data yet.
        // Respect the `allow_untested` flag.
        if (weight - 0.5).abs() < f64::EPSILON {
            return self.config.allow_untested;
        }

        weight >= self.config.min_weight
    }
}

/// Map a `MarketRegime` to the string key used in TOML config.
fn regime_to_key(regime: &MarketRegime) -> &'static str {
    match regime {
        MarketRegime::Trending(_) => "TrendFollowing",
        MarketRegime::MeanReverting => "MeanReverting",
        MarketRegime::Volatile => "Volatile",
        MarketRegime::Uncertain => "Uncertain",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use janus_regime::TrendDirection;

    fn make_gate() -> StrategyGate {
        let mut assets = HashMap::new();

        let btc_config = AssetStrategyConfig {
            enabled_strategies: vec![
                "EMAFlip".to_string(),
                "TrendPullback".to_string(),
                "MomentumSurge".to_string(),
                "BollingerSqueeze".to_string(),
                "MeanReversion".to_string(),
            ],
            disabled_strategies: vec!["BadStrategy".to_string()],
            preferred_regime_strategies: HashMap::from([
                (
                    "TrendFollowing".to_string(),
                    vec!["EMAFlip".to_string(), "TrendPullback".to_string()],
                ),
                (
                    "MeanReverting".to_string(),
                    vec!["BollingerSqueeze".to_string(), "MeanReversion".to_string()],
                ),
            ]),
        };

        assets.insert("BTCUSD".to_string(), btc_config);

        let config = StrategyGatingConfig {
            min_weight: 0.3,
            allow_untested: true,
            assets,
        };

        let tracker = StrategyAffinityTracker::new(5);
        StrategyGate::new(config, tracker)
    }

    #[test]
    fn test_disabled_strategy_blocked() {
        let gate = make_gate();
        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        assert!(!gate.should_run("BadStrategy", "BTCUSD", &regime));
    }

    #[test]
    fn test_allowlist_blocks_unlisted() {
        let gate = make_gate();
        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        // "VWAPScalper" is not in BTCUSD's enabled_strategies
        assert!(!gate.should_run("VWAPScalper", "BTCUSD", &regime));
    }

    #[test]
    fn test_allowlist_passes_listed() {
        let gate = make_gate();
        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        assert!(gate.should_run("EMAFlip", "BTCUSD", &regime));
    }

    #[test]
    fn test_regime_filters_strategies() {
        let gate = make_gate();

        // In trending regime, only TrendFollowing strategies should pass
        let trending = MarketRegime::Trending(TrendDirection::Bullish);
        assert!(gate.should_run("EMAFlip", "BTCUSD", &trending));
        assert!(gate.should_run("TrendPullback", "BTCUSD", &trending));
        assert!(!gate.should_run("MeanReversion", "BTCUSD", &trending));
        assert!(!gate.should_run("BollingerSqueeze", "BTCUSD", &trending));

        // In mean-reverting regime, only MeanReverting strategies should pass
        let mean_rev = MarketRegime::MeanReverting;
        assert!(!gate.should_run("EMAFlip", "BTCUSD", &mean_rev));
        assert!(gate.should_run("BollingerSqueeze", "BTCUSD", &mean_rev));
        assert!(gate.should_run("MeanReversion", "BTCUSD", &mean_rev));
    }

    #[test]
    fn test_no_regime_preference_allows_all() {
        let gate = make_gate();

        // Volatile regime has no preference set → all enabled strategies pass
        let volatile = MarketRegime::Volatile;
        assert!(gate.should_run("EMAFlip", "BTCUSD", &volatile));
        assert!(gate.should_run("MeanReversion", "BTCUSD", &volatile));
    }

    #[test]
    fn test_unknown_asset_allows_all() {
        let gate = make_gate();
        let regime = MarketRegime::Trending(TrendDirection::Bearish);
        // SOLUSD has no config → everything allowed
        assert!(gate.should_run("EMAFlip", "SOLUSD", &regime));
        assert!(gate.should_run("RandomStrat", "SOLUSD", &regime));
    }

    #[test]
    fn test_affinity_blocks_bad_strategy() {
        let mut gate = make_gate();
        // Record bad performance for MomentumSurge on BTC
        for _ in 0..10 {
            gate.tracker_mut()
                .record_trade_result("MomentumSurge", "BTCUSD", -200.0, false);
        }

        // Volatile regime (no regime preference) so only affinity matters
        let volatile = MarketRegime::Volatile;
        // Weight should be < 0.3 due to consistently negative performance
        assert!(
            !gate.should_run("MomentumSurge", "BTCUSD", &volatile),
            "Badly performing strategy should be gated out"
        );
    }

    #[test]
    fn test_untested_strategy_allowed_by_default() {
        let gate = make_gate();
        // EMAFlip has no trade data but allow_untested is true
        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        assert!(gate.should_run("EMAFlip", "BTCUSD", &regime));
    }

    #[test]
    fn test_untested_strategy_blocked_when_flag_off() {
        let config = StrategyGatingConfig {
            min_weight: 0.3,
            allow_untested: false, // strict mode
            assets: HashMap::new(),
        };
        let tracker = StrategyAffinityTracker::new(5);
        let gate = StrategyGate::new(config, tracker);

        let regime = MarketRegime::Trending(TrendDirection::Bullish);
        assert!(
            !gate.should_run("EMAFlip", "BTCUSD", &regime),
            "Untested strategy should be blocked when allow_untested is false"
        );
    }

    #[test]
    fn test_enabled_strategies_filters_list() {
        let gate = make_gate();
        let all = vec![
            "EMAFlip".to_string(),
            "TrendPullback".to_string(),
            "MeanReversion".to_string(),
            "BollingerSqueeze".to_string(),
            "BadStrategy".to_string(),
        ];

        let trending = MarketRegime::Trending(TrendDirection::Bullish);
        let enabled = gate.enabled_strategies("BTCUSD", &trending, &all);
        assert!(enabled.contains(&"EMAFlip"));
        assert!(enabled.contains(&"TrendPullback"));
        assert!(!enabled.contains(&"MeanReversion"));
        assert!(!enabled.contains(&"BadStrategy"));
    }

    #[test]
    fn test_regime_to_key() {
        assert_eq!(
            regime_to_key(&MarketRegime::Trending(TrendDirection::Bullish)),
            "TrendFollowing"
        );
        assert_eq!(
            regime_to_key(&MarketRegime::Trending(TrendDirection::Bearish)),
            "TrendFollowing"
        );
        assert_eq!(regime_to_key(&MarketRegime::MeanReverting), "MeanReverting");
        assert_eq!(regime_to_key(&MarketRegime::Volatile), "Volatile");
        assert_eq!(regime_to_key(&MarketRegime::Uncertain), "Uncertain");
    }

    #[test]
    fn test_case_insensitive_matching() {
        let mut assets = HashMap::new();
        assets.insert(
            "BTCUSD".to_string(),
            AssetStrategyConfig {
                enabled_strategies: vec!["EMAFlip".to_string()],
                disabled_strategies: vec!["badstrat".to_string()],
                preferred_regime_strategies: HashMap::new(),
            },
        );
        let config = StrategyGatingConfig {
            min_weight: 0.3,
            allow_untested: true,
            assets,
        };
        let tracker = StrategyAffinityTracker::new(5);
        let gate = StrategyGate::new(config, tracker);

        let regime = MarketRegime::MeanReverting;
        // Should match case-insensitively
        assert!(gate.should_run("emaflip", "BTCUSD", &regime));
        assert!(!gate.should_run("BadStrat", "BTCUSD", &regime));
    }

    #[test]
    fn test_default_config() {
        let config = StrategyGatingConfig::default();
        assert!((config.min_weight - 0.3).abs() < f64::EPSILON);
        assert!(config.allow_untested);
        assert!(config.assets.is_empty());
    }

    #[test]
    fn test_tracker_access() {
        let mut gate = make_gate();

        gate.tracker_mut()
            .record_trade_result("EMAFlip", "BTCUSD", 100.0, true);

        let perf = gate.tracker().get_performance("EMAFlip", "BTCUSD");
        assert!(perf.is_some());
        assert_eq!(perf.unwrap().total_trades, 1);
    }

    #[test]
    fn test_toml_config_deserialization() {
        let toml_str = r#"
min_weight      = 0.30
allow_untested  = true

[assets.BTCUSD]
enabled_strategies  = ["EMAFlip", "TrendPullback", "MomentumSurge"]
disabled_strategies = ["MeanReversion"]

[assets.BTCUSD.preferred_regime_strategies]
TrendFollowing = ["EMAFlip", "TrendPullback"]
MeanReverting  = ["BollingerSqueeze"]

[assets.ETHUSD]
enabled_strategies  = ["EMAFlip", "MeanReversion", "VWAPScalper"]
disabled_strategies = ["MomentumSurge"]

[assets.ETHUSD.preferred_regime_strategies]
TrendFollowing = ["EMAFlip", "MultiTfTrend"]
MeanReverting  = ["VWAPScalper", "MeanReversion"]

[assets.SOLUSD]
enabled_strategies  = ["EMAFlip", "MomentumSurge", "OpeningRange"]
disabled_strategies = ["MultiTfTrend"]

[assets.SOLUSD.preferred_regime_strategies]
TrendFollowing = ["EMAFlip", "MomentumSurge"]
MeanReverting  = ["OpeningRange"]
Volatile       = ["MomentumSurge"]
"#;

        let config: StrategyGatingConfig =
            toml::from_str(toml_str).expect("TOML should deserialize into StrategyGatingConfig");

        // Global settings
        assert!((config.min_weight - 0.3).abs() < f64::EPSILON);
        assert!(config.allow_untested);

        // BTCUSD
        let btc = config.assets.get("BTCUSD").expect("BTCUSD should exist");
        assert_eq!(btc.enabled_strategies.len(), 3);
        assert!(btc.enabled_strategies.contains(&"EMAFlip".to_string()));
        assert!(
            btc.enabled_strategies
                .contains(&"TrendPullback".to_string())
        );
        assert!(
            btc.enabled_strategies
                .contains(&"MomentumSurge".to_string())
        );
        assert_eq!(btc.disabled_strategies, vec!["MeanReversion"]);
        assert_eq!(
            btc.preferred_regime_strategies
                .get("TrendFollowing")
                .unwrap(),
            &vec!["EMAFlip".to_string(), "TrendPullback".to_string()]
        );
        assert_eq!(
            btc.preferred_regime_strategies
                .get("MeanReverting")
                .unwrap(),
            &vec!["BollingerSqueeze".to_string()]
        );

        // ETHUSD
        let eth = config.assets.get("ETHUSD").expect("ETHUSD should exist");
        assert_eq!(eth.enabled_strategies.len(), 3);
        assert_eq!(eth.disabled_strategies, vec!["MomentumSurge"]);

        // SOLUSD
        let sol = config.assets.get("SOLUSD").expect("SOLUSD should exist");
        assert_eq!(sol.enabled_strategies.len(), 3);
        assert_eq!(sol.disabled_strategies, vec!["MultiTfTrend"]);
        assert_eq!(
            sol.preferred_regime_strategies.get("Volatile").unwrap(),
            &vec!["MomentumSurge".to_string()]
        );
    }

    #[test]
    fn test_toml_config_round_trip_with_gate() {
        let toml_str = r#"
min_weight      = 0.25
allow_untested  = false

[assets.BTCUSD]
enabled_strategies  = ["EMAFlip", "TrendPullback"]
disabled_strategies = ["BadStrat"]

[assets.BTCUSD.preferred_regime_strategies]
TrendFollowing = ["EMAFlip", "TrendPullback"]
MeanReverting  = []
"#;

        let config: StrategyGatingConfig = toml::from_str(toml_str).expect("TOML should parse");

        let tracker = StrategyAffinityTracker::new(5);
        let gate = StrategyGate::new(config, tracker);

        let trending = MarketRegime::Trending(TrendDirection::Bullish);
        let mean_rev = MarketRegime::MeanReverting;

        // allow_untested = false, so untested strategies are blocked by affinity
        // BUT regime gating still applies first:
        // In trending: only EMAFlip, TrendPullback preferred → those pass regime check
        // However they're untested → affinity blocks them
        assert!(!gate.should_run("EMAFlip", "BTCUSD", &trending));

        // BadStrat is explicitly disabled — blocked regardless
        assert!(!gate.should_run("BadStrat", "BTCUSD", &trending));

        // Unknown asset has no config → all pass allowlist & regime, but
        // allow_untested=false blocks at affinity
        assert!(!gate.should_run("EMAFlip", "XYZUSD", &mean_rev));
    }
}
