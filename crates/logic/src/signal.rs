//! Signal validation and filtering logic.
//!
//! Implements signal validation based on user profiles, trading modes,
//! asset classes, and risk tolerance. This is the "Feudal Hierarchy"
//! filtering system that ensures signals match user constraints.

use common::{JanusError, Result, Signal};
use serde::{Deserialize, Serialize};

/// Trading mode (SCALP, SWING, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingMode {
    Scalp,
    Swing,
    Day,
    Position,
}

impl TradingMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            TradingMode::Scalp => "SCALP",
            TradingMode::Swing => "SWING",
            TradingMode::Day => "DAY",
            TradingMode::Position => "POSITION",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "SCALP" => TradingMode::Scalp,
            "DAY" => TradingMode::Day,
            "POSITION" => TradingMode::Position,
            _ => TradingMode::Swing,
        }
    }
}

/// Asset class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetClass {
    Crypto,
    Bitcoin,
    Stocks,
    Forex,
    Commodities,
}

impl AssetClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            AssetClass::Crypto => "CRYPTO",
            AssetClass::Bitcoin => "BITCOIN",
            AssetClass::Stocks => "STOCKS",
            AssetClass::Forex => "FOREX",
            AssetClass::Commodities => "COMMODITIES",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "BITCOIN" => AssetClass::Bitcoin,
            "STOCKS" => AssetClass::Stocks,
            "FOREX" => AssetClass::Forex,
            "COMMODITIES" => AssetClass::Commodities,
            _ => AssetClass::Crypto,
        }
    }
}

/// User profile with trading constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub risk_tolerance: f64, // 0.0 (conservative) to 1.0 (aggressive)
    pub asset_class: AssetClass,
    pub trading_mode: TradingMode,
    #[serde(default)]
    pub allowed_symbols: Vec<String>,
}

impl UserProfile {
    /// Create a default user profile
    pub fn new(user_id: String, mode: TradingMode) -> Self {
        Self {
            user_id,
            risk_tolerance: 0.5,
            asset_class: AssetClass::Crypto,
            trading_mode: mode,
            allowed_symbols: Vec::new(),
        }
    }
}

/// Signal validator that checks signals against user profiles
pub struct SignalValidator;

impl SignalValidator {
    /// Validate a signal against a user profile
    ///
    /// Rules:
    /// 1. If profile.mode == SCALP, reject signals with predicted duration > 15m
    /// 2. If profile.asset_class == BITCOIN, only allow BTC/USD signals
    /// 3. Check risk tolerance against signal strength
    /// 4. Check allowed symbols
    /// 5. If profile.mode == SWING, reject signals with duration < 1h
    pub fn validate_signal(signal: &Signal, profile: &UserProfile) -> Result<()> {
        // Rule 1: SCALP mode - reject signals with duration > 15 minutes
        if profile.trading_mode == TradingMode::Scalp {
            if let Some(duration_sec) = signal.predicted_duration_seconds {
                let duration_min = duration_sec / 60;
                if duration_min > 15 {
                    return Err(JanusError::LogicViolation(format!(
                        "Signal rejected: SCALP mode requires duration <= 15m, got {}m",
                        duration_min
                    )));
                }
            }
        }

        // Rule 2: BITCOIN asset class - only allow BTC symbols
        if profile.asset_class == AssetClass::Bitcoin {
            let symbol_upper = signal.symbol.to_uppercase();
            if !symbol_upper.starts_with("BTC") && !symbol_upper.contains("BTC") {
                return Err(JanusError::LogicViolation(format!(
                    "Signal rejected: BITCOIN asset class only allows BTC symbols, got {}",
                    signal.symbol
                )));
            }
        }

        // Rule 3: Risk tolerance check
        // High strength signals might require higher risk tolerance
        if signal.strength > 0.8 && profile.risk_tolerance < 0.5 {
            return Err(JanusError::LogicViolation(format!(
                "Signal rejected: High strength signal ({:.2}) requires higher risk tolerance (got {:.2})",
                signal.strength, profile.risk_tolerance
            )));
        }

        // Rule 4: Allowed symbols check
        if !profile.allowed_symbols.is_empty() {
            if !profile.allowed_symbols.contains(&signal.symbol) {
                return Err(JanusError::LogicViolation(format!(
                    "Signal rejected: Symbol {} not in allowed list: {:?}",
                    signal.symbol, profile.allowed_symbols
                )));
            }
        }

        // Rule 5: SWING mode - reject very short duration signals
        if profile.trading_mode == TradingMode::Swing {
            if let Some(duration_sec) = signal.predicted_duration_seconds {
                let duration_min = duration_sec / 60;
                if duration_min < 60 {
                    return Err(JanusError::LogicViolation(format!(
                        "Signal rejected: SWING mode requires duration >= 1h, got {}m",
                        duration_min
                    )));
                }
            }
        }

        Ok(())
    }

    /// Create a default user profile
    pub fn create_default_profile(user_id: String, mode: TradingMode) -> UserProfile {
        UserProfile::new(user_id, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::OrderSide;

    fn create_test_signal() -> Signal {
        Signal {
            symbol: "BTC/USD".to_string(),
            side: OrderSide::Buy,
            strength: 0.7,
            confidence: 0.8,
            predicted_duration_seconds: Some(900), // 15 minutes
        }
    }

    #[test]
    fn test_scalp_mode_validation() {
        let signal = create_test_signal();
        let profile = UserProfile::new("user1".to_string(), TradingMode::Scalp);

        // 15 minutes should pass for SCALP
        assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());

        // 16 minutes should fail for SCALP
        let mut long_signal = signal.clone();
        long_signal.predicted_duration_seconds = Some(960); // 16 minutes
        assert!(SignalValidator::validate_signal(&long_signal, &profile).is_err());
    }

    #[test]
    fn test_bitcoin_asset_class_validation() {
        let mut signal = create_test_signal();
        signal.symbol = "BTC/USD".to_string();
        signal.predicted_duration_seconds = Some(3600); // 1 hour for SWING mode
        let mut profile = UserProfile::new("user1".to_string(), TradingMode::Swing);
        profile.asset_class = AssetClass::Bitcoin;

        // BTC symbol should pass
        assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());

        // ETH symbol should fail
        signal.symbol = "ETH/USD".to_string();
        assert!(SignalValidator::validate_signal(&signal, &profile).is_err());
    }

    #[test]
    fn test_risk_tolerance_validation() {
        let mut signal = create_test_signal();
        signal.strength = 0.9; // High strength
        signal.predicted_duration_seconds = Some(3600); // 1 hour for SWING mode
        let mut profile = UserProfile::new("user1".to_string(), TradingMode::Swing);
        profile.risk_tolerance = 0.3; // Low risk tolerance

        // High strength + low risk tolerance should fail
        assert!(SignalValidator::validate_signal(&signal, &profile).is_err());

        // Increase risk tolerance
        profile.risk_tolerance = 0.6;
        assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());
    }

    #[test]
    fn test_allowed_symbols_validation() {
        let mut signal = create_test_signal();
        signal.predicted_duration_seconds = Some(3600); // 1 hour for SWING mode
        let mut profile = UserProfile::new("user1".to_string(), TradingMode::Swing);
        profile.allowed_symbols = vec!["ETH/USD".to_string(), "SOL/USD".to_string()];

        // BTC not in allowed list should fail
        assert!(SignalValidator::validate_signal(&signal, &profile).is_err());

        // Add BTC to allowed list
        profile.allowed_symbols.push("BTC/USD".to_string());
        assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());
    }

    #[test]
    fn test_swing_mode_validation() {
        let mut signal = create_test_signal();
        signal.predicted_duration_seconds = Some(3600); // 1 hour
        let profile = UserProfile::new("user1".to_string(), TradingMode::Swing);

        // 1 hour should pass for SWING
        assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());

        // 30 minutes should fail for SWING
        signal.predicted_duration_seconds = Some(1800);
        assert!(SignalValidator::validate_signal(&signal, &profile).is_err());
    }
}
