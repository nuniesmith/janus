//! Domain types for Project JANUS.
//!
//! These types use the Newtype pattern to prevent catastrophic unit errors
//! and ensure type safety across the system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

#[cfg(feature = "ndarray")]
use ndarray::Array2;

/// Price in the base currency (e.g., USD)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Price(pub f64);

impl Price {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.0)
    }
}

/// Volume/Quantity in the asset units
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Volume(pub f64);

impl Volume {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Volume {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.8}", self.0)
    }
}

/// Market tick - the atomic unit of market data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bid: Price,
    pub ask: Price,
    pub bid_size: Volume,
    pub ask_size: Volume,
    pub last_price: Option<Price>,
    pub last_size: Option<Volume>,
}

impl Tick {
    pub fn mid_price(&self) -> Price {
        Price((self.bid.value() + self.ask.value()) / 2.0)
    }

    pub fn spread(&self) -> Price {
        Price(self.ask.value() - self.bid.value())
    }
}

/// OHLCV Candle - aggregated time window data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Volume,
    pub interval_seconds: u64,
}

impl Candle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: String,
        timestamp: DateTime<Utc>,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: Volume,
        interval_seconds: u64,
    ) -> Self {
        Self {
            symbol,
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            interval_seconds,
        }
    }

    pub fn body_size(&self) -> Price {
        Price((self.close.value() - self.open.value()).abs())
    }

    pub fn range(&self) -> Price {
        Price(self.high.value() - self.low.value())
    }
}

/// Order side (Buy or Sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order state for typestate pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    Unchecked,
    Verified,
    Submitted,
    Filled,
    Cancelled,
    Rejected,
}

/// Order - represents a trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: uuid::Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Volume,
    pub price: Option<Price>,      // None for market orders
    pub stop_price: Option<Price>, // For stop orders
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
}

impl Order {
    pub fn new(
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Option<Price>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            symbol,
            side,
            order_type,
            quantity,
            price,
            stop_price: None,
            timestamp: Utc::now(),
            user_id: None,
        }
    }
}

/// Trade - represents an executed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: uuid::Uuid,
    pub order_id: uuid::Uuid,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Volume,
    pub price: Price,
    pub timestamp: DateTime<Utc>,
    pub fee: Option<Price>,
}

impl Trade {
    pub fn new(
        order_id: uuid::Uuid,
        symbol: String,
        side: OrderSide,
        quantity: Volume,
        price: Price,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            order_id,
            symbol,
            side,
            quantity,
            price,
            timestamp: Utc::now(),
            fee: None,
        }
    }

    pub fn notional_value(&self) -> Price {
        Price(self.quantity.value() * self.price.value())
    }
}

/// Market regime identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    pub id: String,
    pub name: String,
    pub features: Vec<f64>, // Vector embedding for similarity search
    pub volatility: f64,
    pub trend: f64, // -1.0 (bearish) to 1.0 (bullish)
}

/// Action type for trading decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ActionType {
    /// Buy the asset
    Buy,
    /// Sell the asset
    Sell,
    /// Hold current position (no action)
    Hold,
    /// Close all positions
    Close,
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionType::Buy => write!(f, "BUY"),
            ActionType::Sell => write!(f, "SELL"),
            ActionType::Hold => write!(f, "HOLD"),
            ActionType::Close => write!(f, "CLOSE"),
        }
    }
}

/// Trading action with parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Type of action (Buy, Sell, Hold, Close)
    pub action_type: ActionType,
    /// Symbol to trade
    pub symbol: String,
    /// Quantity to trade
    pub quantity: f32,
    /// Optional limit price
    pub price: Option<f32>,
    /// Timestamp when action was decided
    pub timestamp: DateTime<Utc>,
}

impl Action {
    pub fn new(action_type: ActionType, symbol: String, quantity: f32) -> Self {
        Self {
            action_type,
            symbol,
            quantity,
            price: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_price(mut self, price: f32) -> Self {
        self.price = Some(price);
        self
    }
}

/// Metadata associated with a market state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    /// Trading symbol
    pub symbol: String,
    /// State timestamp
    pub timestamp: DateTime<Utc>,
    /// Market regime identifier
    pub regime: Option<String>,
    /// Volatility measure
    pub volatility: Option<f32>,
    /// Volume profile
    pub volume: Option<f32>,
}

impl StateMetadata {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            timestamp: Utc::now(),
            regime: None,
            volatility: None,
            volume: None,
        }
    }
}

/// Market state representation for RL agent
///
/// This struct holds the encoded market state using GAF (Gramian Angular Field)
/// features along with supplementary raw features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// GAF-encoded features as 2D image (when ndarray feature enabled)
    #[cfg(feature = "ndarray")]
    pub gaf_features: Array2<f32>,

    /// Flattened GAF features (always available)
    pub gaf_features_flat: Vec<f32>,

    /// Raw supplementary features (price, volume, technical indicators)
    pub raw_features: Vec<f32>,

    /// Associated metadata
    pub metadata: StateMetadata,
}

impl State {
    /// Create a new state from flattened GAF features
    pub fn from_flat_gaf(
        gaf_features: Vec<f32>,
        raw_features: Vec<f32>,
        metadata: StateMetadata,
    ) -> Self {
        Self {
            #[cfg(feature = "ndarray")]
            gaf_features: {
                let size = (gaf_features.len() as f32).sqrt() as usize;
                Array2::from_shape_vec((size, size), gaf_features.clone())
                    .unwrap_or_else(|_| Array2::zeros((32, 32)))
            },
            gaf_features_flat: gaf_features,
            raw_features,
            metadata,
        }
    }

    #[cfg(feature = "ndarray")]
    pub fn from_gaf_matrix(
        gaf_features: Array2<f32>,
        raw_features: Vec<f32>,
        metadata: StateMetadata,
    ) -> Self {
        let gaf_features_flat = gaf_features.iter().cloned().collect();
        Self {
            gaf_features,
            gaf_features_flat,
            raw_features,
            metadata,
        }
    }

    /// Get total feature dimension
    pub fn feature_dim(&self) -> usize {
        self.gaf_features_flat.len() + self.raw_features.len()
    }

    /// Get combined feature vector
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_dim());
        features.extend_from_slice(&self.gaf_features_flat);
        features.extend_from_slice(&self.raw_features);
        features
    }
}

/// Experience tuple for reinforcement learning (Enhanced for neuromorphic system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Current state
    pub state: State,

    /// Action taken
    pub action: Action,

    /// Reward received (e.g., PnL, Sharpe ratio, shaped reward)
    pub reward: f32,

    /// Next state after action
    pub next_state: State,

    /// Episode termination flag
    pub done: bool,

    /// Priority for prioritized experience replay (TD-error based)
    pub priority: f32,

    /// Timestamp when experience was created
    pub timestamp: DateTime<Utc>,

    /// Optional episode ID for multi-episode tracking
    pub episode_id: Option<uuid::Uuid>,
}

impl Experience {
    pub fn new(state: State, action: Action, reward: f32, next_state: State, done: bool) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
            priority: 1.0, // Default max priority for new experiences
            timestamp: Utc::now(),
            episode_id: None,
        }
    }

    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_episode_id(mut self, episode_id: uuid::Uuid) -> Self {
        self.episode_id = Some(episode_id);
        self
    }
}

/// Simple experience for compatibility (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleExperience {
    pub state: Vec<f64>,
    pub action: i32,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
    pub priority: f64,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_type_display() {
        assert_eq!(ActionType::Buy.to_string(), "BUY");
        assert_eq!(ActionType::Sell.to_string(), "SELL");
        assert_eq!(ActionType::Hold.to_string(), "HOLD");
        assert_eq!(ActionType::Close.to_string(), "CLOSE");
    }

    #[test]
    fn test_action_creation() {
        let action = Action::new(ActionType::Buy, "BTC/USD".to_string(), 1.5);

        assert_eq!(action.action_type, ActionType::Buy);
        assert_eq!(action.symbol, "BTC/USD");
        assert_eq!(action.quantity, 1.5);
        assert!(action.price.is_none());
    }

    #[test]
    fn test_action_with_price() {
        let action = Action::new(ActionType::Sell, "ETH/USD".to_string(), 2.0).with_price(3000.0);

        assert_eq!(action.action_type, ActionType::Sell);
        assert_eq!(action.price, Some(3000.0));
    }

    #[test]
    fn test_state_metadata_creation() {
        let metadata = StateMetadata::new("BTC/USD".to_string());

        assert_eq!(metadata.symbol, "BTC/USD");
        assert!(metadata.regime.is_none());
        assert!(metadata.volatility.is_none());
        assert!(metadata.volume.is_none());
    }

    #[test]
    fn test_state_from_flat_gaf() {
        let gaf_features = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix flattened
        let raw_features = vec![0.5, 0.6, 0.7];
        let metadata = StateMetadata::new("BTC/USD".to_string());

        let state = State::from_flat_gaf(gaf_features.clone(), raw_features.clone(), metadata);

        assert_eq!(state.gaf_features_flat, gaf_features);
        assert_eq!(state.raw_features, raw_features);
        assert_eq!(state.feature_dim(), 7); // 4 + 3
    }

    #[test]
    fn test_state_to_feature_vector() {
        let gaf_features = vec![1.0, 2.0, 3.0, 4.0];
        let raw_features = vec![0.5, 0.6, 0.7];
        let metadata = StateMetadata::new("BTC/USD".to_string());

        let state = State::from_flat_gaf(gaf_features, raw_features, metadata);
        let feature_vec = state.to_feature_vector();

        assert_eq!(feature_vec.len(), 7);
        assert_eq!(feature_vec[0], 1.0);
        assert_eq!(feature_vec[3], 4.0);
        assert_eq!(feature_vec[4], 0.5);
        assert_eq!(feature_vec[6], 0.7);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_state_from_gaf_matrix() {
        use ndarray::arr2;

        let gaf_matrix = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let raw_features = vec![0.5, 0.6];
        let metadata = StateMetadata::new("ETH/USD".to_string());

        let state = State::from_gaf_matrix(gaf_matrix.clone(), raw_features, metadata);

        assert_eq!(state.gaf_features, gaf_matrix);
        assert_eq!(state.gaf_features_flat.len(), 4);
        assert_eq!(state.feature_dim(), 6); // 4 + 2
    }

    #[test]
    fn test_experience_creation() {
        let state = State::from_flat_gaf(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.5],
            StateMetadata::new("BTC/USD".to_string()),
        );
        let action = Action::new(ActionType::Buy, "BTC/USD".to_string(), 1.0);
        let next_state = State::from_flat_gaf(
            vec![1.1, 2.1, 3.1, 4.1],
            vec![0.6],
            StateMetadata::new("BTC/USD".to_string()),
        );

        let experience = Experience::new(state, action.clone(), 10.5, next_state, false);

        assert_eq!(experience.reward, 10.5);
        assert!(!experience.done);
        assert_eq!(experience.priority, 1.0); // Default priority
        assert_eq!(experience.action.action_type, ActionType::Buy);
    }

    #[test]
    fn test_experience_with_priority() {
        let state =
            State::from_flat_gaf(vec![1.0], vec![], StateMetadata::new("BTC/USD".to_string()));
        let action = Action::new(ActionType::Hold, "BTC/USD".to_string(), 0.0);
        let next_state = state.clone();

        let experience = Experience::new(state, action, 0.0, next_state, false).with_priority(5.0);

        assert_eq!(experience.priority, 5.0);
    }

    #[test]
    fn test_experience_with_episode_id() {
        let episode_id = uuid::Uuid::new_v4();
        let state =
            State::from_flat_gaf(vec![1.0], vec![], StateMetadata::new("BTC/USD".to_string()));
        let action = Action::new(ActionType::Hold, "BTC/USD".to_string(), 0.0);
        let next_state = state.clone();

        let experience =
            Experience::new(state, action, 0.0, next_state, true).with_episode_id(episode_id);

        assert_eq!(experience.episode_id, Some(episode_id));
        assert!(experience.done);
    }

    #[test]
    fn test_experience_serialization() {
        let state = State::from_flat_gaf(
            vec![1.0, 2.0],
            vec![0.5],
            StateMetadata::new("BTC/USD".to_string()),
        );
        let action = Action::new(ActionType::Buy, "BTC/USD".to_string(), 1.0);
        let next_state = state.clone();

        let experience = Experience::new(state, action, 10.0, next_state, false);

        // Test JSON serialization
        let json = serde_json::to_string(&experience).unwrap();
        let deserialized: Experience = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.reward, 10.0);
        assert_eq!(deserialized.action.action_type, ActionType::Buy);

        // Test postcard serialization (migrated from bincode)
        let bytes = postcard::to_allocvec(&experience).unwrap();
        let deserialized: Experience = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.reward, 10.0);
    }

    #[test]
    fn test_action_type_hash() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(ActionType::Buy, "buy_action");
        map.insert(ActionType::Sell, "sell_action");

        assert_eq!(map.get(&ActionType::Buy), Some(&"buy_action"));
        assert_eq!(map.get(&ActionType::Sell), Some(&"sell_action"));
    }
}
