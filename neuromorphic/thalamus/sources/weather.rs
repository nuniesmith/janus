//! Weather Data Sources
//!
//! Part of Thalamus region - External Data Sources
//!
//! This module handles weather data ingestion relevant for trading:
//! - Energy markets (heating/cooling demand)
//! - Agricultural commodities
//! - Natural gas demand forecasting
//! - Regional economic impacts
//!
//! Weather data can correlate with:
//! - Energy prices (extreme temperatures)
//! - Agricultural futures (droughts, floods)
//! - Transportation/logistics disruptions
//! - Insurance sector performance

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::DataSource;

/// Weather data point for a specific location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherData {
    /// Location name/identifier
    pub location: String,

    /// Latitude
    pub latitude: f64,

    /// Longitude
    pub longitude: f64,

    /// Temperature in Celsius
    pub temperature_c: f64,

    /// Feels-like temperature in Celsius
    pub feels_like_c: f64,

    /// Humidity percentage (0-100)
    pub humidity: f64,

    /// Wind speed in m/s
    pub wind_speed: f64,

    /// Wind direction in degrees
    pub wind_direction: f64,

    /// Precipitation in mm
    pub precipitation: f64,

    /// Cloud cover percentage (0-100)
    pub cloud_cover: f64,

    /// Atmospheric pressure in hPa
    pub pressure: f64,

    /// UV index
    pub uv_index: f64,

    /// Weather condition (clear, cloudy, rain, snow, etc.)
    pub condition: WeatherCondition,

    /// Timestamp of the data
    pub timestamp: DateTime<Utc>,

    /// Forecast data (if available)
    pub forecast: Option<Vec<WeatherForecast>>,

    /// Alerts/warnings
    pub alerts: Vec<WeatherAlert>,
}

impl WeatherData {
    /// Create weather data with minimal fields
    pub fn new(location: String, temperature_c: f64) -> Self {
        Self {
            location,
            latitude: 0.0,
            longitude: 0.0,
            temperature_c,
            feels_like_c: temperature_c,
            humidity: 0.0,
            wind_speed: 0.0,
            wind_direction: 0.0,
            precipitation: 0.0,
            cloud_cover: 0.0,
            pressure: 1013.25, // Standard atmospheric pressure
            uv_index: 0.0,
            condition: WeatherCondition::Clear,
            timestamp: Utc::now(),
            forecast: None,
            alerts: Vec::new(),
        }
    }

    /// Check if temperature is extreme (hot or cold)
    pub fn is_extreme_temperature(&self) -> bool {
        self.temperature_c > 35.0 || self.temperature_c < -10.0
    }

    /// Check if there's significant precipitation
    pub fn has_precipitation(&self) -> bool {
        self.precipitation > 0.5
    }

    /// Calculate heating degree days (base 18°C)
    pub fn heating_degree_days(&self) -> f64 {
        (18.0 - self.temperature_c).max(0.0)
    }

    /// Calculate cooling degree days (base 18°C)
    pub fn cooling_degree_days(&self) -> f64 {
        (self.temperature_c - 18.0).max(0.0)
    }

    /// Get energy demand indicator (-1.0 to 1.0)
    /// Positive = high energy demand (extreme cold/hot)
    /// Negative = low energy demand (mild weather)
    pub fn energy_demand_indicator(&self) -> f64 {
        let deviation = (self.temperature_c - 20.0).abs();
        let indicator = (deviation / 20.0).min(1.0);

        // Extreme temperatures increase energy demand
        if self.temperature_c < 0.0 || self.temperature_c > 35.0 {
            indicator
        } else if self.temperature_c > 18.0 && self.temperature_c < 25.0 {
            -indicator // Mild weather = lower demand
        } else {
            indicator * 0.5
        }
    }
}

impl Default for WeatherData {
    fn default() -> Self {
        Self::new("Unknown".to_string(), 20.0)
    }
}

/// Weather conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeatherCondition {
    Clear,
    PartlyCloudy,
    Cloudy,
    Overcast,
    Fog,
    Mist,
    Rain,
    HeavyRain,
    Drizzle,
    Thunderstorm,
    Snow,
    HeavySnow,
    Sleet,
    Hail,
    Dust,
    Smoke,
    Haze,
    Tornado,
    Hurricane,
    Unknown,
}

impl WeatherCondition {
    /// Check if condition is severe
    pub fn is_severe(&self) -> bool {
        matches!(
            self,
            WeatherCondition::Thunderstorm
                | WeatherCondition::HeavyRain
                | WeatherCondition::HeavySnow
                | WeatherCondition::Hail
                | WeatherCondition::Tornado
                | WeatherCondition::Hurricane
        )
    }

    /// Check if condition affects transportation
    pub fn affects_transportation(&self) -> bool {
        matches!(
            self,
            WeatherCondition::HeavyRain
                | WeatherCondition::Thunderstorm
                | WeatherCondition::Snow
                | WeatherCondition::HeavySnow
                | WeatherCondition::Fog
                | WeatherCondition::Hail
                | WeatherCondition::Tornado
                | WeatherCondition::Hurricane
        )
    }
}

impl Default for WeatherCondition {
    fn default() -> Self {
        WeatherCondition::Unknown
    }
}

/// Weather forecast for a future time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherForecast {
    /// Forecast timestamp
    pub timestamp: DateTime<Utc>,

    /// Forecasted temperature in Celsius
    pub temperature_c: f64,

    /// Forecasted condition
    pub condition: WeatherCondition,

    /// Probability of precipitation (0-100)
    pub precipitation_probability: f64,

    /// Forecasted precipitation amount in mm
    pub precipitation_amount: f64,

    /// Forecasted wind speed in m/s
    pub wind_speed: f64,

    /// Confidence level of forecast (0-1)
    pub confidence: f64,
}

/// Weather alert/warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherAlert {
    /// Alert type
    pub alert_type: AlertType,

    /// Severity level
    pub severity: AlertSeverity,

    /// Alert title
    pub title: String,

    /// Alert description
    pub description: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// End time
    pub end_time: Option<DateTime<Utc>>,

    /// Affected regions
    pub regions: Vec<String>,
}

/// Types of weather alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    HeatWave,
    ColdWave,
    Flood,
    Drought,
    Storm,
    Hurricane,
    Tornado,
    Wildfire,
    AirQuality,
    Wind,
    Snow,
    Ice,
    Other,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Minor,
    Moderate,
    Severe,
    Extreme,
}

/// Weather source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherSourceConfig {
    /// API key for the weather service
    pub api_key: Option<String>,

    /// Base URL for the API
    pub base_url: String,

    /// Locations to track
    pub locations: Vec<LocationConfig>,

    /// Update interval in seconds
    pub update_interval: u64,

    /// Enable forecast fetching
    pub enable_forecast: bool,

    /// Forecast days to fetch
    pub forecast_days: u32,

    /// Enable alerts
    pub enable_alerts: bool,
}

impl Default for WeatherSourceConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            locations: vec![
                LocationConfig::new("New York", 40.7128, -74.0060),
                LocationConfig::new("London", 51.5074, -0.1278),
                LocationConfig::new("Tokyo", 35.6762, 139.6503),
                LocationConfig::new("Singapore", 1.3521, 103.8198),
            ],
            update_interval: 3600, // 1 hour
            enable_forecast: true,
            forecast_days: 7,
            enable_alerts: true,
        }
    }
}

/// Location configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationConfig {
    /// Location name
    pub name: String,

    /// Latitude
    pub latitude: f64,

    /// Longitude
    pub longitude: f64,

    /// Market relevance (which markets this location affects)
    pub market_relevance: Vec<String>,
}

impl LocationConfig {
    /// Create a new location config
    pub fn new(name: &str, latitude: f64, longitude: f64) -> Self {
        Self {
            name: name.to_string(),
            latitude,
            longitude,
            market_relevance: Vec::new(),
        }
    }

    /// Add market relevance
    pub fn with_markets(mut self, markets: Vec<String>) -> Self {
        self.market_relevance = markets;
        self
    }
}

/// Weather data source implementation
pub struct WeatherSource {
    /// Source name
    name: String,

    /// Configuration
    config: WeatherSourceConfig,

    /// Last update timestamp
    last_update: Option<DateTime<Utc>>,

    /// Cached weather data by location
    cached_data: HashMap<String, WeatherData>,
}

impl WeatherSource {
    /// Create a new weather source
    pub fn new(name: String, config: WeatherSourceConfig) -> Self {
        Self {
            name,
            config,
            last_update: None,
            cached_data: HashMap::new(),
        }
    }

    /// Get cached weather for a location
    pub fn get_location_weather(&self, location: &str) -> Option<&WeatherData> {
        self.cached_data.get(location)
    }

    /// Get all cached weather data
    pub fn all_weather_data(&self) -> &HashMap<String, WeatherData> {
        &self.cached_data
    }

    /// Get configuration
    pub fn config(&self) -> &WeatherSourceConfig {
        &self.config
    }

    /// Calculate aggregate energy demand from all locations
    pub fn aggregate_energy_demand(&self) -> f64 {
        if self.cached_data.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .cached_data
            .values()
            .map(|w| w.energy_demand_indicator())
            .sum();

        total / self.cached_data.len() as f64
    }

    /// Get all active alerts across locations
    pub fn all_alerts(&self) -> Vec<&WeatherAlert> {
        self.cached_data.values().flat_map(|w| &w.alerts).collect()
    }

    /// Check if any location has severe weather
    pub fn has_severe_weather(&self) -> bool {
        self.cached_data.values().any(|w| {
            w.condition.is_severe() || w.alerts.iter().any(|a| a.severity >= AlertSeverity::Severe)
        })
    }

    /// Fetch weather data (placeholder - implement for each provider)
    async fn fetch_weather(&self) -> crate::common::Result<HashMap<String, WeatherData>> {
        // This would be implemented differently for each weather provider
        // (OpenWeatherMap, WeatherAPI, AccuWeather, etc.)
        // For now, return empty to allow compilation
        Ok(HashMap::new())
    }
}

#[async_trait]
impl DataSource for WeatherSource {
    type Data = WeatherData;

    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_latest(&self) -> crate::common::Result<Self::Data> {
        let data = self.fetch_weather().await?;

        // Return the first location's data or a default
        if let Some((_, weather)) = data.into_iter().next() {
            Ok(weather)
        } else {
            Ok(WeatherData::default())
        }
    }

    async fn health_check(&self) -> bool {
        // Basic health check - verify API connectivity
        self.config.api_key.is_some() || !self.config.base_url.is_empty()
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        self.last_update
    }
}

/// Regional weather aggregator for market-specific analysis
pub struct RegionalWeatherAggregator {
    /// Sources by region
    sources: HashMap<String, WeatherSource>,

    /// Last update (reserved for caching)
    #[allow(dead_code)]
    last_update: Option<DateTime<Utc>>,
}

impl RegionalWeatherAggregator {
    /// Create a new regional aggregator
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            last_update: None,
        }
    }

    /// Add a regional source
    pub fn add_region(&mut self, region: String, source: WeatherSource) {
        self.sources.insert(region, source);
    }

    /// Get weather for a specific region
    pub fn get_region(&self, region: &str) -> Option<&WeatherSource> {
        self.sources.get(region)
    }

    /// Check if any region has severe weather
    pub fn any_severe_weather(&self) -> bool {
        self.sources.values().any(|s| s.has_severe_weather())
    }
}

impl Default for RegionalWeatherAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weather_data_creation() {
        let weather = WeatherData::new("New York".to_string(), 25.0);
        assert_eq!(weather.location, "New York");
        assert_eq!(weather.temperature_c, 25.0);
    }

    #[test]
    fn test_extreme_temperature_detection() {
        let hot = WeatherData::new("Desert".to_string(), 40.0);
        assert!(hot.is_extreme_temperature());

        let cold = WeatherData::new("Arctic".to_string(), -20.0);
        assert!(cold.is_extreme_temperature());

        let mild = WeatherData::new("Mild".to_string(), 20.0);
        assert!(!mild.is_extreme_temperature());
    }

    #[test]
    fn test_heating_cooling_degree_days() {
        let cold = WeatherData::new("Cold".to_string(), 10.0);
        assert_eq!(cold.heating_degree_days(), 8.0);
        assert_eq!(cold.cooling_degree_days(), 0.0);

        let hot = WeatherData::new("Hot".to_string(), 30.0);
        assert_eq!(hot.heating_degree_days(), 0.0);
        assert_eq!(hot.cooling_degree_days(), 12.0);
    }

    #[test]
    fn test_weather_condition_severity() {
        assert!(WeatherCondition::Hurricane.is_severe());
        assert!(WeatherCondition::Tornado.is_severe());
        assert!(!WeatherCondition::Clear.is_severe());
        assert!(!WeatherCondition::Cloudy.is_severe());
    }

    #[test]
    fn test_transportation_impact() {
        assert!(WeatherCondition::HeavySnow.affects_transportation());
        assert!(WeatherCondition::Fog.affects_transportation());
        assert!(!WeatherCondition::Clear.affects_transportation());
    }

    #[test]
    fn test_weather_source_config_default() {
        let config = WeatherSourceConfig::default();
        assert!(!config.locations.is_empty());
        assert!(config.enable_forecast);
    }

    #[test]
    fn test_location_config() {
        let location = LocationConfig::new("Chicago", 41.8781, -87.6298)
            .with_markets(vec!["CME".to_string(), "CBOT".to_string()]);

        assert_eq!(location.name, "Chicago");
        assert_eq!(location.market_relevance.len(), 2);
    }

    #[test]
    fn test_regional_aggregator() {
        let aggregator = RegionalWeatherAggregator::new();
        assert!(aggregator.sources.is_empty());
        assert!(!aggregator.any_severe_weather());
    }
}
