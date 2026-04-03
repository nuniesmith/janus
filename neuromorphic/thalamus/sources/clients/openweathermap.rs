//! OpenWeatherMap API Client
//!
//! Provides access to weather data from OpenWeatherMap.
//! API Documentation: https://openweathermap.org/api
//!
//! ## Features
//!
//! - Current weather conditions
//! - Weather forecasts (hourly, daily)
//! - Historical weather data
//! - Air pollution data
//! - Weather alerts
//!
//! ## Rate Limits
//!
//! - Free tier: 60 calls/minute, 1,000,000 calls/month
//! - Various paid tiers available

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use std::collections::HashMap;
use tracing::{debug, error, warn};

use super::{ApiClient, ApiClientConfig, RateLimiter};
use crate::common::{Error, Result};
use crate::thalamus::sources::DataSource;
use crate::thalamus::sources::weather::{
    LocationConfig, WeatherCondition, WeatherData, WeatherForecast,
};

/// OpenWeatherMap API base URL
const OWM_BASE_URL: &str = "https://api.openweathermap.org/data/2.5";

/// OpenWeatherMap One Call API base URL (for comprehensive data)
#[allow(dead_code)]
const OWM_ONECALL_URL: &str = "https://api.openweathermap.org/data/3.0/onecall";

/// Units for temperature and other measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Units {
    /// Kelvin (default)
    Standard,
    /// Celsius
    Metric,
    /// Fahrenheit
    Imperial,
}

impl Units {
    fn as_str(&self) -> &str {
        match self {
            Units::Standard => "standard",
            Units::Metric => "metric",
            Units::Imperial => "imperial",
        }
    }
}

impl Default for Units {
    fn default() -> Self {
        Units::Metric
    }
}

/// OpenWeatherMap API client
pub struct OpenWeatherMapClient {
    /// HTTP client
    client: reqwest::Client,
    /// API configuration
    config: ApiClientConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Temperature units
    units: Units,
    /// Cached weather data by location
    cached_data: tokio::sync::RwLock<HashMap<String, WeatherData>>,
    /// Last update timestamp
    last_update: tokio::sync::RwLock<Option<DateTime<Utc>>>,
    /// Tracked locations
    locations: Vec<LocationConfig>,
}

impl OpenWeatherMapClient {
    /// Create a new OpenWeatherMap client
    pub fn new(api_key: String) -> Result<Self> {
        let config = ApiClientConfig {
            api_key: Some(api_key),
            base_url: OWM_BASE_URL.to_string(),
            ..Default::default()
        };

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(1.0), // 1 request per second (60/minute)
            units: Units::Metric,
            cached_data: tokio::sync::RwLock::new(HashMap::new()),
            last_update: tokio::sync::RwLock::new(None),
            locations: Self::default_locations(),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ApiClientConfig, units: Units) -> Result<Self> {
        if config.api_key.is_none() {
            return Err(Error::Other(
                "OpenWeatherMap requires an API key".to_string(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(1.0),
            units,
            cached_data: tokio::sync::RwLock::new(HashMap::new()),
            last_update: tokio::sync::RwLock::new(None),
            locations: Self::default_locations(),
        })
    }

    /// Set tracked locations
    pub fn set_locations(&mut self, locations: Vec<LocationConfig>) {
        self.locations = locations;
    }

    /// Get default locations for financial markets
    fn default_locations() -> Vec<LocationConfig> {
        vec![
            LocationConfig::new("New York", 40.7128, -74.0060),
            LocationConfig::new("London", 51.5074, -0.1278),
            LocationConfig::new("Tokyo", 35.6762, 139.6503),
            LocationConfig::new("Singapore", 1.3521, 103.8198),
            LocationConfig::new("Hong Kong", 22.3193, 114.1694),
            LocationConfig::new("Frankfurt", 50.1109, 8.6821),
            LocationConfig::new("Chicago", 41.8781, -87.6298),
            LocationConfig::new("Sydney", -33.8688, 151.2093),
        ]
    }

    /// Fetch current weather by coordinates
    pub async fn fetch_weather_by_coords(&self, lat: f64, lon: f64) -> Result<WeatherData> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let url = format!(
            "{}/weather?lat={}&lon={}&units={}&appid={}",
            self.config.base_url,
            lat,
            lon,
            self.units.as_str(),
            api_key
        );

        debug!("Fetching OpenWeatherMap data for ({}, {})", lat, lon);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("OpenWeatherMap API error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: OwmCurrentWeatherResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        let weather_data = self.convert_current_weather(api_response);

        debug!("Fetched weather for {}", weather_data.location);
        Ok(weather_data)
    }

    /// Fetch current weather by city name
    pub async fn fetch_weather_by_city(&self, city: &str) -> Result<WeatherData> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let url = format!(
            "{}/weather?q={}&units={}&appid={}",
            self.config.base_url,
            urlencoding::encode(city),
            self.units.as_str(),
            api_key
        );

        debug!("Fetching OpenWeatherMap data for city: {}", city);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("OpenWeatherMap API error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: OwmCurrentWeatherResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        let weather_data = self.convert_current_weather(api_response);

        Ok(weather_data)
    }

    /// Fetch weather forecast (5 day / 3 hour)
    pub async fn fetch_forecast(&self, lat: f64, lon: f64) -> Result<Vec<WeatherForecast>> {
        self.rate_limiter.wait().await;

        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| Error::Other("API key required".to_string()))?;

        let url = format!(
            "{}/forecast?lat={}&lon={}&units={}&appid={}",
            self.config.base_url,
            lat,
            lon,
            self.units.as_str(),
            api_key
        );

        debug!("Fetching OpenWeatherMap forecast for ({}, {})", lat, lon);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("OpenWeatherMap API error: {} - {}", status, text);
            return Err(Error::Other(format!("API error: {} - {}", status, text)));
        }

        let api_response: OwmForecastResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        let forecasts: Vec<WeatherForecast> = api_response
            .list
            .into_iter()
            .map(|item| self.convert_forecast_item(item))
            .collect();

        debug!("Fetched {} forecast items", forecasts.len());
        Ok(forecasts)
    }

    /// Fetch weather for all tracked locations
    pub async fn fetch_all_locations(&self) -> Result<HashMap<String, WeatherData>> {
        let mut results = HashMap::new();

        for location in &self.locations {
            match self
                .fetch_weather_by_coords(location.latitude, location.longitude)
                .await
            {
                Ok(mut weather) => {
                    weather.location = location.name.clone();
                    results.insert(location.name.clone(), weather);
                }
                Err(e) => {
                    warn!("Failed to fetch weather for {}: {}", location.name, e);
                }
            }
        }

        // Update cache
        {
            let mut cache = self.cached_data.write().await;
            *cache = results.clone();
            let mut last = self.last_update.write().await;
            *last = Some(Utc::now());
        }

        Ok(results)
    }

    /// Convert OWM current weather response to our format
    fn convert_current_weather(&self, response: OwmCurrentWeatherResponse) -> WeatherData {
        let condition = response
            .weather
            .first()
            .map(|w| self.parse_condition(&w.main, w.id))
            .unwrap_or(WeatherCondition::Unknown);

        let timestamp = Utc
            .timestamp_opt(response.dt, 0)
            .single()
            .unwrap_or_else(Utc::now);

        WeatherData {
            location: response.name.unwrap_or_else(|| "Unknown".to_string()),
            latitude: response.coord.lat,
            longitude: response.coord.lon,
            temperature_c: self.to_celsius(response.main.temp),
            feels_like_c: self.to_celsius(response.main.feels_like),
            humidity: response.main.humidity,
            wind_speed: response.wind.speed,
            wind_direction: response.wind.deg.unwrap_or(0.0),
            precipitation: response
                .rain
                .as_ref()
                .map(|r| r.one_hour.unwrap_or(0.0))
                .unwrap_or(0.0),
            cloud_cover: response.clouds.all,
            pressure: response.main.pressure,
            uv_index: 0.0, // Not available in current weather endpoint
            condition,
            timestamp,
            forecast: None,
            alerts: Vec::new(),
        }
    }

    /// Convert OWM forecast item to our format
    fn convert_forecast_item(&self, item: OwmForecastItem) -> WeatherForecast {
        let condition = item
            .weather
            .first()
            .map(|w| self.parse_condition(&w.main, w.id))
            .unwrap_or(WeatherCondition::Unknown);

        let timestamp = Utc
            .timestamp_opt(item.dt, 0)
            .single()
            .unwrap_or_else(Utc::now);

        WeatherForecast {
            timestamp,
            temperature_c: self.to_celsius(item.main.temp),
            condition,
            precipitation_probability: item.pop * 100.0,
            precipitation_amount: item
                .rain
                .as_ref()
                .map(|r| r.three_hour.unwrap_or(0.0))
                .unwrap_or(0.0),
            wind_speed: item.wind.speed,
            confidence: 0.8, // OWM doesn't provide confidence, estimate
        }
    }

    /// Parse weather condition from OWM codes
    fn parse_condition(&self, main: &str, id: i32) -> WeatherCondition {
        match main.to_lowercase().as_str() {
            "clear" => WeatherCondition::Clear,
            "clouds" => match id {
                801 => WeatherCondition::PartlyCloudy,
                802 => WeatherCondition::Cloudy,
                803 | 804 => WeatherCondition::Overcast,
                _ => WeatherCondition::Cloudy,
            },
            "rain" => match id {
                500 | 501 => WeatherCondition::Rain,
                502..=504 => WeatherCondition::HeavyRain,
                300..=321 => WeatherCondition::Drizzle,
                _ => WeatherCondition::Rain,
            },
            "drizzle" => WeatherCondition::Drizzle,
            "thunderstorm" => WeatherCondition::Thunderstorm,
            "snow" => match id {
                600 | 601 => WeatherCondition::Snow,
                602 => WeatherCondition::HeavySnow,
                611..=616 => WeatherCondition::Sleet,
                _ => WeatherCondition::Snow,
            },
            "mist" | "fog" => WeatherCondition::Fog,
            "haze" => WeatherCondition::Haze,
            "dust" | "sand" => WeatherCondition::Dust,
            "smoke" => WeatherCondition::Smoke,
            "tornado" => WeatherCondition::Tornado,
            _ => WeatherCondition::Unknown,
        }
    }

    /// Convert temperature to Celsius if needed
    fn to_celsius(&self, temp: f64) -> f64 {
        match self.units {
            Units::Standard => temp - 273.15,             // Kelvin to Celsius
            Units::Metric => temp,                        // Already Celsius
            Units::Imperial => (temp - 32.0) * 5.0 / 9.0, // Fahrenheit to Celsius
        }
    }

    /// Calculate aggregate energy demand indicator
    pub async fn aggregate_energy_demand(&self) -> f64 {
        let cache = self.cached_data.read().await;

        if cache.is_empty() {
            return 0.0;
        }

        let total: f64 = cache.values().map(|w| w.energy_demand_indicator()).sum();
        total / cache.len() as f64
    }

    /// Check if any location has extreme weather
    pub async fn has_extreme_weather(&self) -> bool {
        let cache = self.cached_data.read().await;
        cache
            .values()
            .any(|w| w.is_extreme_temperature() || w.condition.is_severe())
    }

    /// Get cached weather for a location
    pub async fn get_cached(&self, location: &str) -> Option<WeatherData> {
        let cache = self.cached_data.read().await;
        cache.get(location).cloned()
    }

    /// Get all cached weather data
    pub async fn get_all_cached(&self) -> HashMap<String, WeatherData> {
        self.cached_data.read().await.clone()
    }

    /// Get last update timestamp
    pub async fn last_update_time(&self) -> Option<DateTime<Utc>> {
        *self.last_update.read().await
    }
}

#[async_trait]
impl ApiClient for OpenWeatherMapClient {
    fn name(&self) -> &str {
        "openweathermap"
    }

    fn is_configured(&self) -> bool {
        self.config.api_key.is_some()
    }

    async fn health_check(&self) -> bool {
        if !self.is_configured() {
            return false;
        }

        // Try fetching weather for a known location
        match self.fetch_weather_by_city("London").await {
            Ok(_) => true,
            Err(e) => {
                warn!("OpenWeatherMap health check failed: {}", e);
                false
            }
        }
    }
}

#[async_trait]
impl DataSource for OpenWeatherMapClient {
    type Data = WeatherData;

    fn name(&self) -> &str {
        "openweathermap"
    }

    async fn fetch_latest(&self) -> Result<Self::Data> {
        // Fetch weather for the first location as primary
        if let Some(location) = self.locations.first() {
            self.fetch_weather_by_coords(location.latitude, location.longitude)
                .await
        } else {
            // Default to New York
            self.fetch_weather_by_coords(40.7128, -74.0060).await
        }
    }

    async fn health_check(&self) -> bool {
        <Self as ApiClient>::health_check(self).await
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        None
    }
}

// ============================================================================
// API Response Types
// ============================================================================

/// Current weather response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmCurrentWeatherResponse {
    /// Coordinates
    coord: OwmCoord,
    /// Weather conditions
    weather: Vec<OwmWeatherCondition>,
    /// Main weather data
    main: OwmMainData,
    /// Visibility in meters
    visibility: Option<i32>,
    /// Wind data
    wind: OwmWind,
    /// Cloud data
    clouds: OwmClouds,
    /// Rain data
    rain: Option<OwmPrecipitation>,
    /// Snow data
    snow: Option<OwmPrecipitation>,
    /// Timestamp (Unix)
    dt: i64,
    /// System data
    sys: Option<OwmSys>,
    /// Timezone offset
    timezone: Option<i32>,
    /// City ID
    id: Option<i64>,
    /// City name
    name: Option<String>,
    /// Response code
    cod: Option<serde_json::Value>,
}

/// Forecast response
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmForecastResponse {
    /// Response code
    cod: String,
    /// Message
    message: Option<i32>,
    /// Count of items
    cnt: i32,
    /// Forecast list
    list: Vec<OwmForecastItem>,
    /// City info
    city: Option<OwmCity>,
}

/// Forecast item
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmForecastItem {
    /// Timestamp
    dt: i64,
    /// Main weather data
    main: OwmMainData,
    /// Weather conditions
    weather: Vec<OwmWeatherCondition>,
    /// Clouds
    clouds: OwmClouds,
    /// Wind
    wind: OwmWind,
    /// Visibility
    visibility: Option<i32>,
    /// Probability of precipitation (0-1)
    pop: f64,
    /// Rain data
    rain: Option<OwmPrecipitation3h>,
    /// Snow data
    snow: Option<OwmPrecipitation3h>,
    /// System data
    sys: Option<OwmForecastSys>,
    /// DateTime text
    dt_txt: Option<String>,
}

/// Coordinates
#[derive(Debug, Deserialize)]
struct OwmCoord {
    lon: f64,
    lat: f64,
}

/// Weather condition
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmWeatherCondition {
    /// Weather condition ID
    id: i32,
    /// Group (Rain, Snow, etc.)
    main: String,
    /// Description
    description: String,
    /// Icon code
    icon: String,
}

/// Main weather data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmMainData {
    /// Temperature
    temp: f64,
    /// Feels like temperature
    feels_like: f64,
    /// Minimum temperature
    temp_min: f64,
    /// Maximum temperature
    temp_max: f64,
    /// Atmospheric pressure (hPa)
    pressure: f64,
    /// Humidity (%)
    humidity: f64,
    /// Sea level pressure
    sea_level: Option<f64>,
    /// Ground level pressure
    grnd_level: Option<f64>,
}

/// Wind data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmWind {
    /// Wind speed
    speed: f64,
    /// Wind direction (degrees)
    deg: Option<f64>,
    /// Wind gust
    gust: Option<f64>,
}

/// Cloud data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmClouds {
    /// Cloudiness (%)
    all: f64,
}

/// Precipitation data (current weather)
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmPrecipitation {
    /// Rain/snow volume for last 1 hour (mm)
    #[serde(rename = "1h")]
    one_hour: Option<f64>,
    /// Rain/snow volume for last 3 hours (mm)
    #[serde(rename = "3h")]
    three_hour: Option<f64>,
}

/// Precipitation data for 3-hour forecast
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmPrecipitation3h {
    /// Rain/snow volume for 3 hours (mm)
    #[serde(rename = "3h")]
    three_hour: Option<f64>,
}

/// System data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmSys {
    /// Internal parameter
    #[serde(rename = "type")]
    type_: Option<i32>,
    /// Internal parameter
    id: Option<i64>,
    /// Country code
    country: Option<String>,
    /// Sunrise time (Unix)
    sunrise: Option<i64>,
    /// Sunset time (Unix)
    sunset: Option<i64>,
}

/// Forecast system data
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmForecastSys {
    /// Part of day (d/n)
    pod: Option<String>,
}

/// City information
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OwmCity {
    /// City ID
    id: i64,
    /// City name
    name: String,
    /// Coordinates
    coord: OwmCoord,
    /// Country code
    country: String,
    /// Population
    population: Option<i64>,
    /// Timezone offset
    timezone: i32,
    /// Sunrise time
    sunrise: Option<i64>,
    /// Sunset time
    sunset: Option<i64>,
}

/// URL encoding helper
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_requires_api_key() {
        let config = ApiClientConfig {
            api_key: None,
            base_url: OWM_BASE_URL.to_string(),
            ..Default::default()
        };

        let result = OpenWeatherMapClient::with_config(config, Units::Metric);
        assert!(result.is_err());
    }

    #[test]
    fn test_client_creation_with_key() {
        let client = OpenWeatherMapClient::new("test_api_key".to_string());
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(client.is_configured());
    }

    #[test]
    fn test_units() {
        assert_eq!(Units::Metric.as_str(), "metric");
        assert_eq!(Units::Imperial.as_str(), "imperial");
        assert_eq!(Units::Standard.as_str(), "standard");
    }

    #[test]
    fn test_temperature_conversion() {
        let _client = OpenWeatherMapClient::new("test_key".to_string()).unwrap();

        // Test Kelvin to Celsius (Standard)
        let kelvin_client = OpenWeatherMapClient::with_config(
            ApiClientConfig {
                api_key: Some("test".to_string()),
                ..Default::default()
            },
            Units::Standard,
        )
        .unwrap();
        let celsius = kelvin_client.to_celsius(273.15);
        assert!((celsius - 0.0).abs() < 0.01);

        // Test Fahrenheit to Celsius
        let imperial_client = OpenWeatherMapClient::with_config(
            ApiClientConfig {
                api_key: Some("test".to_string()),
                ..Default::default()
            },
            Units::Imperial,
        )
        .unwrap();
        let celsius = imperial_client.to_celsius(32.0);
        assert!((celsius - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_condition_parsing() {
        let client = OpenWeatherMapClient::new("test_key".to_string()).unwrap();

        assert_eq!(
            client.parse_condition("Clear", 800),
            WeatherCondition::Clear
        );
        assert_eq!(client.parse_condition("Rain", 500), WeatherCondition::Rain);
        assert_eq!(
            client.parse_condition("Rain", 502),
            WeatherCondition::HeavyRain
        );
        assert_eq!(
            client.parse_condition("Thunderstorm", 200),
            WeatherCondition::Thunderstorm
        );
        assert_eq!(client.parse_condition("Snow", 600), WeatherCondition::Snow);
        assert_eq!(
            client.parse_condition("Clouds", 801),
            WeatherCondition::PartlyCloudy
        );
    }

    #[test]
    fn test_default_locations() {
        let locations = OpenWeatherMapClient::default_locations();
        assert!(!locations.is_empty());

        // Should include major financial centers
        let names: Vec<&str> = locations.iter().map(|l| l.name.as_str()).collect();
        assert!(names.contains(&"New York"));
        assert!(names.contains(&"London"));
        assert!(names.contains(&"Tokyo"));
    }

    #[tokio::test]
    async fn test_energy_demand_empty_cache() {
        let client = OpenWeatherMapClient::new("test_key".to_string()).unwrap();
        let demand = client.aggregate_energy_demand().await;
        assert_eq!(demand, 0.0);
    }

    #[tokio::test]
    async fn test_has_extreme_weather_empty_cache() {
        let client = OpenWeatherMapClient::new("test_key".to_string()).unwrap();
        let has_extreme = client.has_extreme_weather().await;
        assert!(!has_extreme);
    }
}
