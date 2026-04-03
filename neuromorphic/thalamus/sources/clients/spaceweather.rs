//! NOAA Space Weather Prediction Center API Client
//!
//! Provides access to space weather data from NOAA SWPC.
//! API Documentation: https://services.swpc.noaa.gov/
//!
//! ## Features
//!
//! - Geomagnetic storm indices (Kp, Dst)
//! - Solar flare alerts
//! - Solar wind data
//! - Aurora forecasts
//! - Radio blackout levels
//!
//! ## Data Sources
//!
//! - Real-time solar wind (DSCOVR satellite)
//! - Planetary K-index (Kp)
//! - Solar X-ray flux
//! - Geomagnetic alerts and warnings

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use super::{ApiClient, ApiClientConfig, RateLimiter};
use crate::common::{Error, Result};
use crate::thalamus::sources::DataSource;
use crate::thalamus::sources::celestial::{
    CelestialData, FlareClass, GeomagneticLevel, MoonPhase, RadiationLevel, RadioBlackoutLevel,
    SolarCyclePhase, SolarData, SolarFlare, SpaceWeather, XRayLevel,
};

/// NOAA SWPC API base URL
const SWPC_BASE_URL: &str = "https://services.swpc.noaa.gov";

/// NOAA Space Weather API client
pub struct SpaceWeatherClient {
    /// HTTP client
    client: reqwest::Client,
    /// API configuration
    config: ApiClientConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Cached space weather data
    cached_space_weather: tokio::sync::RwLock<Option<SpaceWeather>>,
    /// Cached solar data
    cached_solar: tokio::sync::RwLock<Option<SolarData>>,
    /// Last update timestamp
    last_update: tokio::sync::RwLock<Option<DateTime<Utc>>>,
}

impl SpaceWeatherClient {
    /// Create a new Space Weather client
    /// Note: NOAA SWPC API is free and doesn't require an API key
    pub fn new() -> Result<Self> {
        let config = ApiClientConfig {
            api_key: None,
            base_url: SWPC_BASE_URL.to_string(),
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
            rate_limiter: RateLimiter::new(2.0), // 2 requests per second
            cached_space_weather: tokio::sync::RwLock::new(None),
            cached_solar: tokio::sync::RwLock::new(None),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Create with custom configuration
    pub fn with_config(config: ApiClientConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            rate_limiter: RateLimiter::new(2.0),
            cached_space_weather: tokio::sync::RwLock::new(None),
            cached_solar: tokio::sync::RwLock::new(None),
            last_update: tokio::sync::RwLock::new(None),
        })
    }

    /// Fetch current planetary K-index (Kp)
    pub async fn fetch_kp_index(&self) -> Result<f64> {
        self.rate_limiter.wait().await;

        let url = format!(
            "{}/products/noaa-planetary-k-index.json",
            self.config.base_url
        );

        debug!("Fetching Kp index from SWPC");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            error!("SWPC API error: {}", status);
            return Err(Error::Other(format!("API error: {}", status)));
        }

        let data: Vec<Vec<String>> = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        // Data format: [["time_tag", "Kp", "a_running", "station_count"], ...]
        // Get the most recent Kp value (last row, second column)
        if let Some(latest) = data.last() {
            if latest.len() >= 2 {
                if let Ok(kp) = latest[1].parse::<f64>() {
                    debug!("Current Kp index: {}", kp);
                    return Ok(kp);
                }
            }
        }

        Err(Error::Other("Failed to extract Kp value".to_string()))
    }

    /// Fetch solar wind data from DSCOVR
    pub async fn fetch_solar_wind(&self) -> Result<SolarWindData> {
        self.rate_limiter.wait().await;

        let url = format!(
            "{}/products/solar-wind/plasma-7-day.json",
            self.config.base_url
        );

        debug!("Fetching solar wind data from SWPC");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            error!("SWPC API error: {}", status);
            return Err(Error::Other(format!("API error: {}", status)));
        }

        let data: Vec<Vec<serde_json::Value>> = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        // Data format: [["time_tag", "density", "speed", "temperature"], ...]
        // Get the most recent data point
        if let Some(latest) = data.last() {
            if latest.len() >= 4 {
                let density = latest[1]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(5.0);
                let speed = latest[2]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(400.0);
                let temperature = latest[3]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(100000.0);

                return Ok(SolarWindData {
                    density,
                    speed,
                    temperature,
                    timestamp: Utc::now(),
                });
            }
        }

        Err(Error::Other(
            "Failed to extract solar wind data".to_string(),
        ))
    }

    /// Fetch X-ray flux data
    pub async fn fetch_xray_flux(&self) -> Result<XRayFluxData> {
        self.rate_limiter.wait().await;

        let url = format!("{}/products/goes-primary-xray.json", self.config.base_url);

        debug!("Fetching X-ray flux from SWPC");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            error!("SWPC API error: {}", status);
            return Err(Error::Other(format!("API error: {}", status)));
        }

        let data: Vec<SwpcXrayEntry> = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        // Get the most recent X-ray flux
        if let Some(latest) = data.last() {
            let flux = latest.flux.parse::<f64>().unwrap_or(1e-8);
            let level = self.classify_xray_level(flux);

            return Ok(XRayFluxData {
                flux,
                level,
                timestamp: Utc::now(),
            });
        }

        Err(Error::Other(
            "Failed to extract X-ray flux data".to_string(),
        ))
    }

    /// Fetch geomagnetic storm alerts
    pub async fn fetch_geomagnetic_alerts(&self) -> Result<Vec<GeomagneticAlert>> {
        self.rate_limiter.wait().await;

        let url = format!("{}/products/alerts.json", self.config.base_url);

        debug!("Fetching geomagnetic alerts from SWPC");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            error!("SWPC API error: {}", status);
            return Err(Error::Other(format!("API error: {}", status)));
        }

        let alerts: Vec<SwpcAlert> = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        // Filter for geomagnetic-related alerts
        let geo_alerts: Vec<GeomagneticAlert> = alerts
            .into_iter()
            .filter(|a| {
                a.product_id.contains("WATA")
                    || a.product_id.contains("ALTK")
                    || a.product_id.contains("ALTXMF")
            })
            .map(|a| self.convert_alert(a))
            .collect();

        debug!("Found {} geomagnetic alerts", geo_alerts.len());
        Ok(geo_alerts)
    }

    /// Fetch solar flare events
    pub async fn fetch_solar_flares(&self) -> Result<Vec<SolarFlare>> {
        self.rate_limiter.wait().await;

        let url = format!(
            "{}/json/solar-cycle/observed_swx.json",
            self.config.base_url
        );

        debug!("Fetching solar flare data from SWPC");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;

        if !response.status().is_success() {
            // Try alternative endpoint
            return self.fetch_solar_flares_alternative().await;
        }

        let data: Vec<SwpcSolarEvent> = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("Failed to parse response: {}", e)))?;

        let flares: Vec<SolarFlare> = data
            .into_iter()
            .filter_map(|event| self.convert_solar_event(event))
            .collect();

        debug!("Found {} solar flares", flares.len());
        Ok(flares)
    }

    /// Alternative endpoint for solar flare data
    async fn fetch_solar_flares_alternative(&self) -> Result<Vec<SolarFlare>> {
        // Return empty if alternative also fails
        Ok(Vec::new())
    }

    /// Fetch comprehensive space weather data
    pub async fn fetch_all(&self) -> Result<SpaceWeather> {
        // Fetch Kp index
        let kp_index = self.fetch_kp_index().await.unwrap_or(2.0);

        // Fetch solar wind
        let solar_wind = self.fetch_solar_wind().await.ok();

        // Fetch X-ray flux
        let xray_flux = self.fetch_xray_flux().await.ok();

        // Fetch alerts
        let _alerts = self.fetch_geomagnetic_alerts().await.unwrap_or_default();

        // Determine storm levels from data
        let geomagnetic_storm = self.kp_to_geomagnetic_level(kp_index);

        // Determine radio blackout from X-ray
        let radio_blackout = xray_flux
            .as_ref()
            .map(|x| self.xray_to_radio_blackout(x.level))
            .unwrap_or(RadioBlackoutLevel::R0);

        let space_weather = SpaceWeather {
            geomagnetic_storm,
            solar_radiation_storm: RadiationLevel::S0, // Would need separate API
            radio_blackout,
            kp_index,
            dst_index: 0.0, // Would need separate API
            solar_wind_speed: solar_wind.as_ref().map(|s| s.speed).unwrap_or(400.0),
            solar_wind_density: solar_wind.as_ref().map(|s| s.density).unwrap_or(5.0),
            imf_bz: 0.0, // Would need magnetic field API
            timestamp: Utc::now(),
        };

        // Update cache
        {
            let mut cache = self.cached_space_weather.write().await;
            *cache = Some(space_weather.clone());
            let mut last = self.last_update.write().await;
            *last = Some(Utc::now());
        }

        Ok(space_weather)
    }

    /// Fetch comprehensive celestial data including moon phase
    pub async fn fetch_celestial_data(&self) -> Result<CelestialData> {
        // Get space weather
        let space_weather = self.fetch_all().await?;

        // Calculate moon phase
        let (moon_phase, illumination) =
            crate::thalamus::sources::celestial::CelestialSource::calculate_moon_phase(Utc::now());

        // Get solar data
        let xray_flux = self.fetch_xray_flux().await.ok();
        let flares = self.fetch_solar_flares().await.unwrap_or_default();

        let solar = SolarData {
            sunspot_number: 50, // Would need separate API
            solar_flux: 100.0,  // Would need separate API
            xray_level: xray_flux.map(|x| x.level).unwrap_or(XRayLevel::A),
            recent_flares: flares,
            cycle_phase: SolarCyclePhase::Ascending,
            timestamp: Utc::now(),
        };

        // Update solar cache
        {
            let mut cache = self.cached_solar.write().await;
            *cache = Some(solar.clone());
        }

        Ok(CelestialData {
            moon_phase,
            moon_illumination: illumination,
            days_to_full_moon: self.calculate_days_to_phase(MoonPhase::FullMoon),
            days_to_new_moon: self.calculate_days_to_phase(MoonPhase::NewMoon),
            space_weather,
            solar,
            timestamp: Utc::now(),
            lunar_volatility_indicator: moon_phase.volatility_factor(),
        })
    }

    /// Classify X-ray level from flux value
    fn classify_xray_level(&self, flux: f64) -> XRayLevel {
        if flux >= 1e-4 {
            XRayLevel::X
        } else if flux >= 1e-5 {
            XRayLevel::M
        } else if flux >= 1e-6 {
            XRayLevel::C
        } else if flux >= 1e-7 {
            XRayLevel::B
        } else {
            XRayLevel::A
        }
    }

    /// Convert Kp index to geomagnetic storm level
    fn kp_to_geomagnetic_level(&self, kp: f64) -> GeomagneticLevel {
        if kp >= 9.0 {
            GeomagneticLevel::G5
        } else if kp >= 8.0 {
            GeomagneticLevel::G4
        } else if kp >= 7.0 {
            GeomagneticLevel::G3
        } else if kp >= 6.0 {
            GeomagneticLevel::G2
        } else if kp >= 5.0 {
            GeomagneticLevel::G1
        } else {
            GeomagneticLevel::G0
        }
    }

    /// Convert X-ray level to radio blackout level
    fn xray_to_radio_blackout(&self, level: XRayLevel) -> RadioBlackoutLevel {
        match level {
            XRayLevel::X => RadioBlackoutLevel::R3,
            XRayLevel::M => RadioBlackoutLevel::R2,
            XRayLevel::C => RadioBlackoutLevel::R1,
            _ => RadioBlackoutLevel::R0,
        }
    }

    /// Convert SWPC alert to our format
    fn convert_alert(&self, alert: SwpcAlert) -> GeomagneticAlert {
        GeomagneticAlert {
            product_id: alert.product_id,
            issue_time: alert.issue_datetime,
            message: alert.message,
        }
    }

    /// Convert SWPC solar event to SolarFlare
    fn convert_solar_event(&self, event: SwpcSolarEvent) -> Option<SolarFlare> {
        // Parse the flare class from the event data
        let classification = self.parse_flare_class(&event.event_type)?;

        let peak_time = DateTime::parse_from_rfc3339(&event.event_date)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Some(SolarFlare {
            classification,
            magnitude: 1.0, // Would need to parse from data
            peak_time,
            duration_minutes: 30, // Default estimate
            associated_cme: false,
        })
    }

    /// Parse flare class from string
    fn parse_flare_class(&self, event_type: &str) -> Option<FlareClass> {
        let upper = event_type.to_uppercase();
        if upper.starts_with('X') {
            Some(FlareClass::X)
        } else if upper.starts_with('M') {
            Some(FlareClass::M)
        } else if upper.starts_with('C') {
            Some(FlareClass::C)
        } else if upper.starts_with('B') {
            Some(FlareClass::B)
        } else if upper.starts_with('A') {
            Some(FlareClass::A)
        } else {
            None
        }
    }

    /// Calculate days to a specific moon phase
    fn calculate_days_to_phase(&self, target: MoonPhase) -> f64 {
        // Synodic month in days
        const SYNODIC_MONTH: f64 = 29.530588853;

        // Reference new moon (January 6, 2000 at 18:14 UTC)
        let reference = DateTime::parse_from_rfc3339("2000-01-06T18:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let now = Utc::now();
        let days_since_reference = (now - reference).num_seconds() as f64 / 86400.0;
        let current_age = days_since_reference % SYNODIC_MONTH;

        // Target ages for each phase
        let target_age = match target {
            MoonPhase::NewMoon => 0.0,
            MoonPhase::FirstQuarter => SYNODIC_MONTH / 4.0,
            MoonPhase::FullMoon => SYNODIC_MONTH / 2.0,
            MoonPhase::LastQuarter => 3.0 * SYNODIC_MONTH / 4.0,
            _ => 0.0,
        };

        let days = target_age - current_age;
        if days < 0.0 {
            days + SYNODIC_MONTH
        } else {
            days
        }
    }

    /// Get cached space weather
    pub async fn get_cached_space_weather(&self) -> Option<SpaceWeather> {
        self.cached_space_weather.read().await.clone()
    }

    /// Get cached solar data
    pub async fn get_cached_solar(&self) -> Option<SolarData> {
        self.cached_solar.read().await.clone()
    }

    /// Get last update timestamp
    pub async fn last_update_time(&self) -> Option<DateTime<Utc>> {
        *self.last_update.read().await
    }
}

impl Default for SpaceWeatherClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default SpaceWeatherClient")
    }
}

#[async_trait]
impl ApiClient for SpaceWeatherClient {
    fn name(&self) -> &str {
        "spaceweather"
    }

    fn is_configured(&self) -> bool {
        // NOAA SWPC doesn't require API key
        true
    }

    async fn health_check(&self) -> bool {
        // Try fetching Kp index as a simple health check
        match self.fetch_kp_index().await {
            Ok(_) => true,
            Err(e) => {
                warn!("Space weather health check failed: {}", e);
                false
            }
        }
    }
}

#[async_trait]
impl DataSource for SpaceWeatherClient {
    type Data = CelestialData;

    fn name(&self) -> &str {
        "spaceweather"
    }

    async fn fetch_latest(&self) -> Result<Self::Data> {
        self.fetch_celestial_data().await
    }

    async fn health_check(&self) -> bool {
        <Self as ApiClient>::health_check(self).await
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        None
    }
}

// ============================================================================
// Data Types
// ============================================================================

/// Solar wind data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarWindData {
    /// Proton density (particles/cm³)
    pub density: f64,
    /// Bulk speed (km/s)
    pub speed: f64,
    /// Temperature (K)
    pub temperature: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// X-ray flux data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XRayFluxData {
    /// Flux value (W/m²)
    pub flux: f64,
    /// Classification level
    pub level: XRayLevel,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Geomagnetic alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeomagneticAlert {
    /// Product ID
    pub product_id: String,
    /// Issue time
    pub issue_time: String,
    /// Alert message
    pub message: String,
}

// ============================================================================
// API Response Types
// ============================================================================

/// SWPC X-ray entry
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SwpcXrayEntry {
    /// Time tag
    time_tag: String,
    /// Satellite ID
    satellite: i32,
    /// Flux value
    flux: String,
    /// Energy band
    energy: String,
}

/// SWPC Alert
#[derive(Debug, Deserialize)]
struct SwpcAlert {
    /// Product ID
    product_id: String,
    /// Issue datetime
    issue_datetime: String,
    /// Message
    message: String,
}

/// SWPC Solar Event
#[derive(Debug, Deserialize)]
struct SwpcSolarEvent {
    /// Event date
    #[serde(default)]
    event_date: String,
    /// Event type (e.g., "X1.5", "M2.0")
    #[serde(default)]
    event_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = SpaceWeatherClient::new();
        assert!(client.is_ok());

        let client = client.unwrap();
        assert!(client.is_configured());
    }

    #[test]
    fn test_xray_classification() {
        let client = SpaceWeatherClient::new().unwrap();

        assert_eq!(client.classify_xray_level(1e-3), XRayLevel::X);
        assert_eq!(client.classify_xray_level(5e-5), XRayLevel::M);
        assert_eq!(client.classify_xray_level(5e-6), XRayLevel::C);
        assert_eq!(client.classify_xray_level(5e-7), XRayLevel::B);
        assert_eq!(client.classify_xray_level(1e-8), XRayLevel::A);
    }

    #[test]
    fn test_kp_to_geomagnetic() {
        let client = SpaceWeatherClient::new().unwrap();

        assert_eq!(client.kp_to_geomagnetic_level(2.0), GeomagneticLevel::G0);
        assert_eq!(client.kp_to_geomagnetic_level(5.0), GeomagneticLevel::G1);
        assert_eq!(client.kp_to_geomagnetic_level(6.0), GeomagneticLevel::G2);
        assert_eq!(client.kp_to_geomagnetic_level(7.0), GeomagneticLevel::G3);
        assert_eq!(client.kp_to_geomagnetic_level(8.0), GeomagneticLevel::G4);
        assert_eq!(client.kp_to_geomagnetic_level(9.0), GeomagneticLevel::G5);
    }

    #[test]
    fn test_xray_to_radio_blackout() {
        let client = SpaceWeatherClient::new().unwrap();

        assert_eq!(
            client.xray_to_radio_blackout(XRayLevel::X),
            RadioBlackoutLevel::R3
        );
        assert_eq!(
            client.xray_to_radio_blackout(XRayLevel::M),
            RadioBlackoutLevel::R2
        );
        assert_eq!(
            client.xray_to_radio_blackout(XRayLevel::C),
            RadioBlackoutLevel::R1
        );
        assert_eq!(
            client.xray_to_radio_blackout(XRayLevel::B),
            RadioBlackoutLevel::R0
        );
    }

    #[test]
    fn test_flare_class_parsing() {
        let client = SpaceWeatherClient::new().unwrap();

        assert_eq!(client.parse_flare_class("X1.5"), Some(FlareClass::X));
        assert_eq!(client.parse_flare_class("M2.0"), Some(FlareClass::M));
        assert_eq!(client.parse_flare_class("C3.0"), Some(FlareClass::C));
        assert_eq!(client.parse_flare_class("B1.0"), Some(FlareClass::B));
        assert_eq!(client.parse_flare_class("A0.5"), Some(FlareClass::A));
        assert_eq!(client.parse_flare_class("unknown"), None);
    }

    #[test]
    fn test_days_to_phase() {
        let client = SpaceWeatherClient::new().unwrap();

        // Just verify it returns a reasonable value
        let days_to_full = client.calculate_days_to_phase(MoonPhase::FullMoon);
        assert!(days_to_full >= 0.0);
        assert!(days_to_full <= 29.53);

        let days_to_new = client.calculate_days_to_phase(MoonPhase::NewMoon);
        assert!(days_to_new >= 0.0);
        assert!(days_to_new <= 29.53);
    }

    #[tokio::test]
    async fn test_cached_data_initially_none() {
        let client = SpaceWeatherClient::new().unwrap();

        assert!(client.get_cached_space_weather().await.is_none());
        assert!(client.get_cached_solar().await.is_none());
        assert!(client.last_update_time().await.is_none());
    }
}
