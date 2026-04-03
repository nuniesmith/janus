//! Celestial Data Sources
//!
//! Part of Thalamus region - External Data Sources
//!
//! This module handles celestial/astronomical data relevant for trading:
//! - Moon phases (historical correlation with market volatility)
//! - Space weather (solar flares affecting communications/satellites)
//! - Planetary alignments (alternative data for sentiment analysis)
//! - Orbital mechanics (satellite positions for logistics tracking)
//!
//! While some of these correlations may be spurious, they can serve as:
//! - Alternative sentiment indicators
//! - Calendar effects analysis
//! - Risk factors for space-dependent infrastructure

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::DataSource;

/// Aggregated celestial data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialData {
    /// Current moon phase
    pub moon_phase: MoonPhase,

    /// Moon illumination percentage (0-100)
    pub moon_illumination: f64,

    /// Days until next full moon
    pub days_to_full_moon: f64,

    /// Days until next new moon
    pub days_to_new_moon: f64,

    /// Current space weather conditions
    pub space_weather: SpaceWeather,

    /// Solar data
    pub solar: SolarData,

    /// Timestamp of the data
    pub timestamp: DateTime<Utc>,

    /// Lunar cycle volatility indicator
    /// Based on historical correlation between moon phases and market volatility
    pub lunar_volatility_indicator: f64,
}

impl CelestialData {
    /// Create celestial data with minimal fields
    pub fn new(moon_phase: MoonPhase) -> Self {
        Self {
            moon_phase,
            moon_illumination: moon_phase.typical_illumination(),
            days_to_full_moon: 0.0,
            days_to_new_moon: 0.0,
            space_weather: SpaceWeather::default(),
            solar: SolarData::default(),
            timestamp: Utc::now(),
            lunar_volatility_indicator: moon_phase.volatility_factor(),
        }
    }

    /// Calculate lunar volatility indicator
    /// Research suggests volatility may increase around new and full moons
    pub fn calculate_volatility_indicator(&self) -> f64 {
        self.moon_phase.volatility_factor()
    }

    /// Check if space weather could affect communications
    pub fn communication_risk(&self) -> bool {
        self.space_weather.is_severe() || self.solar.is_stormy()
    }

    /// Get trading sentiment based on celestial factors
    /// Returns a value from -1.0 (bearish) to 1.0 (bullish)
    pub fn sentiment_indicator(&self) -> f64 {
        // This is a placeholder implementation
        // Real implementation would use historical correlation data
        let moon_factor: f64 = match self.moon_phase {
            MoonPhase::FullMoon => 0.1, // Slightly bullish historically
            MoonPhase::NewMoon => -0.1, // Slightly bearish historically
            _ => 0.0,
        };

        let space_factor: f64 = if self.space_weather.is_severe() {
            -0.1 // Uncertainty = bearish
        } else {
            0.0
        };

        (moon_factor + space_factor).clamp(-1.0, 1.0)
    }
}

impl Default for CelestialData {
    fn default() -> Self {
        Self::new(MoonPhase::Unknown)
    }
}

/// Moon phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoonPhase {
    /// New moon (0% illumination)
    NewMoon,
    /// Waxing crescent (1-49% illumination, increasing)
    WaxingCrescent,
    /// First quarter (50% illumination, right half)
    FirstQuarter,
    /// Waxing gibbous (51-99% illumination, increasing)
    WaxingGibbous,
    /// Full moon (100% illumination)
    FullMoon,
    /// Waning gibbous (99-51% illumination, decreasing)
    WaningGibbous,
    /// Last quarter (50% illumination, left half)
    LastQuarter,
    /// Waning crescent (49-1% illumination, decreasing)
    WaningCrescent,
    /// Unknown phase
    Unknown,
}

impl MoonPhase {
    /// Calculate moon phase from illumination and whether it's waxing
    pub fn from_illumination(illumination: f64, is_waxing: bool) -> Self {
        match (illumination, is_waxing) {
            (i, _) if i < 1.0 => MoonPhase::NewMoon,
            (i, true) if i < 50.0 => MoonPhase::WaxingCrescent,
            (i, true) if i < 51.0 => MoonPhase::FirstQuarter,
            (i, true) if i < 99.0 => MoonPhase::WaxingGibbous,
            (i, _) if i >= 99.0 => MoonPhase::FullMoon,
            (i, false) if i >= 51.0 => MoonPhase::WaningGibbous,
            (i, false) if i >= 49.0 => MoonPhase::LastQuarter,
            (_, false) => MoonPhase::WaningCrescent,
            _ => MoonPhase::Unknown,
        }
    }

    /// Get typical illumination for this phase
    pub fn typical_illumination(&self) -> f64 {
        match self {
            MoonPhase::NewMoon => 0.0,
            MoonPhase::WaxingCrescent => 25.0,
            MoonPhase::FirstQuarter => 50.0,
            MoonPhase::WaxingGibbous => 75.0,
            MoonPhase::FullMoon => 100.0,
            MoonPhase::WaningGibbous => 75.0,
            MoonPhase::LastQuarter => 50.0,
            MoonPhase::WaningCrescent => 25.0,
            MoonPhase::Unknown => 50.0,
        }
    }

    /// Get volatility factor for this phase
    /// Based on "lunar effect" research in financial markets
    pub fn volatility_factor(&self) -> f64 {
        match self {
            MoonPhase::NewMoon => 1.15,      // Higher volatility
            MoonPhase::FullMoon => 1.10,     // Higher volatility
            MoonPhase::FirstQuarter => 1.05, // Slightly elevated
            MoonPhase::LastQuarter => 1.05,  // Slightly elevated
            _ => 1.0,                        // Normal volatility
        }
    }

    /// Get phase name
    pub fn name(&self) -> &str {
        match self {
            MoonPhase::NewMoon => "New Moon",
            MoonPhase::WaxingCrescent => "Waxing Crescent",
            MoonPhase::FirstQuarter => "First Quarter",
            MoonPhase::WaxingGibbous => "Waxing Gibbous",
            MoonPhase::FullMoon => "Full Moon",
            MoonPhase::WaningGibbous => "Waning Gibbous",
            MoonPhase::LastQuarter => "Last Quarter",
            MoonPhase::WaningCrescent => "Waning Crescent",
            MoonPhase::Unknown => "Unknown",
        }
    }

    /// Check if this is a major phase (new, full, or quarter)
    pub fn is_major_phase(&self) -> bool {
        matches!(
            self,
            MoonPhase::NewMoon
                | MoonPhase::FullMoon
                | MoonPhase::FirstQuarter
                | MoonPhase::LastQuarter
        )
    }
}

impl Default for MoonPhase {
    fn default() -> Self {
        MoonPhase::Unknown
    }
}

/// Space weather conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceWeather {
    /// Geomagnetic storm level (G0-G5)
    pub geomagnetic_storm: GeomagneticLevel,

    /// Solar radiation storm level (S0-S5)
    pub solar_radiation_storm: RadiationLevel,

    /// Radio blackout level (R0-R5)
    pub radio_blackout: RadioBlackoutLevel,

    /// Kp index (0-9, measure of geomagnetic activity)
    pub kp_index: f64,

    /// Dst index (disturbance storm time)
    pub dst_index: f64,

    /// Solar wind speed (km/s)
    pub solar_wind_speed: f64,

    /// Solar wind density (protons/cm³)
    pub solar_wind_density: f64,

    /// Interplanetary magnetic field Bz component (nT)
    /// Negative = more likely to cause geomagnetic storms
    pub imf_bz: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl SpaceWeather {
    /// Create default space weather (quiet conditions)
    pub fn quiet() -> Self {
        Self {
            geomagnetic_storm: GeomagneticLevel::G0,
            solar_radiation_storm: RadiationLevel::S0,
            radio_blackout: RadioBlackoutLevel::R0,
            kp_index: 2.0,
            dst_index: 0.0,
            solar_wind_speed: 400.0,
            solar_wind_density: 5.0,
            imf_bz: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Check if space weather is severe enough to impact operations
    pub fn is_severe(&self) -> bool {
        self.geomagnetic_storm >= GeomagneticLevel::G3
            || self.solar_radiation_storm >= RadiationLevel::S3
            || self.radio_blackout >= RadioBlackoutLevel::R3
            || self.kp_index >= 7.0
    }

    /// Check if space weather could affect GPS/navigation
    pub fn gps_risk(&self) -> bool {
        self.geomagnetic_storm >= GeomagneticLevel::G2 || self.kp_index >= 5.0
    }

    /// Check if space weather could affect HF radio communications
    pub fn hf_radio_risk(&self) -> bool {
        self.radio_blackout >= RadioBlackoutLevel::R2
    }

    /// Get overall severity score (0-10)
    pub fn severity_score(&self) -> f64 {
        let geo_score = self.geomagnetic_storm.score();
        let rad_score = self.solar_radiation_storm.score();
        let radio_score = self.radio_blackout.score();
        let kp_score = self.kp_index / 9.0 * 10.0;

        (geo_score + rad_score + radio_score + kp_score) / 4.0
    }
}

impl Default for SpaceWeather {
    fn default() -> Self {
        Self::quiet()
    }
}

/// Geomagnetic storm levels (NOAA G-Scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GeomagneticLevel {
    G0, // Quiet
    G1, // Minor
    G2, // Moderate
    G3, // Strong
    G4, // Severe
    G5, // Extreme
}

impl GeomagneticLevel {
    /// Get numeric score (0-10)
    pub fn score(&self) -> f64 {
        match self {
            GeomagneticLevel::G0 => 0.0,
            GeomagneticLevel::G1 => 2.0,
            GeomagneticLevel::G2 => 4.0,
            GeomagneticLevel::G3 => 6.0,
            GeomagneticLevel::G4 => 8.0,
            GeomagneticLevel::G5 => 10.0,
        }
    }

    /// Get description
    pub fn description(&self) -> &str {
        match self {
            GeomagneticLevel::G0 => "Quiet",
            GeomagneticLevel::G1 => "Minor storm",
            GeomagneticLevel::G2 => "Moderate storm",
            GeomagneticLevel::G3 => "Strong storm",
            GeomagneticLevel::G4 => "Severe storm",
            GeomagneticLevel::G5 => "Extreme storm",
        }
    }
}

impl Default for GeomagneticLevel {
    fn default() -> Self {
        GeomagneticLevel::G0
    }
}

/// Solar radiation storm levels (NOAA S-Scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RadiationLevel {
    S0, // None
    S1, // Minor
    S2, // Moderate
    S3, // Strong
    S4, // Severe
    S5, // Extreme
}

impl RadiationLevel {
    /// Get numeric score (0-10)
    pub fn score(&self) -> f64 {
        match self {
            RadiationLevel::S0 => 0.0,
            RadiationLevel::S1 => 2.0,
            RadiationLevel::S2 => 4.0,
            RadiationLevel::S3 => 6.0,
            RadiationLevel::S4 => 8.0,
            RadiationLevel::S5 => 10.0,
        }
    }
}

impl Default for RadiationLevel {
    fn default() -> Self {
        RadiationLevel::S0
    }
}

/// Radio blackout levels (NOAA R-Scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RadioBlackoutLevel {
    R0, // None
    R1, // Minor
    R2, // Moderate
    R3, // Strong
    R4, // Severe
    R5, // Extreme
}

impl RadioBlackoutLevel {
    /// Get numeric score (0-10)
    pub fn score(&self) -> f64 {
        match self {
            RadioBlackoutLevel::R0 => 0.0,
            RadioBlackoutLevel::R1 => 2.0,
            RadioBlackoutLevel::R2 => 4.0,
            RadioBlackoutLevel::R3 => 6.0,
            RadioBlackoutLevel::R4 => 8.0,
            RadioBlackoutLevel::R5 => 10.0,
        }
    }
}

impl Default for RadioBlackoutLevel {
    fn default() -> Self {
        RadioBlackoutLevel::R0
    }
}

/// Solar activity data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarData {
    /// Sunspot number
    pub sunspot_number: u32,

    /// Solar flux (10.7 cm radio flux in SFU)
    pub solar_flux: f64,

    /// X-ray background level
    pub xray_level: XRayLevel,

    /// Recent solar flares (last 24 hours)
    pub recent_flares: Vec<SolarFlare>,

    /// Current solar cycle phase
    pub cycle_phase: SolarCyclePhase,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl SolarData {
    /// Check if solar activity is stormy
    pub fn is_stormy(&self) -> bool {
        self.xray_level >= XRayLevel::M
            || self
                .recent_flares
                .iter()
                .any(|f| f.classification >= FlareClass::M)
    }

    /// Get activity level (0-10)
    pub fn activity_level(&self) -> f64 {
        let sunspot_factor = (self.sunspot_number as f64 / 200.0).min(1.0) * 5.0;
        let xray_factor = self.xray_level.score() / 2.0;
        sunspot_factor + xray_factor
    }
}

impl Default for SolarData {
    fn default() -> Self {
        Self {
            sunspot_number: 50,
            solar_flux: 100.0,
            xray_level: XRayLevel::A,
            recent_flares: Vec::new(),
            cycle_phase: SolarCyclePhase::Ascending,
            timestamp: Utc::now(),
        }
    }
}

/// X-ray background levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum XRayLevel {
    A, // Quiet
    B, // Low
    C, // Moderate
    M, // Active
    X, // Major
}

impl XRayLevel {
    /// Get numeric score (0-10)
    pub fn score(&self) -> f64 {
        match self {
            XRayLevel::A => 0.0,
            XRayLevel::B => 2.0,
            XRayLevel::C => 4.0,
            XRayLevel::M => 7.0,
            XRayLevel::X => 10.0,
        }
    }
}

impl Default for XRayLevel {
    fn default() -> Self {
        XRayLevel::A
    }
}

/// Solar flare classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FlareClass {
    A, // Minor
    B, // Small
    C, // Moderate
    M, // Major
    X, // Extreme
}

/// Solar flare event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarFlare {
    /// Flare classification
    pub classification: FlareClass,

    /// Numeric magnitude (e.g., M1.5, X2.0)
    pub magnitude: f64,

    /// Peak time
    pub peak_time: DateTime<Utc>,

    /// Duration in minutes
    pub duration_minutes: u32,

    /// Associated CME (Coronal Mass Ejection)
    pub associated_cme: bool,
}

/// Solar cycle phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolarCyclePhase {
    Minimum,
    Ascending,
    Maximum,
    Descending,
}

impl Default for SolarCyclePhase {
    fn default() -> Self {
        SolarCyclePhase::Ascending
    }
}

/// Celestial source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialSourceConfig {
    /// API key (if using external service)
    pub api_key: Option<String>,

    /// Base URL for API
    pub base_url: String,

    /// Enable moon phase tracking
    pub track_moon: bool,

    /// Enable space weather tracking
    pub track_space_weather: bool,

    /// Enable solar activity tracking
    pub track_solar: bool,

    /// Update interval in seconds
    pub update_interval: u64,

    /// Observer latitude (for moon calculations)
    pub observer_latitude: f64,

    /// Observer longitude (for moon calculations)
    pub observer_longitude: f64,
}

impl Default for CelestialSourceConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            track_moon: true,
            track_space_weather: true,
            track_solar: true,
            update_interval: 3600, // 1 hour
            observer_latitude: 0.0,
            observer_longitude: 0.0,
        }
    }
}

/// Celestial data source implementation
pub struct CelestialSource {
    /// Source name
    name: String,

    /// Configuration
    config: CelestialSourceConfig,

    /// Last update timestamp
    last_update: Option<DateTime<Utc>>,

    /// Cached celestial data
    cached_data: Option<CelestialData>,
}

impl CelestialSource {
    /// Create a new celestial source
    pub fn new(name: String, config: CelestialSourceConfig) -> Self {
        Self {
            name,
            config,
            last_update: None,
            cached_data: None,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CelestialSourceConfig {
        &self.config
    }

    /// Get cached data
    pub fn cached(&self) -> Option<&CelestialData> {
        self.cached_data.as_ref()
    }

    /// Calculate moon phase for a given timestamp
    /// Uses simplified astronomical calculation
    pub fn calculate_moon_phase(timestamp: DateTime<Utc>) -> (MoonPhase, f64) {
        // Synodic month (average time between new moons) in days
        const SYNODIC_MONTH: f64 = 29.530588853;

        // Reference new moon (January 6, 2000 at 18:14 UTC)
        let reference = DateTime::parse_from_rfc3339("2000-01-06T18:14:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let days_since_reference = (timestamp - reference).num_seconds() as f64 / 86400.0;
        let moon_age = days_since_reference % SYNODIC_MONTH;

        // Normalize to 0-1 range
        let phase_fraction = moon_age / SYNODIC_MONTH;

        // Calculate illumination (simplified)
        let illumination = (1.0 - (2.0 * PI * phase_fraction).cos()) / 2.0 * 100.0;

        // Determine if waxing (first half of cycle)
        let is_waxing = phase_fraction < 0.5;

        let phase = MoonPhase::from_illumination(illumination, is_waxing);

        (phase, illumination)
    }

    /// Fetch celestial data (placeholder - implement for each provider)
    async fn fetch_data(&self) -> crate::common::Result<CelestialData> {
        // Calculate moon phase locally
        let (moon_phase, illumination) = Self::calculate_moon_phase(Utc::now());

        let mut data = CelestialData::new(moon_phase);
        data.moon_illumination = illumination;

        // Space weather and solar data would typically come from:
        // - NOAA Space Weather Prediction Center (https://www.swpc.noaa.gov/)
        // - NASA APIs
        // - SDO (Solar Dynamics Observatory)

        Ok(data)
    }
}

#[async_trait]
impl DataSource for CelestialSource {
    type Data = CelestialData;

    fn name(&self) -> &str {
        &self.name
    }

    async fn fetch_latest(&self) -> crate::common::Result<Self::Data> {
        self.fetch_data().await
    }

    async fn health_check(&self) -> bool {
        // Moon calculations are always available locally
        // Space weather requires API access
        self.config.track_moon
            || (self.config.track_space_weather
                && (self.config.api_key.is_some() || !self.config.base_url.is_empty()))
    }

    fn last_update(&self) -> Option<DateTime<Utc>> {
        self.last_update
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moon_phase_creation() {
        let phase = MoonPhase::FullMoon;
        assert_eq!(phase.typical_illumination(), 100.0);
        assert!(phase.is_major_phase());
    }

    #[test]
    fn test_moon_phase_from_illumination() {
        assert_eq!(MoonPhase::from_illumination(0.0, true), MoonPhase::NewMoon);
        assert_eq!(
            MoonPhase::from_illumination(100.0, true),
            MoonPhase::FullMoon
        );
        assert_eq!(
            MoonPhase::from_illumination(50.0, true),
            MoonPhase::FirstQuarter
        );
        assert_eq!(
            MoonPhase::from_illumination(50.0, false),
            MoonPhase::LastQuarter
        );
    }

    #[test]
    fn test_volatility_factors() {
        assert!(MoonPhase::NewMoon.volatility_factor() > 1.0);
        assert!(MoonPhase::FullMoon.volatility_factor() > 1.0);
        assert_eq!(MoonPhase::WaxingCrescent.volatility_factor(), 1.0);
    }

    #[test]
    fn test_space_weather_severity() {
        let quiet = SpaceWeather::quiet();
        assert!(!quiet.is_severe());

        let severe = SpaceWeather {
            geomagnetic_storm: GeomagneticLevel::G4,
            ..SpaceWeather::quiet()
        };
        assert!(severe.is_severe());
    }

    #[test]
    fn test_celestial_data_creation() {
        let data = CelestialData::new(MoonPhase::FullMoon);
        assert_eq!(data.moon_phase, MoonPhase::FullMoon);
        assert_eq!(data.moon_illumination, 100.0);
    }

    #[test]
    fn test_moon_phase_calculation() {
        // Test that calculation returns valid phase
        let (phase, illumination) = CelestialSource::calculate_moon_phase(Utc::now());
        assert!((0.0..=100.0).contains(&illumination));
        assert_ne!(phase, MoonPhase::Unknown);
    }

    #[test]
    fn test_geomagnetic_level_ordering() {
        assert!(GeomagneticLevel::G5 > GeomagneticLevel::G0);
        assert!(GeomagneticLevel::G3 > GeomagneticLevel::G2);
    }

    #[test]
    fn test_solar_data_activity() {
        let quiet_solar = SolarData::default();
        assert!(!quiet_solar.is_stormy());

        let active_solar = SolarData {
            xray_level: XRayLevel::X,
            ..SolarData::default()
        };
        assert!(active_solar.is_stormy());
    }

    #[test]
    fn test_celestial_source_config_default() {
        let config = CelestialSourceConfig::default();
        assert!(config.track_moon);
        assert!(config.track_space_weather);
        assert!(config.track_solar);
    }

    #[test]
    fn test_sentiment_indicator() {
        let full_moon_data = CelestialData::new(MoonPhase::FullMoon);
        let sentiment = full_moon_data.sentiment_indicator();
        assert!((-1.0..=1.0).contains(&sentiment));
    }
}
