//! Risk Manager - Stub implementation

pub struct RiskManager {
    enabled: bool,
}

impl RiskManager {
    pub fn new() -> Self {
        Self { enabled: true }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for RiskManager {
    fn default() -> Self {
        Self::new()
    }
}
