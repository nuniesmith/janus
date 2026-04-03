//! Drive State - Motivational drives for trading

pub struct DriveState {
    profit_drive: f32,
    safety_drive: f32,
}

impl DriveState {
    pub fn new() -> Self {
        Self {
            profit_drive: 0.5,
            safety_drive: 0.8,
        }
    }

    pub fn profit_drive(&self) -> f32 {
        self.profit_drive
    }

    pub fn safety_drive(&self) -> f32 {
        self.safety_drive
    }
}

impl Default for DriveState {
    fn default() -> Self {
        Self::new()
    }
}
