//! Regulation - Homeostatic regulation

pub struct Regulator {
    target_balance: f64,
}

impl Regulator {
    pub fn new(target_balance: f64) -> Self {
        Self { target_balance }
    }

    pub fn target_balance(&self) -> f64 {
        self.target_balance
    }
}
