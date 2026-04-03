//! Forward Model - Predict execution outcomes

pub struct ForwardModel {
    prediction_horizon: usize,
}

impl ForwardModel {
    pub fn new(prediction_horizon: usize) -> Self {
        Self { prediction_horizon }
    }

    pub fn prediction_horizon(&self) -> usize {
        self.prediction_horizon
    }
}
