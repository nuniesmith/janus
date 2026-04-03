//! Error Correction - Learn from execution errors

pub struct ErrorCorrection {
    learning_rate: f32,
}

impl ErrorCorrection {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}
