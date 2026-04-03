//! Motor Control - Fine-grained execution control

pub struct MotorControl {
    precision: f32,
}

impl MotorControl {
    pub fn new(precision: f32) -> Self {
        Self { precision }
    }

    pub fn precision(&self) -> f32 {
        self.precision
    }
}
