//! Preprocessing - Market data preprocessing

pub struct Preprocessor {
    normalize: bool,
}

impl Preprocessor {
    pub fn new(normalize: bool) -> Self {
        Self { normalize }
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }
}
