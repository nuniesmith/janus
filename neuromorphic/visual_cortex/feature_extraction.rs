//! Feature Extraction - Extract features from preprocessed data

pub struct FeatureExtractor {
    num_features: usize,
}

impl FeatureExtractor {
    pub fn new(num_features: usize) -> Self {
        Self { num_features }
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }
}
