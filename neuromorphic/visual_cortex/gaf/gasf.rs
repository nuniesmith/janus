//! Gramian Angular Summation Field
//!
//! Part of the Visual Cortex region
//! Component: gaf
//!
//! GASF encodes temporal correlation by computing cos(θ_i + θ_j)
//! where θ represents the polar angle of normalized time series values.

use crate::common::Result;

/// Gramian Angular Summation Field
///
/// Encodes time series as 2D images using summation of polar angles.
/// GASF preserves temporal correlation and is useful for pattern recognition.
pub struct Gasf {
    pub image_size: usize,
}

impl Default for Gasf {
    fn default() -> Self {
        Self::new(32)
    }
}

impl Gasf {
    /// Create a new GASF encoder
    pub fn new(image_size: usize) -> Self {
        Self { image_size }
    }

    /// Encode time series to GASF image
    pub fn encode(&self, time_series: &[f64]) -> Result<Vec<Vec<f64>>> {
        use super::encoding::{Encoding, GAFType};

        let encoder = Encoding::new(self.image_size, GAFType::GASF);
        encoder.encode(time_series)
    }

    /// Main processing function
    pub fn process(&self, time_series: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.encode(time_series)
    }

    /// Normalize time series to [-1, 1] range
    #[allow(dead_code)]
    fn normalize(&self, series: &[f64]) -> Vec<f64> {
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return vec![0.0; series.len()];
        }

        series
            .iter()
            .map(|&x| 2.0 * (x - min) / (max - min) - 1.0)
            .collect()
    }

    /// Convert normalized values to polar angles
    #[allow(dead_code)]
    fn to_polar(&self, normalized: &[f64]) -> Vec<f64> {
        normalized
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-1.0, 1.0);
                clamped.acos()
            })
            .collect()
    }

    /// Compute GASF matrix: cos(θ_i + θ_j)
    #[allow(dead_code)]
    #[allow(clippy::needless_range_loop)]
    fn compute_gasf_matrix(&self, angles: &[f64]) -> Vec<Vec<f64>> {
        let n = angles.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = (angles[i] + angles[j]).cos();
            }
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let gasf = Gasf::new(8);
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = gasf.encode(&time_series);
        assert!(result.is_ok());

        let matrix = result.unwrap();
        assert_eq!(matrix.len(), 8);
        assert_eq!(matrix[0].len(), 8);
    }

    #[test]
    fn test_normalization() {
        let gasf = Gasf::new(4);
        let series = vec![0.0, 1.0, 2.0, 3.0];
        let normalized = gasf.normalize(&series);

        assert!(normalized.iter().all(|&x| (-1.0..=1.0).contains(&x)));
        assert!((normalized[0] + 1.0).abs() < 1e-10);
        assert!((normalized[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let gasf = Gasf::new(4);
        let time_series = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = gasf.encode(&time_series).unwrap();

        // GASF matrix should be symmetric
        for (i, row) in matrix.iter().enumerate().take(4) {
            for (j, _) in row.iter().enumerate().take(4) {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_diagonal() {
        let gasf = Gasf::new(4);
        let time_series = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = gasf.encode(&time_series).unwrap();

        // Diagonal elements represent cos(2θ_i)
        for (i, row) in matrix.iter().enumerate().take(4) {
            assert!((-1.0..=1.0).contains(&row[i]));
        }
    }
}
