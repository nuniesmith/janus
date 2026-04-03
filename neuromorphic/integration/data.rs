//! Market Data Preprocessing and GAF Transformation Pipeline
//!
//! This module handles loading market data (candles), preprocessing, and transforming
//! into GAF (Gramian Angular Field) images suitable for ViViT training.
//!
//! # Pipeline Overview
//!
//! 1. **Load Candles**: Read OHLCV data from CSV, database, or memory
//! 2. **Preprocess**: Normalize, handle missing values, create sequences
//! 3. **GAF Transform**: Convert time series to 2D images using GAF encoding
//! 4. **Label Generation**: Create labels based on future price movements
//! 5. **Batch Creation**: Group samples into batches for training
//!
//! # Example
//!
//! ```ignore
//! use janus_neuromorphic::integration::data::{MarketDataPipeline, PipelineConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = PipelineConfig::default();
//! let mut pipeline = MarketDataPipeline::new(config);
//!
//! // Load data from CSV
//! pipeline.load_from_csv("data/btc_usd.csv").await?;
//!
//! // Create training batches
//! while let Some(batch) = pipeline.next_batch().await? {
//!     // Train on batch
//! }
//! # Ok(())
//! # }
//! ```

use candle_core::{Device, Result as CandleResult, Tensor};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

/// Market data preprocessing pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of frames per video sequence
    pub num_frames: usize,
    /// GAF image size (height and width)
    pub gaf_image_size: usize,
    /// Number of candles per frame
    pub candles_per_frame: usize,
    /// Features to encode (OHLCV)
    pub features: Vec<GafFeature>,
    /// Prediction horizon (candles ahead)
    pub prediction_horizon: usize,
    /// Price change threshold for buy signal (%)
    pub buy_threshold: f64,
    /// Price change threshold for sell signal (%)
    pub sell_threshold: f64,
    /// Train/validation split ratio
    pub train_split: f64,
    /// Shuffle data
    pub shuffle: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            gaf_image_size: 224,
            candles_per_frame: 60,
            features: vec![GafFeature::Close, GafFeature::Volume, GafFeature::HighLow],
            prediction_horizon: 10,
            buy_threshold: 0.5,
            sell_threshold: -0.5,
            train_split: 0.8,
            shuffle: true,
            seed: Some(42),
        }
    }
}

/// Features that can be encoded as GAF images
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GafFeature {
    /// Close price
    Close,
    /// Open price
    Open,
    /// High price
    High,
    /// Low price
    Low,
    /// Volume
    Volume,
    /// High-Low spread
    HighLow,
    /// Close-Open (candle body)
    CloseOpen,
}

/// Single candle (OHLCV)
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Labeled training sample
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// GAF-encoded frames: [num_frames, height, width, channels]
    pub frames: Vec<Vec<Vec<Vec<f32>>>>,
    /// Label: 0=sell, 1=hold, 2=buy
    pub label: usize,
    /// Original timestamp
    pub timestamp: i64,
    /// Future price change (%)
    pub future_return: f64,
}

/// Batch of training samples
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Input tensor: [batch_size, num_frames, height, width, channels]
    pub inputs: Tensor,
    /// Labels: [batch_size]
    pub labels: Tensor,
    /// Metadata
    pub timestamps: Vec<i64>,
}

/// Main market data preprocessing pipeline
pub struct MarketDataPipeline {
    /// Configuration
    config: PipelineConfig,
    /// Raw candle data
    candles: Vec<Candle>,
    /// Preprocessed samples
    samples: Vec<TrainingSample>,
    /// Current batch index
    current_idx: usize,
    /// Training indices
    train_indices: Vec<usize>,
    /// Validation indices
    val_indices: Vec<usize>,
}

impl MarketDataPipeline {
    /// Create a new pipeline
    pub fn new(config: PipelineConfig) -> Self {
        info!("Initializing market data pipeline");
        info!("  Num frames: {}", config.num_frames);
        info!("  GAF size: {}", config.gaf_image_size);
        info!("  Features: {:?}", config.features);

        Self {
            config,
            candles: Vec::new(),
            samples: Vec::new(),
            current_idx: 0,
            train_indices: Vec::new(),
            val_indices: Vec::new(),
        }
    }

    /// Load candles from CSV file
    ///
    /// Expected format: timestamp,open,high,low,close,volume
    pub async fn load_from_csv<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        info!("Loading candles from CSV: {:?}", path.as_ref());

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let mut count = 0;
        for result in reader.records() {
            let record = result?;

            let candle = Candle {
                timestamp: record[0].parse()?,
                open: record[1].parse()?,
                high: record[2].parse()?,
                low: record[3].parse()?,
                close: record[4].parse()?,
                volume: record[5].parse()?,
            };

            self.candles.push(candle);
            count += 1;
        }

        info!("Loaded {} candles", count);
        Ok(())
    }

    /// Load candles from vector (for testing or in-memory data)
    pub fn load_from_vec(&mut self, candles: Vec<Candle>) {
        info!("Loading {} candles from vector", candles.len());
        self.candles = candles;
    }

    /// Preprocess data and create training samples
    pub async fn preprocess(&mut self) -> anyhow::Result<()> {
        info!("Preprocessing data...");

        if self.candles.is_empty() {
            return Err(anyhow::anyhow!("No candles loaded"));
        }

        let sequence_length = self.config.num_frames * self.config.candles_per_frame;
        let total_length = sequence_length + self.config.prediction_horizon;

        if self.candles.len() < total_length {
            return Err(anyhow::anyhow!(
                "Not enough candles. Need at least {}, have {}",
                total_length,
                self.candles.len()
            ));
        }

        // Create sliding windows
        let num_samples = self.candles.len() - total_length + 1;
        info!("Creating {} samples with sliding window", num_samples);

        for i in 0..num_samples {
            let sample = self.create_sample(i)?;
            self.samples.push(sample);
        }

        info!("Created {} training samples", self.samples.len());

        // Split into train/val
        self.split_train_val();

        Ok(())
    }

    /// Create a single training sample from candle data
    fn create_sample(&self, start_idx: usize) -> anyhow::Result<TrainingSample> {
        let sequence_length = self.config.num_frames * self.config.candles_per_frame;
        let end_idx = start_idx + sequence_length;

        // Extract candles for this sample
        let sample_candles = &self.candles[start_idx..end_idx];

        // Create GAF frames (one per feature channel)
        let mut all_frames = Vec::new();

        for feature in &self.config.features {
            let frames = self.create_gaf_frames(sample_candles, *feature)?;
            all_frames.push(frames);
        }

        // Transpose to [num_frames, height, width, channels]
        let frames = self.transpose_frames(all_frames)?;

        // Compute label based on future price movement
        let current_price = sample_candles.last().unwrap().close;
        let future_idx = end_idx + self.config.prediction_horizon - 1;
        let future_price = self.candles[future_idx].close;
        let future_return = ((future_price - current_price) / current_price) * 100.0;

        let label = if future_return >= self.config.buy_threshold {
            2 // Buy
        } else if future_return <= self.config.sell_threshold {
            0 // Sell
        } else {
            1 // Hold
        };

        Ok(TrainingSample {
            frames,
            label,
            timestamp: sample_candles[0].timestamp,
            future_return,
        })
    }

    /// Create GAF frames for a specific feature
    fn create_gaf_frames(
        &self,
        candles: &[Candle],
        feature: GafFeature,
    ) -> anyhow::Result<Vec<Array2<f64>>> {
        let mut frames = Vec::with_capacity(self.config.num_frames);

        // Extract time series for this feature
        let values: Vec<f64> = candles
            .iter()
            .map(|c| self.extract_feature(c, feature))
            .collect();

        // Create frames by chunking the sequence
        let chunk_size = self.config.candles_per_frame;

        for chunk in values.chunks(chunk_size) {
            let gaf = self.encode_gaf(chunk)?;
            frames.push(gaf);
        }

        Ok(frames)
    }

    /// Extract feature value from candle
    fn extract_feature(&self, candle: &Candle, feature: GafFeature) -> f64 {
        match feature {
            GafFeature::Close => candle.close,
            GafFeature::Open => candle.open,
            GafFeature::High => candle.high,
            GafFeature::Low => candle.low,
            GafFeature::Volume => candle.volume,
            GafFeature::HighLow => candle.high - candle.low,
            GafFeature::CloseOpen => candle.close - candle.open,
        }
    }

    /// Encode time series as GAF image
    fn encode_gaf(&self, values: &[f64]) -> anyhow::Result<Array2<f64>> {
        if values.is_empty() {
            return Err(anyhow::anyhow!("Cannot encode empty sequence"));
        }

        // Normalize to [-1, 1]
        let normalized = self.normalize(values);

        // Resize to target image size
        let resized = self.resize(&normalized, self.config.gaf_image_size);

        // Generate GASF (Gramian Angular Summation Field)
        let gaf = self.gasf(&resized);

        Ok(gaf)
    }

    /// Normalize values to [-1, 1]
    fn normalize(&self, values: &[f64]) -> Vec<f64> {
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f64::EPSILON {
            return vec![0.0; values.len()];
        }

        values
            .iter()
            .map(|&v| 2.0 * (v - min_val) / (max_val - min_val) - 1.0)
            .collect()
    }

    /// Resize using linear interpolation
    fn resize(&self, values: &[f64], target_size: usize) -> Vec<f64> {
        if values.len() == target_size {
            return values.to_vec();
        }

        let mut resized = Vec::with_capacity(target_size);
        for i in 0..target_size {
            let pos = (i as f64) * ((values.len() - 1) as f64) / ((target_size - 1) as f64);
            let idx = pos as usize;
            let frac = pos - (idx as f64);

            if idx + 1 < values.len() {
                let interpolated = values[idx] * (1.0 - frac) + values[idx + 1] * frac;
                resized.push(interpolated);
            } else {
                resized.push(values[idx]);
            }
        }

        resized
    }

    /// Generate GASF matrix
    fn gasf(&self, normalized: &[f64]) -> Array2<f64> {
        let n = normalized.len();
        let mut gaf = Array2::<f64>::zeros((n, n));

        let phi: Vec<f64> = normalized
            .iter()
            .map(|&x| x.clamp(-1.0, 1.0).acos())
            .collect();

        for i in 0..n {
            for j in 0..n {
                gaf[(i, j)] = (phi[i] + phi[j]).cos();
            }
        }

        gaf
    }

    /// Transpose frames from [channels][frames][H][W] to [frames][H][W][channels]
    fn transpose_frames(
        &self,
        channel_frames: Vec<Vec<Array2<f64>>>,
    ) -> anyhow::Result<Vec<Vec<Vec<Vec<f32>>>>> {
        let num_channels = channel_frames.len();
        let num_frames = channel_frames[0].len();
        let height = channel_frames[0][0].nrows();
        let width = channel_frames[0][0].ncols();

        let mut transposed =
            vec![vec![vec![vec![0.0f32; num_channels]; width]; height]; num_frames];

        for (c_idx, channel) in channel_frames.iter().enumerate() {
            for (f_idx, frame) in channel.iter().enumerate() {
                for i in 0..height {
                    for j in 0..width {
                        transposed[f_idx][i][j][c_idx] = frame[(i, j)] as f32;
                    }
                }
            }
        }

        Ok(transposed)
    }

    /// Split data into train and validation sets
    fn split_train_val(&mut self) {
        let total = self.samples.len();
        let train_size = (total as f64 * self.config.train_split) as usize;

        let mut indices: Vec<usize> = (0..total).collect();

        if self.config.shuffle {
            use rand::SeedableRng;
            use rand::seq::SliceRandom;

            let mut rng: rand::rngs::StdRng = if let Some(seed) = self.config.seed {
                rand::rngs::StdRng::seed_from_u64(seed)
            } else {
                rand::make_rng()
            };

            indices.shuffle(&mut rng);
        }

        self.train_indices = indices[..train_size].to_vec();
        self.val_indices = indices[train_size..].to_vec();

        info!(
            "Split: {} training, {} validation",
            self.train_indices.len(),
            self.val_indices.len()
        );
    }

    /// Get next training batch
    pub async fn next_train_batch(
        &mut self,
        batch_size: usize,
        device: &Device,
    ) -> CandleResult<Option<TrainingBatch>> {
        self.next_batch_internal(batch_size, device, true)
    }

    /// Get next validation batch
    pub async fn next_val_batch(
        &mut self,
        batch_size: usize,
        device: &Device,
    ) -> CandleResult<Option<TrainingBatch>> {
        self.next_batch_internal(batch_size, device, false)
    }

    /// Internal batch creation
    fn next_batch_internal(
        &mut self,
        batch_size: usize,
        device: &Device,
        is_train: bool,
    ) -> CandleResult<Option<TrainingBatch>> {
        let indices = if is_train {
            &self.train_indices
        } else {
            &self.val_indices
        };

        if self.current_idx >= indices.len() {
            return Ok(None);
        }

        let end_idx = (self.current_idx + batch_size).min(indices.len());
        let batch_indices = &indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        // Gather samples
        let batch_samples: Vec<&TrainingSample> = batch_indices
            .iter()
            .map(|&idx| &self.samples[idx])
            .collect();

        // Get dimensions from first sample
        let first_sample = batch_samples[0];
        let num_frames = first_sample.frames.len();
        let height = first_sample.frames[0].len();
        let width = first_sample.frames[0][0].len();
        let channels = first_sample.frames[0][0][0].len();

        // Flatten samples into input tensor
        let mut input_data =
            Vec::with_capacity(actual_batch_size * num_frames * height * width * channels);

        let mut labels = Vec::with_capacity(actual_batch_size);
        let mut timestamps = Vec::with_capacity(actual_batch_size);

        for sample in batch_samples {
            timestamps.push(sample.timestamp);
            labels.push(sample.label as u32);

            for frame in &sample.frames {
                for row in frame {
                    for pixel in row {
                        for &channel_val in pixel {
                            input_data.push(channel_val);
                        }
                    }
                }
            }
        }

        // Create tensors
        let inputs = Tensor::from_vec(
            input_data,
            (actual_batch_size, num_frames, height, width, channels),
            device,
        )?;

        let labels_tensor = Tensor::from_vec(labels, actual_batch_size, device)?;

        self.current_idx = end_idx;

        Ok(Some(TrainingBatch {
            inputs,
            labels: labels_tensor,
            timestamps,
        }))
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self, is_train: bool) {
        self.current_idx = 0;

        if self.config.shuffle && is_train {
            use rand::SeedableRng;
            use rand::seq::SliceRandom;

            let mut rng: rand::rngs::StdRng = if let Some(seed) = self.config.seed {
                rand::rngs::StdRng::seed_from_u64(seed + 1) // Different seed each epoch
            } else {
                rand::make_rng()
            };

            self.train_indices.shuffle(&mut rng);
        }
    }

    /// Get number of training samples
    pub fn num_train_samples(&self) -> usize {
        self.train_indices.len()
    }

    /// Get number of validation samples
    pub fn num_val_samples(&self) -> usize {
        self.val_indices.len()
    }

    /// Get label distribution
    pub fn label_distribution(&self) -> (usize, usize, usize) {
        let mut sell_count = 0;
        let mut hold_count = 0;
        let mut buy_count = 0;

        for sample in &self.samples {
            match sample.label {
                0 => sell_count += 1,
                1 => hold_count += 1,
                2 => buy_count += 1,
                _ => {}
            }
        }

        (sell_count, hold_count, buy_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: i as i64,
                open: 100.0 + i as f64,
                high: 105.0 + i as f64,
                low: 95.0 + i as f64,
                close: 102.0 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect()
    }

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = MarketDataPipeline::new(config);
        assert_eq!(pipeline.samples.len(), 0);
    }

    #[tokio::test]
    async fn test_load_and_preprocess() {
        let config = PipelineConfig {
            num_frames: 4,
            candles_per_frame: 10,
            prediction_horizon: 5,
            ..Default::default()
        };

        let mut pipeline = MarketDataPipeline::new(config);
        let candles = create_test_candles(100);
        pipeline.load_from_vec(candles);

        let result = pipeline.preprocess().await;
        assert!(result.is_ok());
        assert!(!pipeline.samples.is_empty());
    }

    #[test]
    fn test_normalize() {
        let config = PipelineConfig::default();
        let pipeline = MarketDataPipeline::new(config);

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = pipeline.normalize(&values);

        assert!(normalized[0] >= -1.0 && normalized[0] <= 1.0);
        assert!(*normalized.last().unwrap() >= -1.0 && *normalized.last().unwrap() <= 1.0);
    }

    #[test]
    fn test_gasf() {
        let config = PipelineConfig::default();
        let pipeline = MarketDataPipeline::new(config);

        let values = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let gaf = pipeline.gasf(&values);

        assert_eq!(gaf.shape(), &[5, 5]);
    }
}
