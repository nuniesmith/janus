//! Gradient-weighted Class Activation Mapping (Grad-CAM)
//!
//! Part of the Visual Cortex region - generates visual explanations for
//! model predictions by highlighting important regions in GAF images.

use crate::common::Result;
use std::collections::HashMap;

/// Grad-CAM configuration
#[derive(Debug, Clone)]
pub struct GradcamConfig {
    /// Target layer name for activation extraction
    pub target_layer: String,
    /// Whether to use guided backpropagation
    pub guided: bool,
    /// Colormap for visualization
    pub colormap: Colormap,
    /// Overlay alpha (0.0 - 1.0)
    pub overlay_alpha: f64,
    /// Threshold for significance (0.0 - 1.0)
    pub significance_threshold: f64,
    /// Whether to normalize output
    pub normalize: bool,
    /// Smoothing kernel size (0 = no smoothing)
    pub smoothing_kernel: usize,
}

impl Default for GradcamConfig {
    fn default() -> Self {
        Self {
            target_layer: "layer4".to_string(),
            guided: true,
            colormap: Colormap::Jet,
            overlay_alpha: 0.5,
            significance_threshold: 0.3,
            normalize: true,
            smoothing_kernel: 3,
        }
    }
}

/// Colormap options for heatmap visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    /// Jet colormap (blue -> green -> red)
    Jet,
    /// Viridis colormap (purple -> green -> yellow)
    Viridis,
    /// Hot colormap (black -> red -> yellow -> white)
    Hot,
    /// Cool colormap (cyan -> magenta)
    Cool,
    /// Grayscale
    Gray,
    /// Red-blue diverging
    RdBu,
}

impl Colormap {
    /// Convert scalar value (0-1) to RGB color
    pub fn to_rgb(&self, value: f64) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);

        match self {
            Self::Jet => {
                let r = (255.0 * Self::jet_r(v)) as u8;
                let g = (255.0 * Self::jet_g(v)) as u8;
                let b = (255.0 * Self::jet_b(v)) as u8;
                (r, g, b)
            }
            Self::Viridis => {
                // Simplified viridis approximation
                let r = (255.0 * (0.267 + 0.329 * v + 0.191 * v * v)) as u8;
                let g = (255.0 * (0.004 + 0.873 * v - 0.107 * v * v)) as u8;
                let b = (255.0 * (0.329 + 0.281 * v - 0.574 * v * v).max(0.0)) as u8;
                (r, g, b)
            }
            Self::Hot => {
                let r = (255.0 * (v * 3.0).min(1.0)) as u8;
                let g = (255.0 * ((v - 0.333) * 3.0).clamp(0.0, 1.0)) as u8;
                let b = (255.0 * ((v - 0.666) * 3.0).clamp(0.0, 1.0)) as u8;
                (r, g, b)
            }
            Self::Cool => {
                let r = (255.0 * v) as u8;
                let g = (255.0 * (1.0 - v)) as u8;
                let b = 255;
                (r, g, b)
            }
            Self::Gray => {
                let gray = (255.0 * v) as u8;
                (gray, gray, gray)
            }
            Self::RdBu => {
                if v < 0.5 {
                    let t = v * 2.0;
                    let r = (255.0 * t) as u8;
                    let b = 255;
                    (r, 128, b)
                } else {
                    let t = (v - 0.5) * 2.0;
                    let r = 255;
                    let b = (255.0 * (1.0 - t)) as u8;
                    (r, 128, b)
                }
            }
        }
    }

    fn jet_r(v: f64) -> f64 {
        if v < 0.35 {
            0.0
        } else if v < 0.66 {
            (v - 0.35) / 0.31
        } else if v < 0.89 {
            1.0
        } else {
            1.0 - (v - 0.89) / 0.11 * 0.5
        }
    }

    fn jet_g(v: f64) -> f64 {
        if v < 0.125 {
            0.0
        } else if v < 0.375 {
            (v - 0.125) / 0.25
        } else if v < 0.64 {
            1.0
        } else if v < 0.91 {
            1.0 - (v - 0.64) / 0.27
        } else {
            0.0
        }
    }

    fn jet_b(v: f64) -> f64 {
        if v < 0.11 {
            0.5 + v / 0.11 * 0.5
        } else if v < 0.34 {
            1.0
        } else if v < 0.65 {
            1.0 - (v - 0.34) / 0.31
        } else {
            0.0
        }
    }
}

/// Layer activations captured during forward pass
#[derive(Debug, Clone)]
pub struct LayerActivations {
    /// Layer name
    pub layer_name: String,
    /// Activation values (channels x height x width)
    pub activations: Vec<Vec<Vec<f64>>>,
    /// Shape (channels, height, width)
    pub shape: (usize, usize, usize),
}

impl LayerActivations {
    /// Create new layer activations
    pub fn new(layer_name: &str, channels: usize, height: usize, width: usize) -> Self {
        Self {
            layer_name: layer_name.to_string(),
            activations: vec![vec![vec![0.0; width]; height]; channels],
            shape: (channels, height, width),
        }
    }

    /// Set activation value
    pub fn set(&mut self, channel: usize, row: usize, col: usize, value: f64) {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.activations[channel][row][col] = value;
        }
    }

    /// Get activation value
    pub fn get(&self, channel: usize, row: usize, col: usize) -> f64 {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.activations[channel][row][col]
        } else {
            0.0
        }
    }

    /// Get channel mean
    pub fn channel_mean(&self, channel: usize) -> f64 {
        if channel >= self.shape.0 {
            return 0.0;
        }

        let mut sum = 0.0;
        for row in &self.activations[channel] {
            for &val in row {
                sum += val;
            }
        }
        sum / (self.shape.1 * self.shape.2) as f64
    }
}

/// Gradients captured during backward pass
#[derive(Debug, Clone)]
pub struct LayerGradients {
    /// Layer name
    pub layer_name: String,
    /// Gradient values (channels x height x width)
    pub gradients: Vec<Vec<Vec<f64>>>,
    /// Shape (channels, height, width)
    pub shape: (usize, usize, usize),
}

impl LayerGradients {
    /// Create new layer gradients
    pub fn new(layer_name: &str, channels: usize, height: usize, width: usize) -> Self {
        Self {
            layer_name: layer_name.to_string(),
            gradients: vec![vec![vec![0.0; width]; height]; channels],
            shape: (channels, height, width),
        }
    }

    /// Set gradient value
    pub fn set(&mut self, channel: usize, row: usize, col: usize, value: f64) {
        if channel < self.shape.0 && row < self.shape.1 && col < self.shape.2 {
            self.gradients[channel][row][col] = value;
        }
    }

    /// Compute global average pooled gradient weights per channel
    pub fn compute_weights(&self) -> Vec<f64> {
        (0..self.shape.0)
            .map(|c| {
                let mut sum = 0.0;
                for row in &self.gradients[c] {
                    for &val in row {
                        sum += val;
                    }
                }
                sum / (self.shape.1 * self.shape.2) as f64
            })
            .collect()
    }
}

/// Grad-CAM heatmap result
#[derive(Debug, Clone)]
pub struct GradcamHeatmap {
    /// Heatmap values (height x width), normalized 0-1
    pub heatmap: Vec<Vec<f64>>,
    /// Shape (height, width)
    pub shape: (usize, usize),
    /// Target class/prediction this heatmap explains
    pub target_class: Option<String>,
    /// Predicted confidence
    pub confidence: f64,
    /// Significant regions (bounding boxes)
    pub significant_regions: Vec<SignificantRegion>,
    /// Layer used for generation
    pub source_layer: String,
    /// Whether guided backprop was used
    pub guided: bool,
}

impl GradcamHeatmap {
    /// Get heatmap value at position
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.shape.0 && col < self.shape.1 {
            self.heatmap[row][col]
        } else {
            0.0
        }
    }

    /// Get maximum activation value
    pub fn max_activation(&self) -> f64 {
        self.heatmap
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(0.0, f64::max)
    }

    /// Get mean activation value
    pub fn mean_activation(&self) -> f64 {
        let total: f64 = self.heatmap.iter().flat_map(|row| row.iter()).sum();
        total / (self.shape.0 * self.shape.1) as f64
    }

    /// Convert to RGB image with colormap
    pub fn to_rgb(&self, colormap: Colormap) -> Vec<Vec<(u8, u8, u8)>> {
        self.heatmap
            .iter()
            .map(|row| row.iter().map(|&v| colormap.to_rgb(v)).collect())
            .collect()
    }

    /// Resize heatmap to target dimensions using bilinear interpolation
    pub fn resize(&self, target_height: usize, target_width: usize) -> GradcamHeatmap {
        let mut resized = vec![vec![0.0; target_width]; target_height];

        let h_scale = self.shape.0 as f64 / target_height as f64;
        let w_scale = self.shape.1 as f64 / target_width as f64;

        for i in 0..target_height {
            for j in 0..target_width {
                let src_y = i as f64 * h_scale;
                let src_x = j as f64 * w_scale;

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(self.shape.0 - 1);
                let x1 = (x0 + 1).min(self.shape.1 - 1);

                let y_frac = src_y - y0 as f64;
                let x_frac = src_x - x0 as f64;

                // Bilinear interpolation
                let v00 = self.get(y0, x0);
                let v01 = self.get(y0, x1);
                let v10 = self.get(y1, x0);
                let v11 = self.get(y1, x1);

                let v = v00 * (1.0 - y_frac) * (1.0 - x_frac)
                    + v01 * (1.0 - y_frac) * x_frac
                    + v10 * y_frac * (1.0 - x_frac)
                    + v11 * y_frac * x_frac;

                resized[i][j] = v;
            }
        }

        GradcamHeatmap {
            heatmap: resized,
            shape: (target_height, target_width),
            target_class: self.target_class.clone(),
            confidence: self.confidence,
            significant_regions: Vec::new(), // Regions need recalculation
            source_layer: self.source_layer.clone(),
            guided: self.guided,
        }
    }
}

/// Significant region in heatmap
#[derive(Debug, Clone)]
pub struct SignificantRegion {
    /// Top-left corner (row, col)
    pub top_left: (usize, usize),
    /// Bottom-right corner (row, col)
    pub bottom_right: (usize, usize),
    /// Mean activation in region
    pub mean_activation: f64,
    /// Max activation in region
    pub max_activation: f64,
    /// Region label/interpretation
    pub label: Option<String>,
}

impl SignificantRegion {
    /// Get region area
    pub fn area(&self) -> usize {
        let height = self.bottom_right.0.saturating_sub(self.top_left.0);
        let width = self.bottom_right.1.saturating_sub(self.top_left.1);
        height * width
    }

    /// Check if point is inside region
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.top_left.0
            && row <= self.bottom_right.0
            && col >= self.top_left.1
            && col <= self.bottom_right.1
    }
}

/// Grad-CAM statistics
#[derive(Debug, Clone, Default)]
pub struct GradcamStats {
    /// Total heatmaps generated
    pub heatmaps_generated: u64,
    /// Average generation time in microseconds
    pub avg_generation_time_us: f64,
    /// Average significant region count
    pub avg_significant_regions: f64,
    /// Cache hits
    pub cache_hits: u64,
    /// Heatmaps by target layer
    pub by_layer: HashMap<String, u64>,
}

/// Gradient-weighted Class Activation Mapping
pub struct Gradcam {
    /// Configuration
    config: GradcamConfig,
    /// Cached activations by layer
    activation_cache: HashMap<String, LayerActivations>,
    /// Cached gradients by layer
    gradient_cache: HashMap<String, LayerGradients>,
    /// Statistics
    stats: GradcamStats,
}

impl Default for Gradcam {
    fn default() -> Self {
        Self::new()
    }
}

impl Gradcam {
    /// Create a new Grad-CAM instance
    pub fn new() -> Self {
        Self::with_config(GradcamConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GradcamConfig) -> Self {
        Self {
            config,
            activation_cache: HashMap::new(),
            gradient_cache: HashMap::new(),
            stats: GradcamStats::default(),
        }
    }

    /// Set target layer
    pub fn with_target_layer(mut self, layer: impl Into<String>) -> Self {
        self.config.target_layer = layer.into();
        self
    }

    /// Enable/disable guided backpropagation
    pub fn with_guided(mut self, guided: bool) -> Self {
        self.config.guided = guided;
        self
    }

    /// Set colormap
    pub fn with_colormap(mut self, colormap: Colormap) -> Self {
        self.config.colormap = colormap;
        self
    }

    /// Register layer activations (called during forward pass)
    pub fn register_activations(&mut self, activations: LayerActivations) {
        self.activation_cache
            .insert(activations.layer_name.clone(), activations);
    }

    /// Register layer gradients (called during backward pass)
    pub fn register_gradients(&mut self, gradients: LayerGradients) {
        self.gradient_cache
            .insert(gradients.layer_name.clone(), gradients);
    }

    /// Generate Grad-CAM heatmap for target layer
    pub fn generate_heatmap(&mut self, target_class: Option<&str>) -> Result<GradcamHeatmap> {
        let start = std::time::Instant::now();

        let layer = &self.config.target_layer;

        // Get activations and gradients
        let activations = self
            .activation_cache
            .get(layer)
            .ok_or_else(|| anyhow::anyhow!("No activations found for layer: {}", layer))?;

        let gradients = self
            .gradient_cache
            .get(layer)
            .ok_or_else(|| anyhow::anyhow!("No gradients found for layer: {}", layer))?;

        // Compute channel weights using global average pooling of gradients
        let weights = gradients.compute_weights();

        // Compute weighted combination of activation maps
        let (height, width) = (activations.shape.1, activations.shape.2);
        let mut heatmap = vec![vec![0.0; width]; height];

        for (c, &weight) in weights.iter().enumerate() {
            for i in 0..height {
                for j in 0..width {
                    heatmap[i][j] += weight * activations.get(c, i, j);
                }
            }
        }

        // Apply ReLU (keep only positive values)
        for row in &mut heatmap {
            for val in row {
                *val = val.max(0.0);
            }
        }

        // Normalize to 0-1 range
        if self.config.normalize {
            let max_val = heatmap
                .iter()
                .flat_map(|r| r.iter())
                .copied()
                .fold(0.0, f64::max);

            if max_val > 0.0 {
                for row in &mut heatmap {
                    for val in row {
                        *val /= max_val;
                    }
                }
            }
        }

        // Apply smoothing if configured
        if self.config.smoothing_kernel > 0 {
            heatmap = self.apply_gaussian_smoothing(&heatmap, self.config.smoothing_kernel);
        }

        // Find significant regions
        let significant_regions =
            self.find_significant_regions(&heatmap, self.config.significance_threshold);

        // Update statistics
        let elapsed = start.elapsed().as_micros() as f64;
        self.stats.heatmaps_generated += 1;
        self.stats.avg_generation_time_us = self.stats.avg_generation_time_us * 0.9 + elapsed * 0.1;
        self.stats.avg_significant_regions =
            self.stats.avg_significant_regions * 0.9 + significant_regions.len() as f64 * 0.1;
        *self.stats.by_layer.entry(layer.clone()).or_insert(0) += 1;

        Ok(GradcamHeatmap {
            heatmap,
            shape: (height, width),
            target_class: target_class.map(String::from),
            confidence: 0.0, // Would be set from model output
            significant_regions,
            source_layer: layer.clone(),
            guided: self.config.guided,
        })
    }

    /// Generate heatmap from raw activation and gradient data
    pub fn generate_from_raw(
        &mut self,
        activations: &[Vec<Vec<f64>>],
        gradients: &[Vec<Vec<f64>>],
        target_class: Option<&str>,
    ) -> Result<GradcamHeatmap> {
        if activations.is_empty() || gradients.is_empty() {
            return Err(anyhow::anyhow!("Empty activations or gradients").into());
        }

        let channels = activations.len();
        let height = activations[0].len();
        let width = if height > 0 {
            activations[0][0].len()
        } else {
            0
        };

        // Compute weights from gradients
        let weights: Vec<f64> = gradients
            .iter()
            .map(|channel| {
                let sum: f64 = channel.iter().flat_map(|row| row.iter()).sum();
                sum / (height * width) as f64
            })
            .collect();

        // Weighted combination
        let mut heatmap = vec![vec![0.0; width]; height];

        for (c, &weight) in weights.iter().enumerate() {
            for i in 0..height {
                for j in 0..width {
                    if c < channels && i < activations[c].len() && j < activations[c][i].len() {
                        heatmap[i][j] += weight * activations[c][i][j];
                    }
                }
            }
        }

        // ReLU
        for row in &mut heatmap {
            for val in row {
                *val = val.max(0.0);
            }
        }

        // Normalize
        let max_val = heatmap
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(0.0, f64::max);

        if max_val > 0.0 {
            for row in &mut heatmap {
                for val in row {
                    *val /= max_val;
                }
            }
        }

        let significant_regions =
            self.find_significant_regions(&heatmap, self.config.significance_threshold);

        Ok(GradcamHeatmap {
            heatmap,
            shape: (height, width),
            target_class: target_class.map(String::from),
            confidence: 0.0,
            significant_regions,
            source_layer: "raw".to_string(),
            guided: self.config.guided,
        })
    }

    /// Apply Gaussian smoothing to heatmap
    fn apply_gaussian_smoothing(&self, heatmap: &[Vec<f64>], kernel_size: usize) -> Vec<Vec<f64>> {
        let height = heatmap.len();
        let width = if height > 0 { heatmap[0].len() } else { 0 };
        let mut result = vec![vec![0.0; width]; height];

        // Simple box filter as Gaussian approximation
        let radius = kernel_size / 2;

        for i in 0..height {
            for j in 0..width {
                let mut sum = 0.0;
                let mut count = 0;

                for di in 0..=radius * 2 {
                    for dj in 0..=radius * 2 {
                        let ni = (i + di).saturating_sub(radius);
                        let nj = (j + dj).saturating_sub(radius);

                        if ni < height && nj < width {
                            sum += heatmap[ni][nj];
                            count += 1;
                        }
                    }
                }

                result[i][j] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        result
    }

    /// Find significant regions above threshold
    fn find_significant_regions(
        &self,
        heatmap: &[Vec<f64>],
        threshold: f64,
    ) -> Vec<SignificantRegion> {
        let height = heatmap.len();
        let width = if height > 0 { heatmap[0].len() } else { 0 };

        let mut visited = vec![vec![false; width]; height];
        let mut regions = Vec::new();

        // Simple connected component analysis
        for i in 0..height {
            for j in 0..width {
                if !visited[i][j] && heatmap[i][j] >= threshold {
                    // BFS to find connected region
                    let mut min_row = i;
                    let mut max_row = i;
                    let mut min_col = j;
                    let mut max_col = j;
                    let mut sum = 0.0;
                    let mut max_val = 0.0;
                    let mut count = 0;

                    let mut queue = vec![(i, j)];
                    visited[i][j] = true;

                    while let Some((ci, cj)) = queue.pop() {
                        min_row = min_row.min(ci);
                        max_row = max_row.max(ci);
                        min_col = min_col.min(cj);
                        max_col = max_col.max(cj);
                        sum += heatmap[ci][cj];
                        max_val = max_val.max(heatmap[ci][cj]);
                        count += 1;

                        // Check 4-connected neighbors
                        for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                            let ni = (ci as i32 + di) as usize;
                            let nj = (cj as i32 + dj) as usize;

                            if ni < height
                                && nj < width
                                && !visited[ni][nj]
                                && heatmap[ni][nj] >= threshold
                            {
                                visited[ni][nj] = true;
                                queue.push((ni, nj));
                            }
                        }
                    }

                    // Only include regions with significant size
                    if count >= 4 {
                        regions.push(SignificantRegion {
                            top_left: (min_row, min_col),
                            bottom_right: (max_row, max_col),
                            mean_activation: sum / count as f64,
                            max_activation: max_val,
                            label: None,
                        });
                    }
                }
            }
        }

        // Sort by mean activation (most significant first)
        regions.sort_by(|a, b| {
            b.mean_activation
                .partial_cmp(&a.mean_activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        regions
    }

    /// Clear caches
    pub fn clear_cache(&mut self) {
        self.activation_cache.clear();
        self.gradient_cache.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &GradcamStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &GradcamConfig {
        &self.config
    }

    /// Main processing function (for compatibility)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradcam_creation() {
        let gradcam = Gradcam::new();
        assert_eq!(gradcam.config.target_layer, "layer4");
    }

    #[test]
    fn test_colormap_jet() {
        let (r, g, b) = Colormap::Jet.to_rgb(0.0);
        assert!(b > r); // Blue at low values

        let (r, g, b) = Colormap::Jet.to_rgb(1.0);
        assert!(r > b); // Red at high values
    }

    #[test]
    fn test_colormap_gray() {
        let (r, g, b) = Colormap::Gray.to_rgb(0.5);
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn test_layer_activations() {
        let mut activations = LayerActivations::new("conv1", 3, 4, 4);
        activations.set(0, 1, 1, 0.5);

        assert!((activations.get(0, 1, 1) - 0.5).abs() < 0.001);
        assert_eq!(activations.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_gradient_weights() {
        let mut gradients = LayerGradients::new("conv1", 2, 2, 2);

        // Set uniform gradients
        for c in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    gradients.set(c, i, j, 1.0);
                }
            }
        }

        let weights = gradients.compute_weights();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_heatmap_resize() {
        let heatmap = GradcamHeatmap {
            heatmap: vec![vec![0.0, 1.0], vec![1.0, 0.5]],
            shape: (2, 2),
            target_class: None,
            confidence: 0.9,
            significant_regions: Vec::new(),
            source_layer: "test".to_string(),
            guided: false,
        };

        let resized = heatmap.resize(4, 4);
        assert_eq!(resized.shape, (4, 4));
    }

    #[test]
    fn test_heatmap_to_rgb() {
        let heatmap = GradcamHeatmap {
            heatmap: vec![vec![0.0, 0.5, 1.0]],
            shape: (1, 3),
            target_class: None,
            confidence: 0.9,
            significant_regions: Vec::new(),
            source_layer: "test".to_string(),
            guided: false,
        };

        let rgb = heatmap.to_rgb(Colormap::Gray);
        assert_eq!(rgb.len(), 1);
        assert_eq!(rgb[0].len(), 3);
    }

    #[test]
    fn test_significant_region() {
        let region = SignificantRegion {
            top_left: (0, 0),
            bottom_right: (10, 10),
            mean_activation: 0.8,
            max_activation: 1.0,
            label: None,
        };

        assert_eq!(region.area(), 100);
        assert!(region.contains(5, 5));
        assert!(!region.contains(15, 15));
    }

    #[test]
    fn test_generate_from_raw() {
        let mut gradcam = Gradcam::new();

        let activations = vec![
            vec![vec![1.0, 0.5], vec![0.5, 0.0]],
            vec![vec![0.0, 0.5], vec![0.5, 1.0]],
        ];

        let gradients = vec![
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        ];

        let result = gradcam
            .generate_from_raw(&activations, &gradients, Some("test_class"))
            .unwrap();

        assert_eq!(result.shape, (2, 2));
        assert!(result.max_activation() <= 1.0);
    }

    #[test]
    fn test_register_and_generate() {
        let mut gradcam = Gradcam::new().with_target_layer("conv1");

        // Create and register activations
        let mut activations = LayerActivations::new("conv1", 2, 4, 4);
        for c in 0..2 {
            for i in 0..4 {
                for j in 0..4 {
                    activations.set(c, i, j, ((i + j) as f64) / 8.0);
                }
            }
        }
        gradcam.register_activations(activations);

        // Create and register gradients
        let mut gradients = LayerGradients::new("conv1", 2, 4, 4);
        for c in 0..2 {
            for i in 0..4 {
                for j in 0..4 {
                    gradients.set(c, i, j, 0.5);
                }
            }
        }
        gradcam.register_gradients(gradients);

        let result = gradcam.generate_heatmap(None).unwrap();
        assert_eq!(result.shape, (4, 4));
    }

    #[test]
    fn test_statistics() {
        let mut gradcam = Gradcam::new();

        let activations = vec![vec![vec![1.0; 2]; 2]];
        let gradients = vec![vec![vec![1.0; 2]; 2]];

        for _ in 0..5 {
            gradcam
                .generate_from_raw(&activations, &gradients, None)
                .unwrap();
        }

        assert_eq!(gradcam.stats().heatmaps_generated, 5);
    }

    #[test]
    fn test_process_compatibility() {
        let gradcam = Gradcam::new();
        assert!(gradcam.process().is_ok());
    }
}
