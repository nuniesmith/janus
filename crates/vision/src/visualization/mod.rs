//! GAF Visualization Tools
//!
//! This module provides visualization utilities for Gramian Angular Fields:
//! - Heatmap generation for GASF/GADF matrices
//! - ASCII art visualization for terminal display
//! - Export to various formats (PPM, CSV)
//! - Color mapping utilities
//!
//! # Example
//!
//! ```ignore
//! use janus_vision::visualization::{GafVisualizer, ColorMap};
//!
//! let gaf_matrix: Vec<Vec<f32>> = compute_gasf(&prices);
//! let visualizer = GafVisualizer::new(ColorMap::Viridis);
//!
//! // Save as PPM image
//! visualizer.save_ppm(&gaf_matrix, "gaf_output.ppm")?;
//!
//! // Print ASCII representation
//! visualizer.print_ascii(&gaf_matrix);
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Color map for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Viridis - perceptually uniform, colorblind-friendly
    Viridis,
    /// Plasma - perceptually uniform, warm colors
    Plasma,
    /// Inferno - perceptually uniform, dark to bright
    Inferno,
    /// Grayscale - simple black to white
    Grayscale,
    /// RedBlue - diverging colormap centered at zero
    RedBlue,
    /// Hot - black to red to yellow to white
    Hot,
}

/// RGB color value
#[derive(Debug, Clone, Copy)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn black() -> Self {
        Self::new(0, 0, 0)
    }

    pub fn white() -> Self {
        Self::new(255, 255, 255)
    }
}

/// GAF visualizer with configurable color mapping
pub struct GafVisualizer {
    /// Color map to use
    color_map: ColorMap,
    /// Minimum value for normalization (if None, computed from data)
    min_val: Option<f32>,
    /// Maximum value for normalization (if None, computed from data)
    max_val: Option<f32>,
}

impl Default for GafVisualizer {
    fn default() -> Self {
        Self::new(ColorMap::Viridis)
    }
}

impl GafVisualizer {
    /// Create a new visualizer with specified color map
    pub fn new(color_map: ColorMap) -> Self {
        Self {
            color_map,
            min_val: None,
            max_val: None,
        }
    }

    /// Set fixed value range for normalization
    pub fn with_range(mut self, min_val: f32, max_val: f32) -> Self {
        self.min_val = Some(min_val);
        self.max_val = Some(max_val);
        self
    }

    /// Get color for a normalized value [0, 1]
    fn get_color(&self, value: f32) -> Rgb {
        let t = value.clamp(0.0, 1.0);

        match self.color_map {
            ColorMap::Viridis => self.viridis(t),
            ColorMap::Plasma => self.plasma(t),
            ColorMap::Inferno => self.inferno(t),
            ColorMap::Grayscale => self.grayscale(t),
            ColorMap::RedBlue => self.red_blue(t),
            ColorMap::Hot => self.hot(t),
        }
    }

    /// Viridis colormap
    fn viridis(&self, t: f32) -> Rgb {
        // Simplified viridis approximation
        let r = (0.267004 + t * (0.329415 + t * (-0.601653 + t * 0.543608))) * 255.0;
        let g = (0.004874 + t * (0.873449 + t * (0.473299 - t * 0.351726))) * 255.0;
        let b = (0.329415 + t * (0.283327 + t * (-0.159474 - t * 0.452912))) * 255.0;

        Rgb::new(
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        )
    }

    /// Plasma colormap
    fn plasma(&self, t: f32) -> Rgb {
        let r = (0.050383 + t * (2.028397 + t * (-0.772906 - t * 0.256425))) * 255.0;
        let g = (0.029803 + t * (0.319898 + t * (0.944297 - t * 0.293825))) * 255.0;
        let b = (0.527975 + t * (0.278203 + t * (-1.344382 + t * 0.537604))) * 255.0;

        Rgb::new(
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        )
    }

    /// Inferno colormap
    fn inferno(&self, t: f32) -> Rgb {
        let r = (0.001462 + t * (1.291495 + t * (0.584836 - t * 0.877409))) * 255.0;
        let g = (0.000466 + t * (0.174918 + t * (1.250592 - t * 0.425866))) * 255.0;
        let b = (0.013866 + t * (0.568724 + t * (-0.749688 + t * 0.166979))) * 255.0;

        Rgb::new(
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        )
    }

    /// Grayscale colormap
    fn grayscale(&self, t: f32) -> Rgb {
        let v = (t * 255.0) as u8;
        Rgb::new(v, v, v)
    }

    /// Red-Blue diverging colormap
    fn red_blue(&self, t: f32) -> Rgb {
        if t < 0.5 {
            // Blue to white
            let s = t * 2.0;
            let r = (s * 255.0) as u8;
            let g = (s * 255.0) as u8;
            let b = 255;
            Rgb::new(r, g, b)
        } else {
            // White to red
            let s = (t - 0.5) * 2.0;
            let r = 255;
            let g = ((1.0 - s) * 255.0) as u8;
            let b = ((1.0 - s) * 255.0) as u8;
            Rgb::new(r, g, b)
        }
    }

    /// Hot colormap (black -> red -> yellow -> white)
    fn hot(&self, t: f32) -> Rgb {
        let r = (t * 3.0).min(1.0) * 255.0;
        let g = ((t - 0.33).max(0.0) * 3.0).min(1.0) * 255.0;
        let b = ((t - 0.67).max(0.0) * 3.0).min(1.0) * 255.0;

        Rgb::new(r as u8, g as u8, b as u8)
    }

    /// Normalize a value to [0, 1] range
    fn normalize(&self, value: f32, data_min: f32, data_max: f32) -> f32 {
        let min_v = self.min_val.unwrap_or(data_min);
        let max_v = self.max_val.unwrap_or(data_max);

        if (max_v - min_v).abs() < 1e-8 {
            0.5
        } else {
            (value - min_v) / (max_v - min_v)
        }
    }

    /// Get data range from matrix
    fn get_data_range(&self, matrix: &[Vec<f32>]) -> (f32, f32) {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for row in matrix {
            for &val in row {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
        }

        (min_val, max_val)
    }

    /// Save GAF matrix as PPM image (portable pixmap format)
    ///
    /// PPM is a simple uncompressed format that can be opened by most image viewers
    pub fn save_ppm(&self, matrix: &[Vec<f32>], path: impl AsRef<Path>) -> std::io::Result<()> {
        let height = matrix.len();
        if height == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Empty matrix",
            ));
        }
        let width = matrix[0].len();

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // PPM header
        writeln!(writer, "P6")?;
        writeln!(writer, "{} {}", width, height)?;
        writeln!(writer, "255")?;

        let (data_min, data_max) = self.get_data_range(matrix);

        // Write pixels
        for row in matrix {
            for &val in row {
                let normalized = self.normalize(val, data_min, data_max);
                let color = self.get_color(normalized);
                writer.write_all(&[color.r, color.g, color.b])?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Save GAF matrix as CSV file
    pub fn save_csv(&self, matrix: &[Vec<f32>], path: impl AsRef<Path>) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        for row in matrix {
            let line: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(writer, "{}", line.join(","))?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Print ASCII art representation of the GAF matrix
    pub fn print_ascii(&self, matrix: &[Vec<f32>]) {
        let ascii_chars = [' ', '░', '▒', '▓', '█'];
        let (data_min, data_max) = self.get_data_range(matrix);

        println!(
            "GAF Matrix ({}x{})",
            matrix.len(),
            matrix.first().map_or(0, |r| r.len())
        );
        println!("{}", "─".repeat(matrix.first().map_or(0, |r| r.len()) + 2));

        for row in matrix {
            print!("│");
            for &val in row {
                let normalized = self.normalize(val, data_min, data_max);
                let idx = ((normalized * (ascii_chars.len() - 1) as f32).round() as usize)
                    .min(ascii_chars.len() - 1);
                print!("{}", ascii_chars[idx]);
            }
            println!("│");
        }

        println!("{}", "─".repeat(matrix.first().map_or(0, |r| r.len()) + 2));
        println!("Range: [{:.4}, {:.4}]", data_min, data_max);
    }

    /// Generate a compact ASCII representation as a string
    pub fn to_ascii_string(&self, matrix: &[Vec<f32>]) -> String {
        let ascii_chars = [' ', '.', ':', '+', '#', '@'];
        let (data_min, data_max) = self.get_data_range(matrix);

        let mut result = String::new();

        for row in matrix {
            for &val in row {
                let normalized = self.normalize(val, data_min, data_max);
                let idx = ((normalized * (ascii_chars.len() - 1) as f32).round() as usize)
                    .min(ascii_chars.len() - 1);
                result.push(ascii_chars[idx]);
            }
            result.push('\n');
        }

        result
    }

    /// Generate HTML visualization with inline CSS
    pub fn to_html(&self, matrix: &[Vec<f32>], title: &str) -> String {
        let (data_min, data_max) = self.get_data_range(matrix);
        let cell_size = 4; // pixels per cell

        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", title));
        html.push_str("<style>\n");
        html.push_str(".gaf-container { display: inline-block; }\n");
        html.push_str(".gaf-row { display: flex; height: ");
        html.push_str(&format!("{}px; }}\n", cell_size));
        html.push_str(".gaf-cell { width: ");
        html.push_str(&format!("{}px; height: {}px; }}\n", cell_size, cell_size));
        html.push_str("</style>\n</head>\n<body>\n");
        html.push_str(&format!("<h2>{}</h2>\n", title));
        html.push_str("<div class=\"gaf-container\">\n");

        for row in matrix {
            html.push_str("<div class=\"gaf-row\">\n");
            for &val in row {
                let normalized = self.normalize(val, data_min, data_max);
                let color = self.get_color(normalized);
                html.push_str(&format!(
                    "<div class=\"gaf-cell\" style=\"background:rgb({},{},{})\"></div>\n",
                    color.r, color.g, color.b
                ));
            }
            html.push_str("</div>\n");
        }

        html.push_str("</div>\n");
        html.push_str(&format!(
            "<p>Size: {}x{}, Range: [{:.4}, {:.4}]</p>\n",
            matrix.len(),
            matrix.first().map_or(0, |r| r.len()),
            data_min,
            data_max
        ));
        html.push_str("</body>\n</html>");

        html
    }

    /// Save as HTML file
    pub fn save_html(
        &self,
        matrix: &[Vec<f32>],
        path: impl AsRef<Path>,
        title: &str,
    ) -> std::io::Result<()> {
        let html = self.to_html(matrix, title);
        std::fs::write(path, html)
    }

    /// Create a difference visualization between two GAF matrices
    pub fn visualize_difference(
        &self,
        matrix1: &[Vec<f32>],
        matrix2: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let height = matrix1.len().min(matrix2.len());
        let width = matrix1
            .first()
            .map_or(0, |r| r.len())
            .min(matrix2.first().map_or(0, |r| r.len()));

        let mut diff = vec![vec![0.0; width]; height];

        for i in 0..height {
            for j in 0..width {
                diff[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }

        diff
    }

    /// Get statistics about the GAF matrix
    pub fn get_statistics(&self, matrix: &[Vec<f32>]) -> GafStatistics {
        let (min_val, max_val) = self.get_data_range(matrix);

        let mut sum = 0.0;
        let mut count = 0usize;

        for row in matrix {
            for &val in row {
                if val.is_finite() {
                    sum += val;
                    count += 1;
                }
            }
        }

        let mean = if count > 0 { sum / count as f32 } else { 0.0 };

        // Compute variance
        let mut variance_sum = 0.0;
        for row in matrix {
            for &val in row {
                if val.is_finite() {
                    let diff = val - mean;
                    variance_sum += diff * diff;
                }
            }
        }
        let variance = if count > 0 {
            variance_sum / count as f32
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        GafStatistics {
            min: min_val,
            max: max_val,
            mean,
            std_dev,
            size: (matrix.len(), matrix.first().map_or(0, |r| r.len())),
        }
    }
}

/// Statistics for a GAF matrix
#[derive(Debug, Clone)]
pub struct GafStatistics {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Matrix size (rows, cols)
    pub size: (usize, usize),
}

impl std::fmt::Display for GafStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GAF Statistics:\n  Size: {}x{}\n  Range: [{:.4}, {:.4}]\n  Mean: {:.4}\n  Std Dev: {:.4}",
            self.size.0, self.size.1, self.min, self.max, self.mean, self.std_dev
        )
    }
}

/// Plot GAF as heatmap (convenience function)
#[cfg(feature = "viz")]
pub fn plot_gaf(matrix: &[Vec<f32>], path: &str) -> std::io::Result<()> {
    let visualizer = GafVisualizer::new(ColorMap::Viridis);
    visualizer.save_ppm(matrix, path)
}

/// Plot GAF with default visualizer
pub fn plot_gaf_default(matrix: &[Vec<f32>], path: &str) -> std::io::Result<()> {
    let visualizer = GafVisualizer::default();
    visualizer.save_ppm(matrix, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.25, 0.5],
            vec![0.25, 0.5, 0.75],
            vec![0.5, 0.75, 1.0],
        ]
    }

    #[test]
    fn test_visualizer_creation() {
        let viz = GafVisualizer::new(ColorMap::Viridis);
        assert_eq!(viz.color_map, ColorMap::Viridis);
    }

    #[test]
    fn test_color_mapping() {
        let viz = GafVisualizer::new(ColorMap::Grayscale);

        // Black for 0
        let black = viz.get_color(0.0);
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        // White for 1
        let white = viz.get_color(1.0);
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);
    }

    #[test]
    fn test_normalization() {
        let viz = GafVisualizer::new(ColorMap::Viridis);

        assert!((viz.normalize(0.0, 0.0, 1.0) - 0.0).abs() < 0.001);
        assert!((viz.normalize(0.5, 0.0, 1.0) - 0.5).abs() < 0.001);
        assert!((viz.normalize(1.0, 0.0, 1.0) - 1.0).abs() < 0.001);

        // Custom range
        assert!((viz.normalize(5.0, 0.0, 10.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ascii_generation() {
        let viz = GafVisualizer::new(ColorMap::Grayscale);
        let matrix = create_test_matrix();

        let ascii = viz.to_ascii_string(&matrix);
        assert!(!ascii.is_empty());
        assert!(ascii.contains('\n'));
    }

    #[test]
    fn test_html_generation() {
        let viz = GafVisualizer::new(ColorMap::Viridis);
        let matrix = create_test_matrix();

        let html = viz.to_html(&matrix, "Test GAF");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test GAF"));
        assert!(html.contains("rgb("));
    }

    #[test]
    fn test_statistics() {
        let viz = GafVisualizer::new(ColorMap::Grayscale);
        let matrix = create_test_matrix();

        let stats = viz.get_statistics(&matrix);
        assert_eq!(stats.size, (3, 3));
        assert!((stats.min - 0.0).abs() < 0.001);
        assert!((stats.max - 1.0).abs() < 0.001);
        assert!((stats.mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_difference_visualization() {
        let viz = GafVisualizer::new(ColorMap::RedBlue);

        let matrix1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix2 = vec![vec![0.5, 1.5], vec![2.5, 3.5]];

        let diff = viz.visualize_difference(&matrix1, &matrix2);

        assert_eq!(diff.len(), 2);
        assert!((diff[0][0] - 0.5).abs() < 0.001);
        assert!((diff[1][1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_data_range() {
        let viz = GafVisualizer::new(ColorMap::Grayscale);

        let matrix = vec![vec![-1.0, 0.0, 1.0], vec![2.0, 3.0, 4.0]];

        let (min, max) = viz.get_data_range(&matrix);
        assert!((min - (-1.0)).abs() < 0.001);
        assert!((max - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_range() {
        let viz = GafVisualizer::new(ColorMap::Grayscale).with_range(-1.0, 1.0);

        // Even though data is 0-1, normalize using -1 to 1 range
        let normalized = viz.normalize(0.0, 0.0, 1.0);
        assert!((normalized - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_all_colormaps() {
        let matrix = create_test_matrix();

        for colormap in [
            ColorMap::Viridis,
            ColorMap::Plasma,
            ColorMap::Inferno,
            ColorMap::Grayscale,
            ColorMap::RedBlue,
            ColorMap::Hot,
        ] {
            let viz = GafVisualizer::new(colormap);
            let stats = viz.get_statistics(&matrix);
            assert!(stats.min.is_finite());
            assert!(stats.max.is_finite());
        }
    }
}
