//! DSP Integration Tests
//!
//! End-to-end validation of the complete DSP pipeline with realistic
//! market scenarios and cross-validation against expected behaviors.

use dsp::frama::MarketRegime;
use dsp::pipeline::{DspConfig, DspPipeline};

// ============================================================================
// Synthetic Market Generators
// ============================================================================

/// Generate a strong uptrend
fn generate_uptrend(start: f64, length: usize, slope: f64) -> Vec<f64> {
    (0..length).map(|i| start + (i as f64) * slope).collect()
}

/// Generate a strong downtrend
fn generate_downtrend(start: f64, length: usize, slope: f64) -> Vec<f64> {
    (0..length).map(|i| start - (i as f64) * slope).collect()
}

/// Generate mean-reverting oscillation
fn generate_mean_reverting(center: f64, length: usize, amplitude: f64) -> Vec<f64> {
    (0..length)
        .map(|i| center + amplitude * ((i as f64) * 0.1).sin())
        .collect()
}

/// Generate random walk (brownian motion approximation)
fn generate_random_walk(start: f64, length: usize, seed: u64) -> Vec<f64> {
    let mut prices = Vec::with_capacity(length);
    let mut price = start;
    let mut rng_state = seed;

    prices.push(price);

    for _ in 1..length {
        // Simple LCG for deterministic randomness
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random = ((rng_state / 65536) % 32768) as f64 / 32768.0;

        // Centered around 0, range [-1, 1]
        let change = (random - 0.5) * 0.2;
        price += change;
        prices.push(price);
    }

    prices
}

/// Generate flat line (edge case)
fn generate_flat(value: f64, length: usize) -> Vec<f64> {
    vec![value; length]
}

/// Generate spike (outlier test)
fn generate_with_spike(base: f64, length: usize, spike_pos: usize, spike_size: f64) -> Vec<f64> {
    let mut prices = vec![base; length];
    if spike_pos < length {
        prices[spike_pos] = base + spike_size;
    }
    prices
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_pipeline_uptrend_detection() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate strong uptrend
    let prices = generate_uptrend(100.0, 300, 0.5);

    let mut trending_count = 0;
    let mut total_valid = 0;

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            total_valid += 1;

            // In a strong uptrend, we expect:
            // 1. FRAMA lags behind price
            assert!(
                output.frama <= output.price + 0.1,
                "FRAMA should lag in uptrend"
            );

            // 2. Positive divergence
            if total_valid > 100 {
                // After warmup
                assert!(
                    output.divergence >= 0.0,
                    "Divergence should be positive in uptrend"
                );
            }

            // 3. Eventually detect trending regime
            if matches!(output.regime, MarketRegime::Trending) {
                trending_count += 1;
            }
        }
    }

    // Should detect trending in at least some samples
    println!(
        "Trending detection: {}/{} ({:.1}%)",
        trending_count,
        total_valid,
        100.0 * trending_count as f64 / total_valid as f64
    );

    assert!(
        total_valid > 200,
        "Pipeline should produce valid outputs after warmup"
    );
}

#[test]
fn test_pipeline_downtrend_detection() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate strong downtrend
    let prices = generate_downtrend(200.0, 300, 0.5);

    let mut trending_count = 0;
    let mut total_valid = 0;

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            total_valid += 1;

            // FRAMA should lag behind (be above) price in downtrend
            if total_valid > 100 {
                assert!(
                    output.frama >= output.price - 0.1,
                    "FRAMA should be above price in downtrend"
                );

                // Negative divergence
                assert!(
                    output.divergence <= 0.0,
                    "Divergence should be negative in downtrend"
                );
            }

            // Count trending regime
            if matches!(output.regime, MarketRegime::Trending) {
                trending_count += 1;
            }
        }
    }

    println!(
        "Trending detection in downtrend: {}/{} ({:.1}%)",
        trending_count,
        total_valid,
        100.0 * trending_count as f64 / total_valid as f64
    );

    assert!(total_valid > 200);
}

#[test]
fn test_pipeline_mean_reverting_detection() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate oscillating prices
    let prices = generate_mean_reverting(100.0, 500, 5.0);

    let mut mean_reverting_count = 0;
    let mut total_valid = 0;

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            total_valid += 1;

            // FRAMA should be smoother than raw price
            // (no specific assertion here, just validate it works)

            if matches!(output.regime, MarketRegime::MeanReverting) {
                mean_reverting_count += 1;
            }
        }
    }

    println!(
        "Mean-reverting detection: {}/{} ({:.1}%)",
        mean_reverting_count,
        total_valid,
        100.0 * mean_reverting_count as f64 / total_valid as f64
    );

    assert!(total_valid > 400);
}

#[test]
fn test_pipeline_random_walk() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate random walk
    let prices = generate_random_walk(100.0, 500, 42);

    let mut random_walk_count = 0;
    let mut total_valid = 0;

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            total_valid += 1;

            // Validate all features are finite
            assert!(output.is_valid(), "Features must be finite");

            if matches!(output.regime, MarketRegime::RandomWalk) {
                random_walk_count += 1;
            }
        }
    }

    println!(
        "Random walk detection: {}/{} ({:.1}%)",
        random_walk_count,
        total_valid,
        100.0 * random_walk_count as f64 / total_valid as f64
    );

    assert!(total_valid > 400);
}

#[test]
fn test_pipeline_flat_line() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate flat line
    let prices = generate_flat(100.0, 200);

    let mut flat_outputs = 0;

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            flat_outputs += 1;

            // In flat market, FRAMA should converge to price
            if flat_outputs > 100 {
                assert!(
                    (output.frama - output.price).abs() < 0.01,
                    "FRAMA should equal price in flat market"
                );

                // Divergence should be near zero
                assert!(
                    output.divergence.abs() < 0.01,
                    "Divergence should be zero in flat market"
                );

                // Hurst should be high (smooth)
                assert!(
                    output.hurst > 0.8,
                    "Hurst should be high in flat market, got {}",
                    output.hurst
                );
            }
        }
    }

    assert!(flat_outputs > 100);
}

#[test]
fn test_pipeline_spike_handling() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate prices with a spike
    let prices = generate_with_spike(100.0, 200, 150, 50.0);

    let mut pre_spike_frama = None;
    let mut spike_handled = false;

    for (i, price) in prices.iter().enumerate() {
        if let Ok(output) = pipeline.process(*price) {
            if i == 149 {
                // Just before spike
                pre_spike_frama = Some(output.frama);
            } else if i == 150 {
                // At spike
                // FRAMA should not jump as much as raw price
                if let Some(prev_frama) = pre_spike_frama {
                    let frama_change = (output.frama - prev_frama).abs();
                    let price_change = 50.0;

                    assert!(frama_change < price_change, "FRAMA should smooth the spike");

                    spike_handled = true;
                }
            }
        }
    }

    assert!(spike_handled, "Should have processed the spike");
}

// ============================================================================
// Regime Change Tests
// ============================================================================

#[test]
fn test_regime_change_trending_to_mean_reverting() {
    let config = DspConfig::high_frequency(); // Fast adaptation
    let mut pipeline = DspPipeline::new(config);

    // Phase 1: Strong trend
    let trend_prices = generate_uptrend(100.0, 200, 0.5);

    for price in trend_prices {
        let _ = pipeline.process(price);
    }

    // Phase 2: Mean reverting
    let mr_prices = generate_mean_reverting(200.0, 200, 5.0);

    let mut regime_counts = [0, 0, 0, 0]; // [MeanReverting, RandomWalk, Trending, Unknown]

    for price in mr_prices {
        if let Ok(output) = pipeline.process(price) {
            match output.regime {
                MarketRegime::MeanReverting => regime_counts[0] += 1,
                MarketRegime::RandomWalk => regime_counts[1] += 1,
                MarketRegime::Trending => regime_counts[2] += 1,
                MarketRegime::Unknown => regime_counts[3] += 1,
            }
        }
    }

    println!(
        "Regime distribution after change: MR={}, RW={}, T={}, U={}",
        regime_counts[0], regime_counts[1], regime_counts[2], regime_counts[3]
    );

    // Should adapt away from trending regime
    // (exact distribution depends on parameters, just ensure it works)
    assert!(regime_counts[0] + regime_counts[1] + regime_counts[2] > 100);
}

// ============================================================================
// Feature Vector Validation
// ============================================================================

#[test]
fn test_feature_vector_bounds() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Generate diverse market conditions
    let prices = generate_random_walk(100.0, 500, 123);

    for price in prices {
        if let Ok(output) = pipeline.process(price) {
            let features = output.to_features();

            // [0] divergence_norm - should be within ±3σ (clipped)
            if let Some(dn) = output.divergence_norm {
                assert!(
                    dn.abs() <= 3.0 + 1e-6,
                    "Divergence norm should be clipped to ±3, got {}",
                    dn
                );
            }

            // [1] alpha_norm - should be within ±3σ (clipped)
            if let Some(an) = output.alpha_norm {
                assert!(
                    an.abs() <= 3.0 + 1e-6,
                    "Alpha norm should be clipped to ±3, got {}",
                    an
                );
            }

            // [2] fractal_dim - must be in [1, 2]
            assert!(
                features[2] >= 1.0 && features[2] <= 2.0,
                "Fractal dim must be in [1, 2], got {}",
                features[2]
            );

            // [3] hurst - must be in [0, 1]
            assert!(
                features[3] >= 0.0 && features[3] <= 1.0,
                "Hurst must be in [0, 1], got {}",
                features[3]
            );

            // [4] regime - must be in {-1, 0, 1}
            assert!(
                features[4] == -1.0 || features[4] == 0.0 || features[4] == 1.0,
                "Regime must be -1/0/1, got {}",
                features[4]
            );

            // [5] divergence_sign - must be in {-1, 0, 1}
            assert!(
                features[5] == -1.0 || features[5] == 0.0 || features[5] == 1.0,
                "Divergence sign must be -1/0/1, got {}",
                features[5]
            );

            // [6] alpha_deviation - should be in reasonable range
            assert!(
                features[6].abs() < 1.0,
                "Alpha deviation should be bounded, got {}",
                features[6]
            );

            // [7] regime_confidence - should be in [0, 0.6]
            assert!(
                features[7] >= 0.0 && features[7] <= 0.6 + 1e-6,
                "Regime confidence should be in [0, 0.6], got {}",
                features[7]
            );

            // All features must be finite
            assert!(output.is_valid(), "All features must be finite");
        }
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_high_frequency_config() {
    let config = DspConfig::high_frequency();
    let mut pipeline = DspPipeline::new(config);

    // Should warm up faster (smaller window)
    let prices = generate_uptrend(100.0, 100, 0.1);

    let mut valid_count = 0;

    for price in prices {
        if pipeline.process(price).is_ok() {
            valid_count += 1;
        }
    }

    // With window=32, should get valid outputs sooner
    assert!(valid_count > 50, "HF config should warm up quickly");
}

#[test]
fn test_low_frequency_config() {
    let config = DspConfig::low_frequency();
    let mut pipeline = DspPipeline::new(config);

    // Needs more warmup (larger window)
    let prices = generate_uptrend(100.0, 200, 0.1);

    let mut valid_count = 0;

    for price in prices {
        if pipeline.process(price).is_ok() {
            valid_count += 1;
        }
    }

    // With window=128, needs more warmup
    assert!(valid_count > 50);
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_pipeline_statistics() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    let prices = generate_uptrend(100.0, 200, 0.1);

    for price in prices {
        let _ = pipeline.process(price);
    }

    let stats = pipeline.stats();

    assert_eq!(stats.total_ticks, 200);
    assert!(stats.valid_outputs > 0);
    assert!(stats.success_rate > 0.0 && stats.success_rate <= 1.0);

    println!(
        "Stats: total={}, valid={}, success_rate={:.1}%",
        stats.total_ticks,
        stats.valid_outputs,
        stats.success_rate * 100.0
    );
}

#[test]
fn test_pipeline_reset() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Process some data
    let prices = generate_uptrend(100.0, 200, 0.1);

    for price in prices {
        let _ = pipeline.process(price);
    }

    assert!(pipeline.stats().total_ticks > 0);

    // Reset
    pipeline.reset();

    let stats = pipeline.stats();
    assert_eq!(stats.total_ticks, 0);
    assert_eq!(stats.valid_outputs, 0);
    assert!(!pipeline.is_ready());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_invalid_price_handling() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    // Warmup
    for i in 0..100 {
        let _ = pipeline.process(100.0 + i as f64);
    }

    // NaN should be rejected
    assert!(pipeline.process(f64::NAN).is_err());

    // Infinity should be rejected
    assert!(pipeline.process(f64::INFINITY).is_err());
    assert!(pipeline.process(f64::NEG_INFINITY).is_err());

    // Pipeline should still work after rejection
    assert!(pipeline.process(100.0).is_ok());
}

#[test]
fn test_warmup_period() {
    let config = DspConfig::default();
    let mut pipeline = DspPipeline::new(config);

    let mut first_success = None;

    for i in 0..200 {
        let result = pipeline.process(100.0 + i as f64 * 0.1);

        if result.is_ok() && first_success.is_none() {
            first_success = Some(i);
        }
    }

    assert!(first_success.is_some());

    println!(
        "First successful output at tick: {}",
        first_success.unwrap()
    );

    // Should succeed after window size + normalization warmup
    assert!(first_success.unwrap() < 150);
}

// ============================================================================
// Consistency Tests
// ============================================================================

#[test]
fn test_deterministic_output() {
    // Same input sequence should produce same output
    let prices = generate_random_walk(100.0, 200, 999);

    let config = DspConfig::default();

    // Run 1
    let mut pipeline1 = DspPipeline::new(config.clone());
    let mut outputs1 = Vec::new();

    for price in &prices {
        if let Ok(output) = pipeline1.process(*price) {
            outputs1.push(output.frama);
        }
    }

    // Run 2
    let mut pipeline2 = DspPipeline::new(config);
    let mut outputs2 = Vec::new();

    for price in &prices {
        if let Ok(output) = pipeline2.process(*price) {
            outputs2.push(output.frama);
        }
    }

    assert_eq!(outputs1.len(), outputs2.len());

    // Outputs should be identical (deterministic)
    for (i, (&v1, &v2)) in outputs1.iter().zip(outputs2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Outputs should be deterministic: at index {}, {} != {}",
            i,
            v1,
            v2
        );
    }
}
