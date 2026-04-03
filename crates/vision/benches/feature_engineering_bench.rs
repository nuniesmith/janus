//! Benchmarks for feature engineering performance.
//!
//! Run with:
//! ```bash
//! cargo bench --package vision --bench feature_engineering_bench
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vision::{
    data::{
        csv_loader::OhlcvCandle,
        dataset::{OhlcvDataset, SequenceConfig},
    },
    preprocessing::features::{FeatureConfig, FeatureEngineer},
};

use chrono::{Duration, Utc};

/// Generate synthetic OHLCV data for benchmarking
fn generate_test_candles(n: usize) -> Vec<OhlcvCandle> {
    let start_time = Utc::now();
    let mut candles = Vec::with_capacity(n);
    let mut price = 100.0;

    for i in 0..n {
        let timestamp = start_time + Duration::minutes(i as i64);

        // Simple random walk
        let change = 0.02 * (((i * 7) % 100) as f64 / 50.0 - 1.0);
        price *= 1.0 + change;
        price = price.max(50.0).min(200.0);

        let high = price * 1.005;
        let low = price * 0.995;
        let close = low + (high - low) * 0.5;
        let volume = 1000000.0;

        candles.push(OhlcvCandle::new(timestamp, price, high, low, close, volume));
    }

    candles
}

/// Benchmark: Basic feature computation (OHLCV + returns only)
fn bench_minimal_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimal_features");

    for size in [100, 500, 1000, 5000].iter() {
        let candles = generate_test_candles(*size);
        let dataset = OhlcvDataset::from_candles(
            candles,
            SequenceConfig {
                sequence_length: 60,
                stride: 10,
                prediction_horizon: 1,
            },
        )
        .unwrap();

        let config = FeatureConfig::minimal();
        let engineer = FeatureEngineer::new(config);

        group.throughput(Throughput::Elements(dataset.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, ds| {
            b.iter(|| engineer.compute_dataset_features(black_box(ds)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Common indicators (19 features)
fn bench_common_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_indicators");

    for size in [100, 500, 1000, 5000].iter() {
        let candles = generate_test_candles(*size);
        let dataset = OhlcvDataset::from_candles(
            candles,
            SequenceConfig {
                sequence_length: 60,
                stride: 10,
                prediction_horizon: 1,
            },
        )
        .unwrap();

        let config = FeatureConfig::with_common_indicators();
        let engineer = FeatureEngineer::new(config);

        group.throughput(Throughput::Elements(dataset.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &dataset, |b, ds| {
            b.iter(|| engineer.compute_dataset_features(black_box(ds)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark: Parallel vs Sequential
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_comparison");

    let candles = generate_test_candles(5000);
    let dataset = OhlcvDataset::from_candles(
        candles,
        SequenceConfig {
            sequence_length: 60,
            stride: 10,
            prediction_horizon: 1,
        },
    )
    .unwrap();

    let config = FeatureConfig::with_common_indicators();
    let engineer = FeatureEngineer::new(config);

    group.throughput(Throughput::Elements(dataset.len() as u64));

    // Sequential
    group.bench_function("sequential", |b| {
        b.iter(|| {
            engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // Parallel
    group.bench_function("parallel", |b| {
        b.iter(|| {
            engineer
                .compute_dataset_features_parallel(black_box(&dataset))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark: Individual indicators
fn bench_individual_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("individual_indicators");

    let candles = generate_test_candles(1000);
    let dataset = OhlcvDataset::from_candles(
        candles,
        SequenceConfig {
            sequence_length: 60,
            stride: 10,
            prediction_horizon: 1,
        },
    )
    .unwrap();

    // SMA only
    let sma_config = FeatureConfig {
        include_ohlcv: true,
        sma_periods: vec![10, 20],
        ..Default::default()
    };
    let sma_engineer = FeatureEngineer::new(sma_config);

    group.bench_function("sma", |b| {
        b.iter(|| {
            sma_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // EMA only
    let ema_config = FeatureConfig {
        include_ohlcv: true,
        ema_periods: vec![12, 26],
        ..Default::default()
    };
    let ema_engineer = FeatureEngineer::new(ema_config);

    group.bench_function("ema", |b| {
        b.iter(|| {
            ema_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // RSI only
    let rsi_config = FeatureConfig {
        include_ohlcv: true,
        rsi_period: Some(14),
        ..Default::default()
    };
    let rsi_engineer = FeatureEngineer::new(rsi_config);

    group.bench_function("rsi", |b| {
        b.iter(|| {
            rsi_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // MACD only
    let macd_config = FeatureConfig {
        include_ohlcv: true,
        macd_config: Some((12, 26, 9)),
        ..Default::default()
    };
    let macd_engineer = FeatureEngineer::new(macd_config);

    group.bench_function("macd", |b| {
        b.iter(|| {
            macd_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // ATR only
    let atr_config = FeatureConfig {
        include_ohlcv: true,
        atr_period: Some(14),
        ..Default::default()
    };
    let atr_engineer = FeatureEngineer::new(atr_config);

    group.bench_function("atr", |b| {
        b.iter(|| {
            atr_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    // Bollinger Bands only
    let bb_config = FeatureConfig {
        include_ohlcv: true,
        bollinger_bands: Some((20, 2.0)),
        ..Default::default()
    };
    let bb_engineer = FeatureEngineer::new(bb_config);

    group.bench_function("bollinger_bands", |b| {
        b.iter(|| {
            bb_engineer
                .compute_dataset_features(black_box(&dataset))
                .unwrap()
        });
    });

    group.finish();
}

/// Benchmark: Sequence length impact
fn bench_sequence_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_length");

    let candles = generate_test_candles(5000);
    let config = FeatureConfig::with_common_indicators();

    for seq_len in [30, 60, 120, 240].iter() {
        let dataset = OhlcvDataset::from_candles(
            candles.clone(),
            SequenceConfig {
                sequence_length: *seq_len,
                stride: 10,
                prediction_horizon: 1,
            },
        )
        .unwrap();

        let engineer = FeatureEngineer::new(config.clone());

        group.throughput(Throughput::Elements(dataset.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &dataset, |b, ds| {
            b.iter(|| {
                engineer
                    .compute_dataset_features_parallel(black_box(ds))
                    .unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_minimal_features,
    bench_common_indicators,
    bench_parallel_vs_sequential,
    bench_individual_indicators,
    bench_sequence_length,
);
criterion_main!(benches);
