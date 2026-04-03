use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use janus_rate_limiter::{TokenBucket, TokenBucketConfig};
use std::hint::black_box;
use std::time::Duration;

fn benchmark_single_threaded_acquire(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_threaded_acquire");

    for capacity in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("no_sliding_window", capacity),
            capacity,
            |b, &capacity| {
                let config = TokenBucketConfig {
                    capacity,
                    refill_rate: (capacity as f64) / 60.0,
                    sliding_window: false,
                    safety_margin: 1.0,
                    window_duration: Duration::from_secs(60),
                };
                let bucket = TokenBucket::new(config).unwrap();

                b.iter(|| {
                    // Acquire small amounts repeatedly
                    for _ in 0..10 {
                        let _ = bucket.acquire(black_box(1));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("with_sliding_window", capacity),
            capacity,
            |b, &capacity| {
                let config = TokenBucketConfig {
                    capacity,
                    refill_rate: (capacity as f64) / 60.0,
                    sliding_window: true,
                    safety_margin: 1.0,
                    window_duration: Duration::from_secs(60),
                };
                let bucket = TokenBucket::new(config).unwrap();

                b.iter(|| {
                    for _ in 0..10 {
                        let _ = bucket.acquire(black_box(1));
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_acquire(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_acquire");
    group.sample_size(50); // Reduce sample size for concurrent tests

    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let config = TokenBucketConfig {
                        capacity: 10000,
                        refill_rate: 1000.0,
                        sliding_window: false,
                        safety_margin: 1.0,
                        window_duration: Duration::from_secs(60),
                    };
                    let bucket = std::sync::Arc::new(TokenBucket::new(config).unwrap());

                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let bucket = bucket.clone();
                            std::thread::spawn(move || {
                                for _ in 0..100 {
                                    let _ = bucket.acquire(1);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_header_update(c: &mut Criterion) {
    c.bench_function("header_update", |b| {
        let config = TokenBucketConfig::binance_spot();
        let bucket = TokenBucket::new(config).unwrap();

        b.iter(|| {
            bucket.update_from_headers(black_box(3000), black_box(6000));
        });
    });
}

fn benchmark_metrics_read(c: &mut Criterion) {
    c.bench_function("metrics_read", |b| {
        let config = TokenBucketConfig::binance_spot();
        let bucket = TokenBucket::new(config).unwrap();

        b.iter(|| {
            let _ = black_box(bucket.metrics());
        });
    });
}

fn benchmark_realistic_workload(c: &mut Criterion) {
    c.bench_function("realistic_binance_workload", |b| {
        let config = TokenBucketConfig::binance_spot();
        let bucket = TokenBucket::new(config).unwrap();

        b.iter(|| {
            // Simulate realistic API call pattern:
            // - Most calls are lightweight (weight 1-5)
            // - Occasional heavy calls (weight 10-50)
            // - Periodic header updates

            for i in 0..100 {
                let weight = if i % 10 == 0 {
                    black_box(20) // Heavy call every 10 requests
                } else {
                    black_box(2) // Light call
                };

                let _ = bucket.acquire(weight);

                // Simulate header update every 20 requests
                if i % 20 == 0 {
                    bucket.update_from_headers(i * 2, 6000);
                }
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_single_threaded_acquire,
    benchmark_concurrent_acquire,
    benchmark_header_update,
    benchmark_metrics_read,
    benchmark_realistic_workload
);

criterion_main!(benches);
