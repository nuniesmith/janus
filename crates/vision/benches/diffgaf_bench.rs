use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use burn::backend::NdArray;
use burn::tensor::Tensor;

type TestBackend = NdArray<f32>;

use vision::diffgaf::combined::DiffGafLstmConfig;
use vision::diffgaf::config::DiffGAFConfig;

/// Benchmark the DiffGAF transform (time series → GAF image)
fn bench_diffgaf_transform(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("diffgaf_transform");

    for &(batch, time_steps, features) in &[
        (1, 30, 5),
        (1, 60, 5),
        (4, 60, 5),
        (8, 60, 5),
        (16, 60, 5),
        (4, 120, 5),
        (4, 60, 10),
    ] {
        let config = DiffGAFConfig {
            num_features: features,
            time_steps,
            ..Default::default()
        };
        let diffgaf = config.init::<TestBackend>(&device);

        let id = BenchmarkId::new(
            "transform",
            format!("b{}_t{}_f{}", batch, time_steps, features),
        );

        group.bench_with_input(id, &(batch, time_steps, features), |b, &(bs, ts, fs)| {
            b.iter(|| {
                let input = Tensor::<TestBackend, 3>::ones([bs, ts, fs], &device);
                let output = diffgaf.forward(input);
                black_box(output)
            })
        });
    }

    group.finish();
}

/// Benchmark the combined DiffGAF + LSTM forward pass
fn bench_diffgaf_lstm_forward(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("diffgaf_lstm_forward");
    group.sample_size(20); // Fewer samples since LSTM is heavier

    for &(batch, time_steps, hidden, classes) in &[
        (1, 60, 32, 3),
        (4, 60, 32, 3),
        (4, 60, 64, 3),
        (8, 60, 32, 3),
    ] {
        let config = DiffGafLstmConfig {
            input_features: 5,
            time_steps,
            lstm_hidden_size: hidden,
            num_lstm_layers: 1,
            num_classes: classes,
            dropout: 0.0,
            gaf_pool_size: 16,
            bidirectional: false,
        };

        let model = config.init::<TestBackend>(&device);

        let id = BenchmarkId::new(
            "forward",
            format!("b{}_t{}_h{}_c{}", batch, time_steps, hidden, classes),
        );

        group.bench_with_input(id, &batch, |b, &bs| {
            b.iter(|| {
                let input = Tensor::<TestBackend, 3>::ones([bs, time_steps, 5], &device);
                let output = model.forward(input);
                black_box(output)
            })
        });
    }

    group.finish();
}

/// Benchmark softmax classification output
fn bench_diffgaf_lstm_classify(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("diffgaf_lstm_classify");
    group.sample_size(20);

    let config = DiffGafLstmConfig {
        input_features: 5,
        time_steps: 60,
        lstm_hidden_size: 32,
        num_lstm_layers: 1,
        num_classes: 3,
        dropout: 0.0,
        gaf_pool_size: 16,
        bidirectional: false,
    };

    let model = config.init::<TestBackend>(&device);

    for &batch in &[1, 4, 8, 16] {
        let id = BenchmarkId::new("classify_softmax", format!("b{}", batch));

        group.bench_with_input(id, &batch, |b, &bs| {
            b.iter(|| {
                let input = Tensor::<TestBackend, 3>::ones([bs, 60, 5], &device);
                let probs = model.forward_with_softmax(input);
                black_box(probs)
            })
        });
    }

    group.finish();
}

/// Benchmark tensor creation and GAF pool sizing impact
fn bench_gaf_pool_size(c: &mut Criterion) {
    let device = Default::default();

    let mut group = c.benchmark_group("gaf_pool_size");
    group.sample_size(20);

    for &pool_size in &[4, 8, 16, 32] {
        let config = DiffGafLstmConfig {
            input_features: 5,
            time_steps: 60,
            lstm_hidden_size: 32,
            num_lstm_layers: 1,
            num_classes: 3,
            dropout: 0.0,
            gaf_pool_size: pool_size,
            bidirectional: false,
        };

        let model = config.init::<TestBackend>(&device);

        let id = BenchmarkId::new("pool", format!("k{}", pool_size));

        group.bench_with_input(id, &pool_size, |b, &_ps| {
            b.iter(|| {
                let input = Tensor::<TestBackend, 3>::ones([4, 60, 5], &device);
                let output = model.forward(input);
                black_box(output)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_diffgaf_transform,
    bench_diffgaf_lstm_forward,
    bench_diffgaf_lstm_classify,
    bench_gaf_pool_size,
);
criterion_main!(benches);
