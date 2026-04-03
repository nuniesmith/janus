//! Orderbook throughput benchmarks.
//!
//! Measures the performance of core order book operations:
//! - Snapshot application
//! - Delta updates
//! - Liquidity consumption (market order sweep)
//! - VWAP calculation
//! - Matching engine order submission
//! - Delta batch application

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use janus_lob::matching_engine::{MatchingEngine, MatchingEngineConfig};
use janus_lob::order_types::{Order, Side};
use janus_lob::orderbook::{
    OrderBook, OrderBookDelta, OrderBookDeltaBatch, OrderBookSnapshot, PriceLevel,
};

use chrono::Utc;

/// Generate a synthetic snapshot with `n_levels` on each side.
fn synthetic_snapshot(n_levels: usize) -> OrderBookSnapshot {
    let base_bid = dec!(50000.0);
    let base_ask = dec!(50001.0);
    let tick = dec!(0.5);

    let bids: Vec<PriceLevel> = (0..n_levels)
        .map(|i| {
            let price = base_bid - tick * Decimal::from(i as u64);
            let qty = dec!(1.0) + Decimal::from(i as u64 % 10) * dec!(0.1);
            PriceLevel::new(price, qty)
        })
        .collect();

    let asks: Vec<PriceLevel> = (0..n_levels)
        .map(|i| {
            let price = base_ask + tick * Decimal::from(i as u64);
            let qty = dec!(1.0) + Decimal::from(i as u64 % 10) * dec!(0.1);
            PriceLevel::new(price, qty)
        })
        .collect();

    OrderBookSnapshot {
        symbol: "BTC/USDT".into(),
        bids,
        asks,
        timestamp: Utc::now(),
        sequence: 1,
    }
}

/// Build a pre-populated order book with `n_levels` per side.
fn populated_book(n_levels: usize) -> OrderBook {
    let mut book = OrderBook::new("BTC/USDT");
    book.apply_snapshot(synthetic_snapshot(n_levels)).unwrap();
    book
}

// ── Benchmark: apply_snapshot ──────────────────────────────────────────────

fn bench_apply_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_snapshot");

    for n_levels in [100, 500, 1000] {
        let snapshot = synthetic_snapshot(n_levels);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_levels", n_levels)),
            &snapshot,
            |b, snap| {
                b.iter(|| {
                    let mut book = OrderBook::new("BTC/USDT");
                    book.apply_snapshot(black_box(snap.clone())).unwrap();
                    black_box(&book);
                });
            },
        );
    }

    group.finish();
}

// ── Benchmark: apply_delta ────────────────────────────────────────────────

fn bench_apply_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_delta");

    for n_levels in [100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_levels_base", n_levels)),
            &n_levels,
            |b, &n| {
                b.iter_batched(
                    || {
                        let book = populated_book(n);
                        let seq = book.stats().last_sequence;
                        // Pre-generate 1000 deltas
                        let deltas: Vec<OrderBookDelta> = (0..1000u64)
                            .map(|i| {
                                let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                                let price = if side == Side::Buy {
                                    dec!(50000.0) - Decimal::from(i % n as u64) * dec!(0.5)
                                } else {
                                    dec!(50001.0) + Decimal::from(i % n as u64) * dec!(0.5)
                                };
                                OrderBookDelta {
                                    symbol: "BTC/USDT".into(),
                                    side,
                                    price,
                                    quantity: dec!(1.5) + Decimal::from(i % 5) * dec!(0.1),
                                    timestamp: Utc::now(),
                                    sequence: seq + i + 1,
                                }
                            })
                            .collect();
                        (book, deltas)
                    },
                    |(mut book, deltas)| {
                        for delta in deltas {
                            let _ = book.apply_delta(black_box(delta));
                        }
                        black_box(&book);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Benchmark: apply_delta_batch ──────────────────────────────────────────

fn bench_apply_delta_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_delta_batch");

    for batch_size in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            &batch_size,
            |b, &bs| {
                b.iter_batched(
                    || {
                        let book = populated_book(200);
                        let seq = book.stats().last_sequence + 1;
                        let mut batch = OrderBookDeltaBatch::new("BTC/USDT", seq);
                        for i in 0..bs {
                            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                            let price = if side == Side::Buy {
                                dec!(50000.0) - Decimal::from(i as u64 % 200) * dec!(0.5)
                            } else {
                                dec!(50001.0) + Decimal::from(i as u64 % 200) * dec!(0.5)
                            };
                            batch.add(
                                side,
                                price,
                                dec!(2.0) + Decimal::from(i as u64 % 5) * dec!(0.1),
                            );
                        }
                        (book, batch)
                    },
                    |(mut book, batch)| {
                        let _ = book.apply_delta_batch(black_box(batch));
                        black_box(&book);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Benchmark: consume_liquidity ──────────────────────────────────────────

fn bench_consume_liquidity(c: &mut Criterion) {
    let mut group = c.benchmark_group("consume_liquidity");

    for sweep_levels in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_levels", sweep_levels)),
            &sweep_levels,
            |b, &n_sweep| {
                b.iter_batched(
                    || {
                        let book = populated_book(500);
                        // Each level has ~1.0-1.9 qty, so sweeping n levels needs ~n * 1.5 qty
                        let sweep_qty = Decimal::from(n_sweep as u64) * dec!(1.5);
                        (book, sweep_qty)
                    },
                    |(mut book, qty)| {
                        let fills = book
                            .asks_mut()
                            .consume_liquidity(black_box(qty), None, None);
                        black_box(fills);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ── Benchmark: VWAP calculation ───────────────────────────────────────────

fn bench_vwap(c: &mut Criterion) {
    let mut group = c.benchmark_group("vwap");

    let book = populated_book(1000);

    for depth in [dec!(10.0), dec!(50.0), dec!(200.0)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("depth_{}", depth)),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let vwap = book.asks().vwap_for_depth(black_box(d));
                    black_box(vwap);
                });
            },
        );
    }

    for depth in [dec!(10.0), dec!(50.0), dec!(200.0)] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("buy_vwap_depth_{}", depth)),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let vwap = book.buy_vwap(black_box(d));
                    black_box(vwap);
                });
            },
        );
    }

    group.finish();
}

// ── Benchmark: weighted_mid_price & impact_cost ───────────────────────────

fn bench_book_analytics(c: &mut Criterion) {
    let mut group = c.benchmark_group("book_analytics");
    let book = populated_book(500);

    group.bench_function("mid_price", |b| {
        b.iter(|| {
            black_box(book.mid_price());
        });
    });

    group.bench_function("weighted_mid_price", |b| {
        b.iter(|| {
            black_box(book.weighted_mid_price());
        });
    });

    group.bench_function("spread_bps", |b| {
        b.iter(|| {
            black_box(book.spread_bps());
        });
    });

    group.bench_function("impact_cost_buy_1", |b| {
        b.iter(|| {
            black_box(book.impact_cost(Side::Buy, dec!(1.0)));
        });
    });

    group.bench_function("impact_cost_buy_10", |b| {
        b.iter(|| {
            black_box(book.impact_cost(Side::Buy, dec!(10.0)));
        });
    });

    group.bench_function("imbalance_5", |b| {
        b.iter(|| {
            black_box(book.order_book_imbalance(5));
        });
    });

    group.finish();
}

// ── Benchmark: matching engine submit ─────────────────────────────────────

fn bench_matching_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_engine");

    let config = MatchingEngineConfig::default()
        .without_market_impact()
        .without_latency();

    group.bench_function("market_buy_single_level", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                let book = populated_book(100);
                let order = Order::market(Side::Buy, dec!(0.5)).with_symbol("BTC/USDT");
                (engine, book, order)
            },
            |(mut engine, mut book, order)| {
                let result = engine.submit(&mut book, black_box(order));
                let _ = black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("market_buy_sweep_10_levels", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                let book = populated_book(100);
                // Each level ~1.0-1.9 qty, so 15 units should sweep ~10+ levels
                let order = Order::market(Side::Buy, dec!(15.0)).with_symbol("BTC/USDT");
                (engine, book, order)
            },
            |(mut engine, mut book, order)| {
                let result = engine.submit(&mut book, black_box(order));
                let _ = black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("market_sell_single_level", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                let book = populated_book(100);
                let order = Order::market(Side::Sell, dec!(0.5)).with_symbol("BTC/USDT");
                (engine, book, order)
            },
            |(mut engine, mut book, order)| {
                let result = engine.submit(&mut book, black_box(order));
                let _ = black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("limit_order_rests", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                let book = populated_book(100);
                // Buy well below best ask — will rest
                let order =
                    Order::limit(Side::Buy, dec!(49900.0), dec!(1.0)).with_symbol("BTC/USDT");
                (engine, book, order)
            },
            |(mut engine, mut book, order)| {
                let result = engine.submit(&mut book, black_box(order));
                let _ = black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("post_only_rests", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                let book = populated_book(100);
                let order =
                    Order::post_only(Side::Buy, dec!(49999.0), dec!(1.0)).with_symbol("BTC/USDT");
                (engine, book, order)
            },
            |(mut engine, mut book, order)| {
                let result = engine.submit(&mut book, black_box(order));
                let _ = black_box(result);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("throughput_1000_market_orders", |b| {
        b.iter_batched(
            || {
                let engine = MatchingEngine::new(config.clone());
                // Large book so we don't exhaust liquidity
                let book = populated_book(1000);
                let orders: Vec<Order> = (0..1000)
                    .map(|i| {
                        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                        Order::market(side, dec!(0.01)).with_symbol("BTC/USDT")
                    })
                    .collect();
                (engine, book, orders)
            },
            |(mut engine, mut book, orders)| {
                for order in orders {
                    let _ = engine.submit(&mut book, black_box(order));
                }
                black_box(&book);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_apply_snapshot,
    bench_apply_delta,
    bench_apply_delta_batch,
    bench_consume_liquidity,
    bench_vwap,
    bench_book_analytics,
    bench_matching_engine,
);
criterion_main!(benches);
