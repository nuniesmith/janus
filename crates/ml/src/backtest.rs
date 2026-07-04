//! Event-driven PnL backtest for CNN directional signals.
//!
//! Turns a stream of per-bar directional predictions into simulated trades and a
//! cost-aware PnL, so a classification edge can be judged as *money*, not just
//! precision. The trade mechanics mirror the training labeler
//! ([`crate::labeler`]): enter in the predicted direction at the signal bar's
//! close, bracket with a `tp_atr_mult` / `sl_atr_mult` ATR target/stop, and hold
//! up to `max_hold_bars` (TP/SL are checked intrabar via the high/low; ties
//! resolve to the stop, the conservative assumption). One position at a time —
//! new signals while in a trade are ignored, as a single-lot live bot would.
//!
//! This backtests the CNN signal **standalone**. The live loop runs the CNN as
//! one consensus vote behind additional gates (min-agreement, risk, the 0.7
//! confidence floor), which would only *filter* these trades further — so a
//! standalone profit is necessary, not sufficient, for a live edge, and a
//! standalone loss is decisive.

/// Backtest parameters. Defaults match the champion labeler bracket.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Take-profit distance in entry-bar ATRs.
    pub tp_atr_mult: f64,
    /// Stop-loss distance in entry-bar ATRs.
    pub sl_atr_mult: f64,
    /// Max bars held before a timeout exit at the close.
    pub max_hold_bars: usize,
    /// Wilder ATR period.
    pub atr_period: usize,
    /// Round-trip is charged `2 * cost_per_side` (taker fee + slippage), as a
    /// fraction of notional. 0.0006 ≈ 6 bps/side.
    pub cost_per_side: f64,
    /// Minimum softmax confidence for a prediction to open a trade.
    pub conf_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            tp_atr_mult: 1.5,
            sl_atr_mult: 1.0,
            max_hold_bars: 90,
            atr_period: 14,
            cost_per_side: 0.0006,
            conf_threshold: 0.5,
        }
    }
}

/// Backtest outcome over one price segment.
#[derive(Debug, Clone, Default)]
pub struct BacktestReport {
    /// Trades opened.
    pub n_trades: usize,
    /// Trades that closed with a positive net return.
    pub wins: usize,
    /// `wins / n_trades`.
    pub win_rate: f64,
    /// Summed per-trade return before costs (fraction of notional).
    pub gross_return: f64,
    /// Summed per-trade return after costs.
    pub net_return: f64,
    /// `net_return / n_trades`.
    pub avg_net_per_trade: f64,
    /// Net return in R multiples (units of the risk = `sl_atr_mult` ATR).
    pub sum_r: f64,
    /// Per-trade Sharpe (mean / stddev of net trade returns; not annualised).
    pub sharpe: f64,
    /// Worst peak-to-trough drawdown on the cumulative net-return curve.
    pub max_drawdown: f64,
    /// Long / short trade counts.
    pub long_trades: usize,
    pub short_trades: usize,
}

/// Wilder ATR (EWM of true range, `adjust=false`), one value per bar.
fn wilder_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 0..n {
        let hl = high[i] - low[i];
        let (hc, lc) = if i == 0 {
            (0.0, 0.0)
        } else {
            (
                (high[i] - close[i - 1]).abs(),
                (low[i] - close[i - 1]).abs(),
            )
        };
        tr[i] = hl.max(hc).max(lc);
    }
    let alpha = 1.0 / period as f64;
    let mut atr = vec![0.0; n];
    if n > 0 {
        atr[0] = tr[0];
        for i in 1..n {
            atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i];
        }
    }
    atr
}

/// Simulate trades from directional predictions aligned bar-for-bar with the
/// price arrays. `preds[j] = (class, confidence)` for the same bar as
/// `high[j]/low[j]/close[j]` (class 1 = long, 2 = short; others never trade).
pub fn backtest_signals(
    preds: &[(usize, f64)],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    cfg: &BacktestConfig,
) -> BacktestReport {
    let n = close.len();
    if n == 0 || preds.len() != n {
        return BacktestReport::default();
    }
    let atr = wilder_atr(high, low, close, cfg.atr_period);

    let mut returns: Vec<f64> = Vec::new();
    let mut r_multiples: Vec<f64> = Vec::new();
    let (mut long_trades, mut short_trades) = (0usize, 0usize);
    let mut i = 0usize;
    while i < n {
        let (class, conf) = preds[i];
        let dir = match class {
            1 => 1.0,  // long
            2 => -1.0, // short
            _ => {
                i += 1;
                continue;
            }
        };
        if conf < cfg.conf_threshold || atr[i] <= 0.0 {
            i += 1;
            continue;
        }

        // Enter at this bar's close; bracket sized by the entry-bar ATR.
        let entry = close[i];
        let a = atr[i];
        let (tp, sl) = if dir > 0.0 {
            (entry + cfg.tp_atr_mult * a, entry - cfg.sl_atr_mult * a)
        } else {
            (entry - cfg.tp_atr_mult * a, entry + cfg.sl_atr_mult * a)
        };

        // Walk forward for an intrabar TP/SL, else time out at the close.
        let last = (i + cfg.max_hold_bars).min(n - 1);
        let mut exit = close[last];
        let mut j = i + 1;
        while j <= last {
            let hit_sl = if dir > 0.0 {
                low[j] <= sl
            } else {
                high[j] >= sl
            };
            let hit_tp = if dir > 0.0 {
                high[j] >= tp
            } else {
                low[j] <= tp
            };
            if hit_sl {
                exit = sl; // ties → stop first (conservative)
                break;
            } else if hit_tp {
                exit = tp;
                break;
            }
            j += 1;
        }

        let gross = dir * (exit - entry) / entry;
        let net = gross - 2.0 * cfg.cost_per_side;
        returns.push(net);
        // R multiple: net return over the trade's own risk (the stop distance
        // as a fraction of entry). A clean win ≈ +tp/sl R minus costs.
        let risk_frac = cfg.sl_atr_mult * a / entry;
        if risk_frac > 1e-12 {
            r_multiples.push(net / risk_frac);
        }
        if dir > 0.0 {
            long_trades += 1;
        } else {
            short_trades += 1;
        }

        // Flat until the trade closes (no overlapping positions).
        i = j.max(i + 1);
    }

    finalize(returns, r_multiples, long_trades, short_trades, cfg)
}

fn finalize(
    returns: Vec<f64>,
    r_multiples: Vec<f64>,
    long_trades: usize,
    short_trades: usize,
    cfg: &BacktestConfig,
) -> BacktestReport {
    let n_trades = returns.len();
    if n_trades == 0 {
        return BacktestReport::default();
    }
    let wins = returns.iter().filter(|&&r| r > 0.0).count();
    let net_return: f64 = returns.iter().sum();
    let gross_return = net_return + 2.0 * cfg.cost_per_side * n_trades as f64;
    let mean = net_return / n_trades as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_trades as f64;
    let std = var.sqrt();
    let sharpe = if std > 1e-12 { mean / std } else { 0.0 };

    // Drawdown on the cumulative (additive) net-return equity curve.
    let (mut equity, mut peak, mut max_dd) = (0.0f64, 0.0f64, 0.0f64);
    for r in &returns {
        equity += r;
        peak = peak.max(equity);
        max_dd = max_dd.max(peak - equity);
    }

    let sum_r: f64 = r_multiples.iter().sum();

    BacktestReport {
        n_trades,
        wins,
        win_rate: wins as f64 / n_trades as f64,
        gross_return,
        net_return,
        avg_net_per_trade: mean,
        sum_r,
        sharpe,
        max_drawdown: max_dd,
        long_trades,
        short_trades,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_signals_no_trades() {
        let n = 50;
        let flat = vec![(0usize, 0.9); n];
        let px = vec![100.0; n];
        let r = backtest_signals(&flat, &px, &px, &px, &BacktestConfig::default());
        assert_eq!(r.n_trades, 0);
    }

    #[test]
    fn winning_long_takes_tp_net_of_costs() {
        // Flat then a long signal; price ramps up so TP (1.5 ATR) is hit.
        let n = 20;
        let mut high = vec![100.0; n];
        let mut low = vec![100.0; n];
        let mut close = vec![100.0; n];
        // establish some ATR via small ranges, then a strong up move.
        for i in 0..n {
            high[i] = 100.0 + i as f64 * 0.5 + 0.5;
            low[i] = 100.0 + i as f64 * 0.5 - 0.5;
            close[i] = 100.0 + i as f64 * 0.5;
        }
        let mut preds = vec![(0usize, 0.9); n];
        preds[2] = (1, 0.9); // long at bar 2
        let cfg = BacktestConfig {
            cost_per_side: 0.0001,
            ..Default::default()
        };
        let r = backtest_signals(&preds, &high, &low, &close, &cfg);
        assert_eq!(r.n_trades, 1);
        assert_eq!(r.long_trades, 1);
        // Up-trend long that reaches TP must be net positive after small costs.
        assert!(r.net_return > 0.0, "net {}", r.net_return);
        assert_eq!(r.wins, 1);
    }

    #[test]
    fn low_confidence_is_skipped() {
        let n = 20;
        let px: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let hi: Vec<f64> = px.iter().map(|p| p + 0.5).collect();
        let lo: Vec<f64> = px.iter().map(|p| p - 0.5).collect();
        let mut preds = vec![(0usize, 0.9); n];
        preds[2] = (1, 0.3); // below default 0.5 threshold
        let r = backtest_signals(&preds, &hi, &lo, &px, &BacktestConfig::default());
        assert_eq!(r.n_trades, 0);
    }
}
