//! Sequence anomaly detector for detecting gaps and duplicates

use janus_core::{Exchange, MarketDataEvent, Symbol};
use std::collections::HashMap;

use super::{AnomalyDetector, AnomalyResult, AnomalySeverity};
use crate::Result;

/// Sequence anomaly detector
pub struct SequenceAnomalyDetector {
    /// Maximum allowed gap in sequence numbers
    max_gap: u64,

    /// Last sequence number per (exchange, symbol)
    last_sequences: HashMap<(Exchange, Symbol), u64>,
}

impl SequenceAnomalyDetector {
    /// Create a new sequence anomaly detector
    pub fn new(max_gap: u64) -> Self {
        Self {
            max_gap,
            last_sequences: HashMap::new(),
        }
    }

    /// Extract sequence number from event (if available)
    fn extract_sequence(&self, event: &MarketDataEvent) -> Option<(Exchange, Symbol, u64)> {
        match event {
            MarketDataEvent::Trade(_trade) => {
                // Most exchanges include sequence in trade_id or separate field
                // This is a simplified version - real implementation would parse trade_id
                None
            }
            MarketDataEvent::OrderBook(_ob) => {
                // Order books often have sequence numbers
                None
            }
            _ => None,
        }
    }
}

impl AnomalyDetector for SequenceAnomalyDetector {
    fn name(&self) -> &str {
        "sequence_anomaly_detector"
    }

    fn detect(&mut self, event: &MarketDataEvent) -> Result<AnomalyResult> {
        let symbol = match event {
            MarketDataEvent::Trade(t) => &t.symbol,
            MarketDataEvent::Ticker(t) => &t.symbol,
            MarketDataEvent::Kline(k) => &k.symbol,
            MarketDataEvent::OrderBook(ob) => &ob.symbol,
            MarketDataEvent::Liquidation(l) => &l.symbol,
            MarketDataEvent::FundingRate(f) => &f.symbol,
        };

        if let Some((exchange, sym, seq)) = self.extract_sequence(event) {
            let key = (exchange, sym.clone());

            if let Some(&last_seq) = self.last_sequences.get(&key) {
                let gap = seq.saturating_sub(last_seq);

                if gap > self.max_gap {
                    self.last_sequences.insert(key, seq);

                    return Ok(AnomalyResult::detected(
                        self.name(),
                        sym,
                        AnomalySeverity::High,
                        format!(
                            "Sequence gap detected: {} -> {} (gap: {})",
                            last_seq, seq, gap
                        ),
                    )
                    .with_score(gap as f64));
                }

                if gap == 0 {
                    return Ok(AnomalyResult::detected(
                        self.name(),
                        sym,
                        AnomalySeverity::Low,
                        format!("Duplicate sequence number: {}", seq),
                    )
                    .with_score(0.0));
                }
            }

            self.last_sequences.insert(key, seq);
        }

        Ok(AnomalyResult::none(self.name(), symbol.clone()))
    }

    fn reset(&mut self) {
        self.last_sequences.clear();
    }
}
