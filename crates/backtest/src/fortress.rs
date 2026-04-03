//! # Temporal Fortress
//!
//! Zero-lookahead state machine using double-buffering to prevent future information leakage.
//!
//! ## Architecture
//!
//! The temporal fortress maintains two separate state buffers:
//! - **Known State**: All data available up to and including the current timestamp
//! - **Future State**: Data that exists but hasn't been "observed" yet
//!
//! This architecture guarantees that strategies cannot access future information,
//! preventing the most common backtesting pitfall: look-ahead bias.
//!
//! ## State Transitions
//!
//! ```text
//! Time:  T0    T1    T2    T3
//!        |     |     |     |
//! Known: [====]
//! Future:      [===========]
//!
//! After advance(T1):
//! Known: [==========]
//! Future:            [=====]
//! ```

use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

/// Errors that can occur in the temporal fortress
#[derive(Error, Debug)]
pub enum FortressError {
    #[error("Attempted to access future data at {requested}, current time is {current}")]
    LookaheadViolation {
        requested: DateTime<Utc>,
        current: DateTime<Utc>,
    },

    #[error("Cannot rewind time from {current} to {requested}")]
    TimeRewind {
        current: DateTime<Utc>,
        requested: DateTime<Utc>,
    },

    #[error("No data available at timestamp {0}")]
    NoDataAvailable(DateTime<Utc>),
}

/// A time-stamped event that can be buffered
pub trait TimestampedEvent: Clone {
    fn timestamp(&self) -> DateTime<Utc>;
}

/// Double-buffered temporal state container
///
/// Maintains strict separation between "known" (past/present) and "future" data.
/// The barrier timestamp controls what data is visible to the strategy.
pub struct TemporalFortress<T: TimestampedEvent> {
    /// Current "now" barrier - strategies can only see data <= this time
    barrier_time: Option<DateTime<Utc>>,

    /// Events that are "known" (timestamp <= barrier_time)
    /// Stored in temporal order for efficient range queries
    known_buffer: BTreeMap<DateTime<Utc>, Vec<T>>,

    /// Events that are "future" (timestamp > barrier_time)
    /// Stored in a queue for efficient frontier advancement
    future_buffer: VecDeque<(DateTime<Utc>, T)>,

    /// Statistics for monitoring
    lookback_window: Option<chrono::Duration>,
    total_events_processed: u64,
    barrier_advances: u64,
}

impl<T: TimestampedEvent> TemporalFortress<T> {
    /// Create a new temporal fortress
    pub fn new() -> Self {
        Self {
            barrier_time: None,
            known_buffer: BTreeMap::new(),
            future_buffer: VecDeque::new(),
            lookback_window: None,
            total_events_processed: 0,
            barrier_advances: 0,
        }
    }

    /// Create a fortress with a lookback window (older data will be pruned)
    pub fn with_lookback(window: chrono::Duration) -> Self {
        Self {
            barrier_time: None,
            known_buffer: BTreeMap::new(),
            future_buffer: VecDeque::new(),
            lookback_window: Some(window),
            total_events_processed: 0,
            barrier_advances: 0,
        }
    }

    /// Load events into the future buffer (must be called before starting replay)
    ///
    /// Events are automatically sorted by timestamp and placed in the future buffer.
    /// This is typically called during initialization with historical data.
    pub fn load_events(&mut self, events: Vec<T>) {
        for event in events {
            let ts = event.timestamp();
            self.future_buffer.push_back((ts, event));
        }

        // Sort the future buffer by timestamp
        let mut buffer: Vec<_> = self.future_buffer.drain(..).collect();
        buffer.sort_by_key(|(ts, _)| *ts);
        self.future_buffer = buffer.into_iter().collect();
    }

    /// Advance the barrier time, making more data "known"
    ///
    /// This is the core operation - it moves the temporal boundary forward,
    /// transferring events from future to known buffer.
    ///
    /// # Errors
    ///
    /// Returns `TimeRewind` if attempting to move barrier backwards.
    pub fn advance_to(&mut self, new_time: DateTime<Utc>) -> Result<usize, FortressError> {
        // Prevent time travel to the past
        if let Some(current) = self.barrier_time
            && new_time < current
        {
            return Err(FortressError::TimeRewind {
                current,
                requested: new_time,
            });
        }

        let mut transferred = 0;

        // Move events from future to known buffer
        while let Some((ts, _)) = self.future_buffer.front() {
            if *ts <= new_time {
                let (ts, event) = self.future_buffer.pop_front().unwrap();
                self.known_buffer.entry(ts).or_default().push(event);
                transferred += 1;
                self.total_events_processed += 1;
            } else {
                break;
            }
        }

        self.barrier_time = Some(new_time);
        self.barrier_advances += 1;

        // Prune old data if lookback window is set
        if let Some(window) = self.lookback_window {
            let cutoff = new_time - window;
            self.known_buffer = self.known_buffer.split_off(&cutoff);
        }

        Ok(transferred)
    }

    /// Get all known events at a specific timestamp
    ///
    /// # Errors
    ///
    /// Returns `LookaheadViolation` if requesting data from the future.
    pub fn get_at(&self, time: DateTime<Utc>) -> Result<&[T], FortressError> {
        if let Some(barrier) = self.barrier_time
            && time > barrier
        {
            return Err(FortressError::LookaheadViolation {
                requested: time,
                current: barrier,
            });
        }

        Ok(self
            .known_buffer
            .get(&time)
            .map(|v| v.as_slice())
            .unwrap_or(&[]))
    }

    /// Get all known events in a time range [start, end]
    ///
    /// # Errors
    ///
    /// Returns `LookaheadViolation` if requesting data from the future.
    pub fn get_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<&T>, FortressError> {
        if let Some(barrier) = self.barrier_time
            && end > barrier
        {
            return Err(FortressError::LookaheadViolation {
                requested: end,
                current: barrier,
            });
        }

        let mut result = Vec::new();
        for (_, events) in self.known_buffer.range(start..=end) {
            result.extend(events.iter());
        }

        Ok(result)
    }

    /// Get the most recent known events (up to `limit`)
    pub fn get_recent(&self, limit: usize) -> Vec<&T> {
        let mut result = Vec::new();

        for (_, events) in self.known_buffer.iter().rev() {
            for event in events.iter().rev() {
                result.push(event);
                if result.len() >= limit {
                    return result;
                }
            }
        }

        result
    }

    /// Get all known events in chronological order
    pub fn get_all_known(&self) -> Vec<&T> {
        let mut result = Vec::new();
        for (_, events) in self.known_buffer.iter() {
            result.extend(events.iter());
        }
        result
    }

    /// Get the current barrier time
    pub fn current_time(&self) -> Option<DateTime<Utc>> {
        self.barrier_time
    }

    /// Get the next future timestamp (peek without advancing)
    pub fn peek_next_time(&self) -> Option<DateTime<Utc>> {
        self.future_buffer.front().map(|(ts, _)| *ts)
    }

    /// Check if there are more events in the future
    pub fn has_more(&self) -> bool {
        !self.future_buffer.is_empty()
    }

    /// Get statistics about the fortress state
    pub fn stats(&self) -> FortressStats {
        FortressStats {
            known_count: self.known_buffer.values().map(|v| v.len()).sum(),
            future_count: self.future_buffer.len(),
            total_processed: self.total_events_processed,
            barrier_advances: self.barrier_advances,
            current_time: self.barrier_time,
        }
    }

    /// Clear all buffers and reset state
    pub fn reset(&mut self) {
        self.barrier_time = None;
        self.known_buffer.clear();
        self.future_buffer.clear();
        self.total_events_processed = 0;
        self.barrier_advances = 0;
    }
}

impl<T: TimestampedEvent> Default for TemporalFortress<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the temporal fortress state
#[derive(Debug, Clone)]
pub struct FortressStats {
    pub known_count: usize,
    pub future_count: usize,
    pub total_processed: u64,
    pub barrier_advances: u64,
    pub current_time: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct TestEvent {
        timestamp: DateTime<Utc>,
        value: f64,
    }

    impl TimestampedEvent for TestEvent {
        fn timestamp(&self) -> DateTime<Utc> {
            self.timestamp
        }
    }

    fn make_event(secs: i64, value: f64) -> TestEvent {
        TestEvent {
            timestamp: DateTime::from_timestamp(secs, 0).unwrap(),
            value,
        }
    }

    #[test]
    fn test_basic_advancement() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        let events = vec![
            make_event(1000, 100.0),
            make_event(2000, 200.0),
            make_event(3000, 300.0),
        ];

        fortress.load_events(events);

        // Initially, barrier is None
        assert_eq!(fortress.current_time(), None);
        assert_eq!(fortress.stats().known_count, 0);
        assert_eq!(fortress.stats().future_count, 3);

        // Advance to t=1500 (should reveal first event)
        let t1 = DateTime::from_timestamp(1500, 0).unwrap();
        let transferred = fortress.advance_to(t1).unwrap();
        assert_eq!(transferred, 1);
        assert_eq!(fortress.stats().known_count, 1);
        assert_eq!(fortress.stats().future_count, 2);

        // Advance to t=2500 (should reveal second event)
        let t2 = DateTime::from_timestamp(2500, 0).unwrap();
        fortress.advance_to(t2).unwrap();
        assert_eq!(fortress.stats().known_count, 2);
        assert_eq!(fortress.stats().future_count, 1);
    }

    #[test]
    fn test_lookahead_prevention() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        let events = vec![make_event(1000, 100.0), make_event(2000, 200.0)];

        fortress.load_events(events);
        fortress
            .advance_to(DateTime::from_timestamp(1500, 0).unwrap())
            .unwrap();

        // Should be able to access t=1000
        let result = fortress.get_at(DateTime::from_timestamp(1000, 0).unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);

        // Should NOT be able to access t=2000 (future)
        let result = fortress.get_at(DateTime::from_timestamp(2000, 0).unwrap());
        assert!(matches!(
            result,
            Err(FortressError::LookaheadViolation { .. })
        ));
    }

    #[test]
    fn test_time_rewind_prevention() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        fortress
            .advance_to(DateTime::from_timestamp(2000, 0).unwrap())
            .unwrap();

        // Should not be able to rewind
        let result = fortress.advance_to(DateTime::from_timestamp(1000, 0).unwrap());
        assert!(matches!(result, Err(FortressError::TimeRewind { .. })));
    }

    #[test]
    fn test_range_query() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        let events = vec![
            make_event(1000, 100.0),
            make_event(2000, 200.0),
            make_event(3000, 300.0),
            make_event(4000, 400.0),
        ];

        fortress.load_events(events);
        fortress
            .advance_to(DateTime::from_timestamp(3500, 0).unwrap())
            .unwrap();

        // Query range [1500, 2500] should return event at t=2000
        let start = DateTime::from_timestamp(1500, 0).unwrap();
        let end = DateTime::from_timestamp(2500, 0).unwrap();
        let result = fortress.get_range(start, end).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 200.0);

        // Query range that includes future should fail
        let end_future = DateTime::from_timestamp(4000, 0).unwrap();
        let result = fortress.get_range(start, end_future);
        assert!(matches!(
            result,
            Err(FortressError::LookaheadViolation { .. })
        ));
    }

    #[test]
    fn test_lookback_window() {
        let window = chrono::Duration::seconds(1000);
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::with_lookback(window);

        let events = vec![
            make_event(1000, 100.0),
            make_event(2000, 200.0),
            make_event(3000, 300.0),
        ];

        fortress.load_events(events);

        // Advance to t=3500
        fortress
            .advance_to(DateTime::from_timestamp(3500, 0).unwrap())
            .unwrap();

        // Event at t=1000 should be pruned (outside window)
        // Event at t=2000 should be pruned (outside 1000s window from t=3500)
        // Only event at t=3000 should remain (within 1000s window from t=3500)
        assert_eq!(fortress.stats().known_count, 1);

        // Verify we can't access pruned data
        let result = fortress
            .get_at(DateTime::from_timestamp(1000, 0).unwrap())
            .unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_peek_next() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        let events = vec![make_event(1000, 100.0), make_event(2000, 200.0)];

        fortress.load_events(events);

        // Peek should show t=1000
        assert_eq!(
            fortress.peek_next_time(),
            Some(DateTime::from_timestamp(1000, 0).unwrap())
        );

        // Advance to t=1500
        fortress
            .advance_to(DateTime::from_timestamp(1500, 0).unwrap())
            .unwrap();

        // Peek should now show t=2000
        assert_eq!(
            fortress.peek_next_time(),
            Some(DateTime::from_timestamp(2000, 0).unwrap())
        );

        // Advance to t=2500
        fortress
            .advance_to(DateTime::from_timestamp(2500, 0).unwrap())
            .unwrap();

        // No more future events
        assert_eq!(fortress.peek_next_time(), None);
        assert!(!fortress.has_more());
    }

    #[test]
    fn test_get_recent() {
        let mut fortress: TemporalFortress<TestEvent> = TemporalFortress::new();

        let events = vec![
            make_event(1000, 100.0),
            make_event(2000, 200.0),
            make_event(3000, 300.0),
            make_event(4000, 400.0),
        ];

        fortress.load_events(events);
        fortress
            .advance_to(DateTime::from_timestamp(3500, 0).unwrap())
            .unwrap();

        // Get 2 most recent
        let recent = fortress.get_recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].value, 300.0); // Most recent first
        assert_eq!(recent[1].value, 200.0);

        // Get more than available
        let recent = fortress.get_recent(10);
        assert_eq!(recent.len(), 3);
    }
}
