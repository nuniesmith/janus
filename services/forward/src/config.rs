//! Forward-loop execution tunables that were formerly bare literals at the
//! live signal-generation call site.
//!
//! Naming them here does two things and changes no behaviour:
//!   1. Documents the constants the Gate-A measurement pivots on, so a future
//!      reader knows the number is load-bearing rather than incidental.
//!   2. Makes them env-overridable for a *deliberately* re-baselined measurement
//!      window — WITHOUT altering the production default, the comparison
//!      operator, or the tie behaviour when the override is unset.
//!
//! Prime directive for this module: with the environment unset, every resolved
//! value is bit-identical to the literal it replaced.

/// The execution floor: the consensus confidence at/above which an actionable
/// Buy/Sell is submitted to the execution service. Strictly below it, the
/// decision is still *recorded* — it was that Buy/Sell — but tagged
/// `below_exec_floor` and never submitted.
///
/// This is the 0.7 "guillotine" the Gate-A long-skew analysis pivots on: above
/// it the stream is ~99% long, below it it is nearly balanced
/// (GATE_A_PREREGISTRATION §4g). Whether that guillotine is right or wrong is a
/// measurement question, so the constant is named and overridable — but its
/// production value MUST stay 0.7. Changing it is a regime change, not
/// instrumentation.
pub const DEFAULT_EXEC_FLOOR: f64 = 0.7;

/// Environment variable overriding [`DEFAULT_EXEC_FLOOR`]. Unset ⇒ exactly the
/// default (bit-identical to the former literal). Set it ONLY to open a new,
/// deliberately re-baselined measurement window.
pub const EXEC_FLOOR_ENV: &str = "JANUS_EXEC_FLOOR";

/// Resolved forward-loop floors. `Copy` and allocation-free: resolved once at
/// loop start, logged, then compared per candle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardFloors {
    /// Confidence floor for the execution submit. The live-loop comparison is
    /// `avg_confidence >= exec_floor` (inclusive — a confidence *exactly* on the
    /// floor is submittable), matching the original `avg_confidence >= 0.7`.
    pub exec_floor: f64,
}

impl Default for ForwardFloors {
    fn default() -> Self {
        Self {
            exec_floor: DEFAULT_EXEC_FLOOR,
        }
    }
}

impl ForwardFloors {
    /// Resolve from the process environment. Thin wrapper over [`resolve`] so
    /// the parsing logic can be tested without mutating global env state.
    ///
    /// [`resolve`]: ForwardFloors::resolve
    pub fn from_env() -> Self {
        Self::resolve(std::env::var(EXEC_FLOOR_ENV).ok().as_deref())
    }

    /// Pure resolver used by [`from_env`] and the tests.
    ///
    /// `None` (unset) ⇒ the frozen defaults. A finite, non-negative parse
    /// overrides the default; anything else — a malformed number, a negative,
    /// `NaN`, or `inf` — is *ignored* and falls back to the default, so a typo
    /// in the env can never silently move the floor.
    ///
    /// [`from_env`]: ForwardFloors::from_env
    pub fn resolve(exec_floor_raw: Option<&str>) -> Self {
        let exec_floor = exec_floor_raw
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(DEFAULT_EXEC_FLOOR);
        Self { exec_floor }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_env_resolves_to_the_frozen_default() {
        assert_eq!(ForwardFloors::resolve(None).exec_floor, DEFAULT_EXEC_FLOOR);
        assert_eq!(ForwardFloors::default().exec_floor, 0.7_f64);
    }

    #[test]
    fn a_valid_override_is_applied() {
        assert_eq!(ForwardFloors::resolve(Some("0.85")).exec_floor, 0.85);
        assert_eq!(ForwardFloors::resolve(Some(" 0.5 ")).exec_floor, 0.5);
        assert_eq!(ForwardFloors::resolve(Some("0")).exec_floor, 0.0);
    }

    #[test]
    fn malformed_or_unsafe_overrides_fall_back_to_the_default() {
        for bad in ["", "garbage", "-1", "-0.1", "nan", "NaN", "inf", "1e999"] {
            assert_eq!(
                ForwardFloors::resolve(Some(bad)).exec_floor,
                DEFAULT_EXEC_FLOOR,
                "{bad:?} must not move the floor"
            );
        }
    }

    /// ZERO-CHANGE PROOF: with the environment unset, the NEW named-config
    /// comparison (`c >= default.exec_floor`) is bit-identical to the OLD
    /// literal comparison (`c >= 0.7`) for every input on a dense grid — the
    /// exact tie, both representable neighbours of 0.7, and hand-picked
    /// boundary decimals. If the default ever drifts off 0.7, the partition
    /// diverges somewhere in the grid and this fails.
    #[test]
    fn default_exec_floor_partition_is_bit_identical_to_legacy_070() {
        let floor = ForwardFloors::default().exec_floor;
        assert_eq!(floor, 0.7_f64, "the frozen default is the literal 0.7");

        let mut grid: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
        // The tie and its two nearest representable f64 neighbours — the only
        // place a `>=` vs `>` or a one-ULP drift would show up.
        grid.push(0.7_f64);
        grid.push(f64::from_bits(0.7_f64.to_bits() + 1)); // next-representable above
        grid.push(f64::from_bits(0.7_f64.to_bits() - 1)); // next-representable below
        grid.push(0.6999999999999999);
        grid.push(0.7000000000000001);
        for &c in &grid {
            assert_eq!(
                c >= floor,
                c >= 0.7_f64,
                "execution-floor partition diverged at c={c:?}"
            );
        }
    }
}
