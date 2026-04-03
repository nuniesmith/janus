-- Migration: 004_fix_decimal_to_float.sql
-- Description: Change DECIMAL columns to DOUBLE PRECISION for sqlx f64 compatibility
--
-- SQLx maps Rust f64 to PostgreSQL DOUBLE PRECISION (FLOAT8), not DECIMAL/NUMERIC.
-- This migration converts all decimal columns to DOUBLE PRECISION to fix type mismatches.

-- ============================================================================
-- Fix signals table
-- ============================================================================

ALTER TABLE signals
    ALTER COLUMN confidence TYPE DOUBLE PRECISION USING confidence::DOUBLE PRECISION,
    ALTER COLUMN strength TYPE DOUBLE PRECISION USING strength::DOUBLE PRECISION;

ALTER TABLE signals
    ALTER COLUMN model_confidence TYPE DOUBLE PRECISION USING model_confidence::DOUBLE PRECISION;

ALTER TABLE signals
    ALTER COLUMN strategy_score TYPE DOUBLE PRECISION USING strategy_score::DOUBLE PRECISION;

-- ============================================================================
-- Fix signal_performance table
-- ============================================================================

ALTER TABLE signal_performance
    ALTER COLUMN signal_confidence TYPE DOUBLE PRECISION USING signal_confidence::DOUBLE PRECISION,
    ALTER COLUMN signal_strength TYPE DOUBLE PRECISION USING signal_strength::DOUBLE PRECISION;

-- ============================================================================
-- Fix strategy_performance table
-- ============================================================================

ALTER TABLE strategy_performance
    ALTER COLUMN avg_confidence TYPE DOUBLE PRECISION USING avg_confidence::DOUBLE PRECISION,
    ALTER COLUMN avg_strength TYPE DOUBLE PRECISION USING avg_strength::DOUBLE PRECISION;

-- ============================================================================
-- Fix performance_stats table (if it has decimal columns)
-- ============================================================================

ALTER TABLE performance_stats
    ALTER COLUMN win_rate TYPE DOUBLE PRECISION USING win_rate::DOUBLE PRECISION,
    ALTER COLUMN loss_rate TYPE DOUBLE PRECISION USING loss_rate::DOUBLE PRECISION,
    ALTER COLUMN avg_win TYPE DOUBLE PRECISION USING avg_win::DOUBLE PRECISION,
    ALTER COLUMN avg_loss TYPE DOUBLE PRECISION USING avg_loss::DOUBLE PRECISION,
    ALTER COLUMN avg_win_pct TYPE DOUBLE PRECISION USING avg_win_pct::DOUBLE PRECISION,
    ALTER COLUMN avg_loss_pct TYPE DOUBLE PRECISION USING avg_loss_pct::DOUBLE PRECISION,
    ALTER COLUMN largest_win_pct TYPE DOUBLE PRECISION USING largest_win_pct::DOUBLE PRECISION,
    ALTER COLUMN largest_loss_pct TYPE DOUBLE PRECISION USING largest_loss_pct::DOUBLE PRECISION,
    ALTER COLUMN profit_factor TYPE DOUBLE PRECISION USING profit_factor::DOUBLE PRECISION,
    ALTER COLUMN expectancy TYPE DOUBLE PRECISION USING expectancy::DOUBLE PRECISION,
    ALTER COLUMN expectancy_pct TYPE DOUBLE PRECISION USING expectancy_pct::DOUBLE PRECISION,
    ALTER COLUMN sharpe_ratio TYPE DOUBLE PRECISION USING sharpe_ratio::DOUBLE PRECISION,
    ALTER COLUMN sortino_ratio TYPE DOUBLE PRECISION USING sortino_ratio::DOUBLE PRECISION,
    ALTER COLUMN calmar_ratio TYPE DOUBLE PRECISION USING calmar_ratio::DOUBLE PRECISION,
    ALTER COLUMN max_drawdown TYPE DOUBLE PRECISION USING max_drawdown::DOUBLE PRECISION,
    ALTER COLUMN avg_drawdown TYPE DOUBLE PRECISION USING avg_drawdown::DOUBLE PRECISION,
    ALTER COLUMN recovery_factor TYPE DOUBLE PRECISION USING recovery_factor::DOUBLE PRECISION,
    ALTER COLUMN avg_r_multiple TYPE DOUBLE PRECISION USING avg_r_multiple::DOUBLE PRECISION;

-- ============================================================================
-- Fix risk_metrics table
-- ============================================================================

ALTER TABLE risk_metrics
    ALTER COLUMN exposure_percentage TYPE DOUBLE PRECISION USING exposure_percentage::DOUBLE PRECISION,
    ALTER COLUMN largest_position_pct TYPE DOUBLE PRECISION USING largest_position_pct::DOUBLE PRECISION,
    ALTER COLUMN top_5_concentration_pct TYPE DOUBLE PRECISION USING top_5_concentration_pct::DOUBLE PRECISION,
    ALTER COLUMN avg_correlation TYPE DOUBLE PRECISION USING avg_correlation::DOUBLE PRECISION,
    ALTER COLUMN max_correlation TYPE DOUBLE PRECISION USING max_correlation::DOUBLE PRECISION,
    ALTER COLUMN portfolio_heat TYPE DOUBLE PRECISION USING portfolio_heat::DOUBLE PRECISION,
    ALTER COLUMN portfolio_volatility TYPE DOUBLE PRECISION USING portfolio_volatility::DOUBLE PRECISION,
    ALTER COLUMN realized_volatility TYPE DOUBLE PRECISION USING realized_volatility::DOUBLE PRECISION,
    ALTER COLUMN current_drawdown TYPE DOUBLE PRECISION USING current_drawdown::DOUBLE PRECISION,
    ALTER COLUMN drawdown_from_peak TYPE DOUBLE PRECISION USING drawdown_from_peak::DOUBLE PRECISION;

-- ============================================================================
-- Add comments documenting the change
-- ============================================================================

COMMENT ON COLUMN signals.confidence IS 'Signal confidence score (0.0 to 1.0) - DOUBLE PRECISION for sqlx compatibility';
COMMENT ON COLUMN signals.strength IS 'Signal strength score (0.0 to 1.0) - DOUBLE PRECISION for sqlx compatibility';
