-- Migration: 002_create_portfolio_tables.sql
-- Description: Create portfolio and positions tables for risk management
-- Created: Week 7

-- Portfolio table - stores portfolio configurations and state
CREATE TABLE IF NOT EXISTS portfolios (
    -- Primary identification
    portfolio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Portfolio metadata
    name VARCHAR(100) NOT NULL,
    account_id VARCHAR(100) NOT NULL,

    -- Account information
    initial_balance DECIMAL(20, 8) NOT NULL,
    current_balance DECIMAL(20, 8) NOT NULL,

    -- Performance tracking
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    total_pnl_percentage DECIMAL(10, 4) DEFAULT 0,
    daily_pnl DECIMAL(20, 8) DEFAULT 0,
    daily_pnl_percentage DECIMAL(10, 4) DEFAULT 0,

    -- Risk metrics
    max_drawdown DECIMAL(10, 4) DEFAULT 0,
    current_drawdown DECIMAL(10, 4) DEFAULT 0,
    sharpe_ratio DECIMAL(10, 4),

    -- Exposure tracking
    total_exposure DECIMAL(20, 8) DEFAULT 0,
    exposure_percentage DECIMAL(10, 4) DEFAULT 0,

    -- Position counts
    active_positions INT DEFAULT 0,
    total_positions_opened INT DEFAULT 0,
    winning_positions INT DEFAULT 0,
    losing_positions INT DEFAULT 0,

    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_reset_at TIMESTAMPTZ,

    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'archived')),

    -- Configuration (JSONB for flexibility)
    risk_config JSONB,

    UNIQUE(account_id, name)
);

-- Positions table - stores individual trading positions
CREATE TABLE IF NOT EXISTS positions (
    -- Primary identification
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign keys
    portfolio_id UUID NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(signal_id),

    -- Position metadata
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('Long', 'Short')),

    -- Entry information
    entry_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    position_value DECIMAL(20, 8) NOT NULL,

    -- Risk parameters
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    risk_amount DECIMAL(20, 8),
    risk_percentage DECIMAL(10, 4),
    risk_reward_ratio DECIMAL(10, 4),

    -- Current state
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_percentage DECIMAL(10, 4),

    -- Timing
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,

    -- Exit information (filled when position is closed)
    exit_price DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    realized_pnl_percentage DECIMAL(10, 4),

    -- Exit reason
    exit_reason VARCHAR(50) CHECK (exit_reason IN ('stop_loss', 'take_profit', 'manual', 'trailing_stop', 'time_exit', 'signal')),

    -- Status
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'closed', 'partially_closed')),

    -- Additional metadata
    metadata JSONB,

    -- Notes
    notes TEXT
);

-- Position updates table - tracks all position updates for audit trail
CREATE TABLE IF NOT EXISTS position_updates (
    update_id BIGSERIAL PRIMARY KEY,
    position_id UUID NOT NULL REFERENCES positions(position_id) ON DELETE CASCADE,

    -- Update information
    update_type VARCHAR(20) NOT NULL CHECK (update_type IN ('open', 'update', 'close', 'stop_update', 'tp_update')),

    -- Price at update
    price DECIMAL(20, 8) NOT NULL,

    -- Updated fields
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),

    -- PnL at update time
    unrealized_pnl DECIMAL(20, 8),
    unrealized_pnl_percentage DECIMAL(10, 4),

    -- Timing
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB
);

-- Portfolio snapshots - daily snapshots for performance tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    portfolio_id UUID NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,

    -- Snapshot date
    snapshot_date DATE NOT NULL,
    snapshot_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Balance information
    balance DECIMAL(20, 8) NOT NULL,
    equity DECIMAL(20, 8) NOT NULL,

    -- Performance metrics
    daily_pnl DECIMAL(20, 8),
    daily_pnl_percentage DECIMAL(10, 4),
    cumulative_pnl DECIMAL(20, 8),
    cumulative_pnl_percentage DECIMAL(10, 4),

    -- Risk metrics
    exposure DECIMAL(20, 8),
    exposure_percentage DECIMAL(10, 4),
    drawdown DECIMAL(10, 4),

    -- Position counts
    active_positions INT,

    -- Metadata
    metadata JSONB,

    UNIQUE(portfolio_id, snapshot_date)
);

-- Indexes for portfolios
CREATE INDEX IF NOT EXISTS idx_portfolios_account_id ON portfolios(account_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios(status);
CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at DESC);

-- Indexes for positions
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_positions_signal_id ON positions(signal_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_closed_at ON positions(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status ON positions(portfolio_id, status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);

-- Indexes for position updates
CREATE INDEX IF NOT EXISTS idx_position_updates_position_id ON position_updates(position_id);
CREATE INDEX IF NOT EXISTS idx_position_updates_updated_at ON position_updates(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_position_updates_type ON position_updates(update_type);

-- Indexes for snapshots
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_portfolio_id ON portfolio_snapshots(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_portfolio_date ON portfolio_snapshots(portfolio_id, snapshot_date DESC);

-- Function to update portfolio updated_at timestamp
CREATE OR REPLACE FUNCTION update_portfolio_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to update position updated_at timestamp
CREATE OR REPLACE FUNCTION update_position_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
DROP TRIGGER IF EXISTS portfolio_updated_at_trigger ON portfolios;
CREATE TRIGGER portfolio_updated_at_trigger
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION update_portfolio_updated_at();

DROP TRIGGER IF EXISTS position_updated_at_trigger ON positions;
CREATE TRIGGER position_updated_at_trigger
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_position_updated_at();

-- Function to automatically update portfolio metrics when positions change
CREATE OR REPLACE FUNCTION update_portfolio_metrics()
RETURNS TRIGGER AS $$
DECLARE
    v_total_exposure DECIMAL(20, 8);
    v_active_count INT;
    v_total_unrealized_pnl DECIMAL(20, 8);
BEGIN
    -- Calculate current metrics for the portfolio
    SELECT
        COALESCE(SUM(position_value), 0),
        COUNT(*),
        COALESCE(SUM(unrealized_pnl), 0)
    INTO
        v_total_exposure,
        v_active_count,
        v_total_unrealized_pnl
    FROM positions
    WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
    AND status = 'open';

    -- Update portfolio
    UPDATE portfolios
    SET
        total_exposure = v_total_exposure,
        exposure_percentage = CASE
            WHEN current_balance > 0 THEN (v_total_exposure / current_balance) * 100
            ELSE 0
        END,
        active_positions = v_active_count,
        updated_at = NOW()
    WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id);

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update portfolio metrics on position changes
DROP TRIGGER IF EXISTS position_metrics_trigger ON positions;
CREATE TRIGGER position_metrics_trigger
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_portfolio_metrics();

-- Comments
COMMENT ON TABLE portfolios IS 'Portfolio configurations and performance tracking';
COMMENT ON TABLE positions IS 'Individual trading positions';
COMMENT ON TABLE position_updates IS 'Audit trail of position modifications';
COMMENT ON TABLE portfolio_snapshots IS 'Daily snapshots of portfolio state for historical analysis';

COMMENT ON COLUMN positions.risk_reward_ratio IS 'Expected risk/reward ratio at position opening';
COMMENT ON COLUMN positions.unrealized_pnl IS 'Current unrealized profit/loss';
COMMENT ON COLUMN positions.realized_pnl IS 'Actual profit/loss after position close';
