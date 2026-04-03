-- Migration: 003_create_metrics_tables.sql
-- Description: Create performance metrics and analytics tables
-- Created: Week 7

-- Trade metrics table - detailed metrics for each trade
CREATE TABLE IF NOT EXISTS trade_metrics (
    metric_id BIGSERIAL PRIMARY KEY,

    -- Foreign keys
    position_id UUID NOT NULL REFERENCES positions(position_id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(signal_id),
    portfolio_id UUID NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,

    -- Trade identification
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,

    -- Entry/Exit
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,

    -- Risk metrics
    initial_risk DECIMAL(20, 8),
    actual_risk DECIMAL(20, 8),
    risk_reward_ratio DECIMAL(10, 4),

    -- Performance
    gross_pnl DECIMAL(20, 8) NOT NULL,
    gross_pnl_percentage DECIMAL(10, 4) NOT NULL,
    net_pnl DECIMAL(20, 8),
    net_pnl_percentage DECIMAL(10, 4),

    -- Fees/Costs
    commission DECIMAL(20, 8) DEFAULT 0,
    slippage DECIMAL(20, 8) DEFAULT 0,

    -- Holding duration
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL,
    holding_duration_seconds BIGINT,
    holding_duration_bars INT,

    -- Exit details
    exit_reason VARCHAR(50),
    hit_stop_loss BOOLEAN DEFAULT FALSE,
    hit_take_profit BOOLEAN DEFAULT FALSE,

    -- Market conditions at entry
    entry_volatility DECIMAL(10, 6),
    entry_atr DECIMAL(20, 8),
    entry_rsi DECIMAL(5, 2),

    -- Market conditions at exit
    exit_volatility DECIMAL(10, 6),
    exit_atr DECIMAL(20, 8),
    exit_rsi DECIMAL(5, 2),

    -- Maximum favorable/adverse excursion
    max_favorable_excursion DECIMAL(20, 8),
    max_favorable_excursion_pct DECIMAL(10, 4),
    max_adverse_excursion DECIMAL(20, 8),
    max_adverse_excursion_pct DECIMAL(10, 4),

    -- Additional metrics
    metadata JSONB,

    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Performance statistics table - aggregated performance metrics
CREATE TABLE IF NOT EXISTS performance_stats (
    stat_id BIGSERIAL PRIMARY KEY,

    -- Scope
    portfolio_id UUID NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
    symbol VARCHAR(50), -- NULL for portfolio-wide stats
    timeframe VARCHAR(20), -- e.g., 'daily', 'weekly', 'monthly', 'all_time'

    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    -- Trade counts
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    breakeven_trades INT DEFAULT 0,

    -- Win/Loss metrics
    win_rate DECIMAL(5, 4),
    loss_rate DECIMAL(5, 4),

    -- PnL metrics
    gross_profit DECIMAL(20, 8) DEFAULT 0,
    gross_loss DECIMAL(20, 8) DEFAULT 0,
    net_profit DECIMAL(20, 8) DEFAULT 0,

    -- Average metrics
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    avg_win_pct DECIMAL(10, 4),
    avg_loss_pct DECIMAL(10, 4),

    -- Best/Worst
    largest_win DECIMAL(20, 8),
    largest_loss DECIMAL(20, 8),
    largest_win_pct DECIMAL(10, 4),
    largest_loss_pct DECIMAL(10, 4),

    -- Consecutive streaks
    max_consecutive_wins INT,
    max_consecutive_losses INT,
    current_streak INT,
    current_streak_type VARCHAR(10), -- 'win' or 'loss'

    -- Profit factor
    profit_factor DECIMAL(10, 4),

    -- Expectancy
    expectancy DECIMAL(20, 8),
    expectancy_pct DECIMAL(10, 4),

    -- Risk metrics
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    calmar_ratio DECIMAL(10, 4),

    -- Drawdown metrics
    max_drawdown DECIMAL(10, 4),
    max_drawdown_duration_days INT,
    avg_drawdown DECIMAL(10, 4),

    -- Recovery metrics
    recovery_factor DECIMAL(10, 4),

    -- R-multiples
    avg_r_multiple DECIMAL(10, 4),

    -- Holding time
    avg_holding_duration_seconds BIGINT,
    median_holding_duration_seconds BIGINT,

    -- Additional statistics
    total_volume DECIMAL(20, 8),
    total_commission DECIMAL(20, 8),

    -- Metadata
    metadata JSONB,

    -- Timing
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(portfolio_id, symbol, timeframe, period_start)
);

-- Risk metrics table - periodic risk assessments
CREATE TABLE IF NOT EXISTS risk_metrics (
    risk_id BIGSERIAL PRIMARY KEY,

    -- Scope
    portfolio_id UUID NOT NULL REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,

    -- Timestamp
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Portfolio metrics
    portfolio_value DECIMAL(20, 8) NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,

    -- Exposure metrics
    total_exposure DECIMAL(20, 8),
    exposure_percentage DECIMAL(10, 4),
    gross_exposure DECIMAL(20, 8),
    net_exposure DECIMAL(20, 8),

    -- Position metrics
    active_positions INT DEFAULT 0,
    max_position_size DECIMAL(20, 8),
    avg_position_size DECIMAL(20, 8),

    -- Concentration risk
    largest_position_pct DECIMAL(10, 4),
    top_5_concentration_pct DECIMAL(10, 4),

    -- Correlation risk
    avg_correlation DECIMAL(5, 4),
    max_correlation DECIMAL(5, 4),

    -- Portfolio heat (total risk as % of portfolio)
    portfolio_heat DECIMAL(10, 4),

    -- Individual position risks
    total_risk_amount DECIMAL(20, 8),
    avg_risk_per_position DECIMAL(20, 8),
    max_risk_per_position DECIMAL(20, 8),

    -- VaR (Value at Risk)
    var_95 DECIMAL(20, 8),
    var_99 DECIMAL(20, 8),
    cvar_95 DECIMAL(20, 8), -- Conditional VaR

    -- Volatility
    portfolio_volatility DECIMAL(10, 6),
    realized_volatility DECIMAL(10, 6),

    -- Drawdown
    current_drawdown DECIMAL(10, 4),
    drawdown_from_peak DECIMAL(20, 8),
    peak_portfolio_value DECIMAL(20, 8),

    -- Risk limits status
    limits_exceeded JSONB, -- Array of limit violations

    -- Metadata
    metadata JSONB
);

-- Signal performance tracking - tracks how signals performed
CREATE TABLE IF NOT EXISTS signal_performance (
    performance_id BIGSERIAL PRIMARY KEY,

    -- Signal reference
    signal_id UUID NOT NULL REFERENCES signals(signal_id) ON DELETE CASCADE,
    position_id UUID REFERENCES positions(position_id),

    -- Signal details
    symbol VARCHAR(50) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,

    -- Signal quality
    signal_confidence DECIMAL(5, 4),
    signal_strength DECIMAL(5, 4),

    -- Source information
    source_type VARCHAR(50),
    strategy_name VARCHAR(100),
    model_name VARCHAR(100),

    -- Performance outcome
    was_executed BOOLEAN DEFAULT FALSE,
    execution_delay_seconds INT,

    -- If executed
    actual_entry_price DECIMAL(20, 8),
    slippage DECIMAL(20, 8),
    slippage_pct DECIMAL(10, 4),

    -- Result
    outcome VARCHAR(20), -- 'win', 'loss', 'breakeven', 'pending', 'not_executed'
    pnl DECIMAL(20, 8),
    pnl_percentage DECIMAL(10, 4),

    -- Accuracy metrics
    entry_accuracy DECIMAL(10, 4), -- How close was entry to signal price
    stop_hit BOOLEAN,
    target_hit BOOLEAN,

    -- Timing
    signal_timestamp TIMESTAMPTZ NOT NULL,
    execution_timestamp TIMESTAMPTZ,
    exit_timestamp TIMESTAMPTZ,

    -- Metadata
    metadata JSONB,

    -- Created
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Strategy performance - aggregate metrics per strategy
CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_perf_id BIGSERIAL PRIMARY KEY,

    -- Strategy identification
    strategy_name VARCHAR(100) NOT NULL,
    portfolio_id UUID REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,

    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(20),

    -- Signal metrics
    total_signals INT DEFAULT 0,
    signals_executed INT DEFAULT 0,
    signals_filtered INT DEFAULT 0,
    execution_rate DECIMAL(5, 4),

    -- Performance metrics
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    win_rate DECIMAL(5, 4),

    -- PnL
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    avg_pnl DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),

    -- Quality metrics
    avg_confidence DECIMAL(5, 4),
    avg_strength DECIMAL(5, 4),

    -- Risk-adjusted returns
    sharpe_ratio DECIMAL(10, 4),

    -- Additional metrics
    metadata JSONB,

    -- Timing
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(strategy_name, portfolio_id, period_start, timeframe)
);

-- Indexes for trade_metrics
CREATE INDEX IF NOT EXISTS idx_trade_metrics_position_id ON trade_metrics(position_id);
CREATE INDEX IF NOT EXISTS idx_trade_metrics_signal_id ON trade_metrics(signal_id);
CREATE INDEX IF NOT EXISTS idx_trade_metrics_portfolio_id ON trade_metrics(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_trade_metrics_symbol ON trade_metrics(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_metrics_entry_time ON trade_metrics(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trade_metrics_exit_time ON trade_metrics(exit_time DESC);

-- Indexes for performance_stats
CREATE INDEX IF NOT EXISTS idx_performance_stats_portfolio_id ON performance_stats(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_performance_stats_symbol ON performance_stats(symbol);
CREATE INDEX IF NOT EXISTS idx_performance_stats_timeframe ON performance_stats(timeframe);
CREATE INDEX IF NOT EXISTS idx_performance_stats_period ON performance_stats(period_start DESC, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_performance_stats_portfolio_period ON performance_stats(portfolio_id, period_start DESC);

-- Indexes for risk_metrics
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio_id ON risk_metrics(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_measured_at ON risk_metrics(measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio_time ON risk_metrics(portfolio_id, measured_at DESC);

-- Indexes for signal_performance
CREATE INDEX IF NOT EXISTS idx_signal_performance_signal_id ON signal_performance(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_performance_position_id ON signal_performance(position_id);
CREATE INDEX IF NOT EXISTS idx_signal_performance_symbol ON signal_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_signal_performance_strategy ON signal_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_signal_performance_timestamp ON signal_performance(signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signal_performance_outcome ON signal_performance(outcome);

-- Indexes for strategy_performance
CREATE INDEX IF NOT EXISTS idx_strategy_performance_name ON strategy_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_portfolio ON strategy_performance(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_period ON strategy_performance(period_start DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_name_period ON strategy_performance(strategy_name, period_start DESC);

-- Comments
COMMENT ON TABLE trade_metrics IS 'Detailed metrics for each completed trade';
COMMENT ON TABLE performance_stats IS 'Aggregated performance statistics by time period';
COMMENT ON TABLE risk_metrics IS 'Portfolio risk metrics snapshots';
COMMENT ON TABLE signal_performance IS 'Track performance of generated signals';
COMMENT ON TABLE strategy_performance IS 'Aggregate performance metrics per strategy';

COMMENT ON COLUMN trade_metrics.max_favorable_excursion IS 'Maximum profit reached during the trade';
COMMENT ON COLUMN trade_metrics.max_adverse_excursion IS 'Maximum loss reached during the trade';
COMMENT ON COLUMN performance_stats.sharpe_ratio IS 'Risk-adjusted return metric';
COMMENT ON COLUMN performance_stats.profit_factor IS 'Gross profit divided by gross loss';
COMMENT ON COLUMN risk_metrics.portfolio_heat IS 'Total risk as percentage of portfolio value';
COMMENT ON COLUMN risk_metrics.var_95 IS 'Value at Risk at 95% confidence level';
