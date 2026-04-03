//! Discord Webhook Notifications
//!
//! This module provides Discord webhook integration for sending
//! trading notifications to Discord channels.
//!
//! Features:
//! - Trade signal notifications with SL/TP levels
//! - Order fill notifications with P&L
//! - Position updates
//! - Error notifications
//! - Bot lifecycle notifications (started/stopped)
//! - Daily summaries

use crate::error::{ExecutionError, Result};
use crate::types::{Fill, Order, OrderSide, Position};
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use tracing::{debug, error, info, warn};

/// Discord webhook notifier
pub struct DiscordNotifier {
    webhook_url: String,
    client: Client,
    enabled: bool,
    notify_on_signal: bool,
    notify_on_fill: bool,
    notify_on_error: bool,
    /// Default stop-loss percentage for signal notifications
    stop_loss_pct: f64,
    /// Default take-profit percentage for signal notifications
    take_profit_pct: f64,
    /// Bot name for footer
    bot_name: String,
}

impl DiscordNotifier {
    /// Create a new Discord notifier
    pub fn new(
        webhook_url: String,
        enabled: bool,
        notify_on_signal: bool,
        notify_on_fill: bool,
        notify_on_error: bool,
    ) -> Self {
        Self {
            webhook_url,
            client: Client::new(),
            enabled,
            notify_on_signal,
            notify_on_fill,
            notify_on_error,
            stop_loss_pct: 2.0,
            take_profit_pct: 5.0,
            bot_name: "FKS Execution Service".to_string(),
        }
    }

    /// Create a new Discord notifier with custom SL/TP percentages
    pub fn with_risk_params(
        webhook_url: String,
        enabled: bool,
        notify_on_signal: bool,
        notify_on_fill: bool,
        notify_on_error: bool,
        stop_loss_pct: f64,
        take_profit_pct: f64,
    ) -> Self {
        Self {
            webhook_url,
            client: Client::new(),
            enabled,
            notify_on_signal,
            notify_on_fill,
            notify_on_error,
            stop_loss_pct,
            take_profit_pct,
            bot_name: "FKS Execution Service".to_string(),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        let webhook_url = std::env::var("DISCORD_WEBHOOK_GENERAL").unwrap_or_else(|_| String::new());
        let enabled = std::env::var("DISCORD_ENABLE_NOTIFICATIONS")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);
        let notify_on_signal = std::env::var("DISCORD_NOTIFY_ON_SIGNAL")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);
        let notify_on_fill = std::env::var("DISCORD_NOTIFY_ON_FILL")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);
        let notify_on_error = std::env::var("DISCORD_NOTIFY_ON_ERROR")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);
        let stop_loss_pct = std::env::var("DISCORD_DEFAULT_STOP_LOSS_PCT")
            .unwrap_or_else(|_| "2.0".to_string())
            .parse()
            .unwrap_or(2.0);
        let take_profit_pct = std::env::var("DISCORD_DEFAULT_TAKE_PROFIT_PCT")
            .unwrap_or_else(|_| "5.0".to_string())
            .parse()
            .unwrap_or(5.0);
        let bot_name = std::env::var("DISCORD_BOT_NAME")
            .unwrap_or_else(|_| "FKS Execution Service".to_string());

        Self {
            webhook_url,
            client: Client::new(),
            enabled,
            notify_on_signal,
            notify_on_fill,
            notify_on_error,
            stop_loss_pct,
            take_profit_pct,
            bot_name,
        }
    }

    /// Check if notifier is enabled and configured
    pub fn is_enabled(&self) -> bool {
        self.enabled && !self.webhook_url.is_empty()
    }

    /// Notify that a trading signal was received (enhanced with SL/TP levels)
    #[allow(clippy::too_many_arguments)]
    pub async fn notify_signal_received(
        &self,
        signal_id: &str,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        price: Option<f64>,
        confidence: f64,
        strategy_id: &str,
    ) -> Result<()> {
        if !self.is_enabled() || !self.notify_on_signal {
            return Ok(());
        }

        debug!("Sending signal notification to Discord: {}", signal_id);

        // If we have a price, create enhanced embed with SL/TP
        if let Some(entry_price) = price {
            return self
                .notify_signal_with_levels(
                    signal_id,
                    symbol,
                    side,
                    quantity,
                    entry_price,
                    confidence,
                    strategy_id,
                    self.stop_loss_pct,
                    self.take_profit_pct,
                )
                .await;
        }

        // Fallback for market orders without price
        let side_emoji = match side {
            OrderSide::Buy => "🟢",
            OrderSide::Sell => "🔴",
        };

        let embed = DiscordEmbed {
            title: format!("{} Trading Signal Received", side_emoji),
            description: Some(format!(
                "New trading signal from strategy `{}`",
                strategy_id
            )),
            color: match side {
                OrderSide::Buy => 3066993,   // Green
                OrderSide::Sell => 15158332, // Red
            },
            fields: vec![
                EmbedField {
                    name: "Symbol".to_string(),
                    value: symbol.to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "Side".to_string(),
                    value: format!("{:?}", side).to_uppercase(),
                    inline: true,
                },
                EmbedField {
                    name: "Quantity".to_string(),
                    value: format!("{:.8}", quantity),
                    inline: true,
                },
                EmbedField {
                    name: "Price".to_string(),
                    value: "Market".to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "Confidence".to_string(),
                    value: format!("{:.1}%", confidence * 100.0),
                    inline: true,
                },
                EmbedField {
                    name: "Signal ID".to_string(),
                    value: signal_id.to_string(),
                    inline: false,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify trading signal with SL/TP levels for manual execution
    #[allow(clippy::too_many_arguments)]
    pub async fn notify_signal_with_levels(
        &self,
        signal_id: &str,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        entry_price: f64,
        confidence: f64,
        strategy_id: &str,
        stop_loss_pct: f64,
        take_profit_pct: f64,
    ) -> Result<()> {
        if !self.is_enabled() || !self.notify_on_signal {
            return Ok(());
        }

        debug!(
            "Sending enhanced signal notification to Discord: {}",
            signal_id
        );

        let total_value = entry_price * quantity;

        // Calculate SL/TP based on side
        let (stop_loss_price, take_profit_price) = match side {
            OrderSide::Buy => {
                // Long position: SL below entry, TP above entry
                let sl = entry_price * (1.0 - stop_loss_pct / 100.0);
                let tp = entry_price * (1.0 + take_profit_pct / 100.0);
                (sl, tp)
            }
            OrderSide::Sell => {
                // Short position: SL above entry, TP below entry
                let sl = entry_price * (1.0 + stop_loss_pct / 100.0);
                let tp = entry_price * (1.0 - take_profit_pct / 100.0);
                (sl, tp)
            }
        };

        // Calculate risk/reward
        let risk_amount = total_value * (stop_loss_pct / 100.0);
        let reward_amount = total_value * (take_profit_pct / 100.0);
        let risk_reward_ratio = take_profit_pct / stop_loss_pct;

        let side_emoji = match side {
            OrderSide::Buy => "🟢",
            OrderSide::Sell => "🔴",
        };

        let side_name = match side {
            OrderSide::Buy => "BUY (LONG)",
            OrderSide::Sell => "SELL (SHORT)",
        };

        let sl_direction = match side {
            OrderSide::Buy => format!("-{:.1}%", stop_loss_pct),
            OrderSide::Sell => format!("+{:.1}%", stop_loss_pct),
        };

        let tp_direction = match side {
            OrderSide::Buy => format!("+{:.1}%", take_profit_pct),
            OrderSide::Sell => format!("-{:.1}%", take_profit_pct),
        };

        let execution_steps = match side {
            OrderSide::Buy => {
                "1️⃣ Enter LONG at entry price\n2️⃣ Set Stop Loss immediately\n3️⃣ Set Take Profit immediately\n4️⃣ Log trade in journal"
            }
            OrderSide::Sell => {
                "1️⃣ Enter SHORT at entry price\n2️⃣ Set Stop Loss immediately\n3️⃣ Set Take Profit immediately\n4️⃣ Log trade in journal"
            }
        };

        let embed = DiscordEmbed {
            title: format!("{} {} Signal", side_emoji, side_name),
            description: Some(format!(
                "**{}**\nStrategy: `{}`\nConfidence: **{:.1}%**",
                symbol,
                strategy_id,
                confidence * 100.0
            )),
            color: match side {
                OrderSide::Buy => 65280,     // Bright green
                OrderSide::Sell => 16711680, // Bright red
            },
            fields: vec![
                EmbedField {
                    name: "📍 Entry Price".to_string(),
                    value: format!("**${:.2}**", entry_price),
                    inline: true,
                },
                EmbedField {
                    name: "📦 Position Size".to_string(),
                    value: format!("**${:.2}**", total_value),
                    inline: true,
                },
                EmbedField {
                    name: "📊 Quantity".to_string(),
                    value: format!("{:.8}", quantity),
                    inline: true,
                },
                EmbedField {
                    name: "🛑 Stop Loss".to_string(),
                    value: format!("**${:.2}**\n({})", stop_loss_price, sl_direction),
                    inline: true,
                },
                EmbedField {
                    name: "🎯 Take Profit".to_string(),
                    value: format!("**${:.2}**\n({})", take_profit_price, tp_direction),
                    inline: true,
                },
                EmbedField {
                    name: "⚖️ Risk/Reward".to_string(),
                    value: format!("**1:{:.1}**", risk_reward_ratio),
                    inline: true,
                },
                EmbedField {
                    name: "💰 Potential Profit".to_string(),
                    value: format!("+${:.2}", reward_amount),
                    inline: true,
                },
                EmbedField {
                    name: "⚠️ Potential Loss".to_string(),
                    value: format!("-${:.2}", risk_amount),
                    inline: true,
                },
                EmbedField {
                    name: "🔑 Signal ID".to_string(),
                    value: signal_id.to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "⏰ Execution Steps".to_string(),
                    value: execution_steps.to_string(),
                    inline: false,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: format!("{} • Manual Execution", self.bot_name),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify that an order was filled
    pub async fn notify_order_filled(
        &self,
        order: &Order,
        fill: &Fill,
        position: Option<&Position>,
    ) -> Result<()> {
        if !self.is_enabled() || !self.notify_on_fill {
            return Ok(());
        }

        debug!("Sending order fill notification to Discord: {}", order.id);

        let side_emoji = match order.side {
            OrderSide::Buy => "✅",
            OrderSide::Sell => "💰",
        };

        let mut fields = vec![
            EmbedField {
                name: "Symbol".to_string(),
                value: order.symbol.clone(),
                inline: true,
            },
            EmbedField {
                name: "Side".to_string(),
                value: format!("{:?}", order.side).to_uppercase(),
                inline: true,
            },
            EmbedField {
                name: "Filled Quantity".to_string(),
                value: format!("{:.8}", fill.quantity),
                inline: true,
            },
            EmbedField {
                name: "Fill Price".to_string(),
                value: format!("${:.2}", fill.price),
                inline: true,
            },
            EmbedField {
                name: "Total Value".to_string(),
                value: format!("${:.2}", fill.quantity * fill.price),
                inline: true,
            },
            EmbedField {
                name: "Fee".to_string(),
                value: format!("${:.4} {}", fill.fee, fill.fee_currency),
                inline: true,
            },
        ];

        // Add position info if available
        if let Some(pos) = position {
            let pnl_emoji = if pos.unrealized_pnl >= rust_decimal::Decimal::ZERO {
                "📈"
            } else {
                "📉"
            };

            fields.push(EmbedField {
                name: format!("{} Position P&L", pnl_emoji),
                value: format!(
                    "Unrealized: ${} | Realized: ${}",
                    pos.unrealized_pnl, pos.realized_pnl
                ),
                inline: false,
            });
        }

        fields.push(EmbedField {
            name: "Order ID".to_string(),
            value: order.id.clone(),
            inline: false,
        });

        let embed = DiscordEmbed {
            title: format!("{} Order Filled", side_emoji),
            description: Some("Order successfully executed".to_string()),
            color: match order.side {
                OrderSide::Buy => 3066993,   // Green
                OrderSide::Sell => 15158332, // Red
            },
            fields,
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: "FKS Execution Service".to_string(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify about a position update
    pub async fn notify_position_update(&self, position: &Position, reason: &str) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        debug!(
            "Sending position update notification to Discord: {}",
            position.symbol
        );

        let pnl_emoji = if position.unrealized_pnl >= rust_decimal::Decimal::ZERO {
            "📈"
        } else {
            "📉"
        };

        let embed = DiscordEmbed {
            title: format!("{} Position Update", pnl_emoji),
            description: Some(reason.to_string()),
            color: if position.unrealized_pnl >= rust_decimal::Decimal::ZERO {
                3066993 // Green
            } else {
                15158332 // Red
            },
            fields: vec![
                EmbedField {
                    name: "Symbol".to_string(),
                    value: position.symbol.clone(),
                    inline: true,
                },
                EmbedField {
                    name: "Side".to_string(),
                    value: format!("{:?}", position.side).to_uppercase(),
                    inline: true,
                },
                EmbedField {
                    name: "Quantity".to_string(),
                    value: format!("{:.8}", position.quantity),
                    inline: true,
                },
                EmbedField {
                    name: "Entry Price".to_string(),
                    value: format!("${}", position.average_entry_price),
                    inline: true,
                },
                EmbedField {
                    name: "Current Price".to_string(),
                    value: format!("${}", position.current_price),
                    inline: true,
                },
                EmbedField {
                    name: "Unrealized P&L".to_string(),
                    value: format!("${}", position.unrealized_pnl),
                    inline: true,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: "FKS Execution Service".to_string(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify about an error
    pub async fn notify_error(
        &self,
        error_type: &str,
        error_message: &str,
        context: &str,
    ) -> Result<()> {
        if !self.is_enabled() || !self.notify_on_error {
            return Ok(());
        }

        warn!("Sending error notification to Discord: {}", error_type);

        let embed = DiscordEmbed {
            title: "❌ Execution Error".to_string(),
            description: Some(format!("**{}**", error_type)),
            color: 15158332, // Red
            fields: vec![
                EmbedField {
                    name: "Error".to_string(),
                    value: error_message.to_string(),
                    inline: false,
                },
                EmbedField {
                    name: "Context".to_string(),
                    value: context.to_string(),
                    inline: false,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: "FKS Execution Service - Alert".to_string(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify about daily summary
    pub async fn notify_daily_summary(
        &self,
        total_trades: usize,
        winning_trades: usize,
        losing_trades: usize,
        total_pnl: f64,
        win_rate: f64,
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        debug!("Sending daily summary to Discord");

        let pnl_emoji = if total_pnl >= 0.0 { "💰" } else { "⚠️" };

        let embed = DiscordEmbed {
            title: format!("{} Daily Trading Summary", pnl_emoji),
            description: Some("End of day performance report".to_string()),
            color: if total_pnl >= 0.0 {
                3066993 // Green
            } else {
                15158332 // Red
            },
            fields: vec![
                EmbedField {
                    name: "Total Trades".to_string(),
                    value: total_trades.to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "Winning Trades".to_string(),
                    value: winning_trades.to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "Losing Trades".to_string(),
                    value: losing_trades.to_string(),
                    inline: true,
                },
                EmbedField {
                    name: "Win Rate".to_string(),
                    value: format!("{:.1}%", win_rate * 100.0),
                    inline: true,
                },
                EmbedField {
                    name: "Total P&L".to_string(),
                    value: format!("${:.2}", total_pnl),
                    inline: true,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify that the bot has started
    pub async fn notify_bot_started(&self, version: &str, dry_run: bool) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        info!("Sending bot started notification to Discord");

        let mode = if dry_run {
            "**🧪 DRY RUN MODE**"
        } else {
            "**🔴 LIVE TRADING**"
        };

        let embed = DiscordEmbed {
            title: "🤖 Trading Bot Started".to_string(),
            description: Some(format!("Version: {}\nMode: {}", version, mode)),
            color: 65280, // Green
            fields: vec![
                EmbedField {
                    name: "Default Stop Loss".to_string(),
                    value: format!("{:.1}%", self.stop_loss_pct),
                    inline: true,
                },
                EmbedField {
                    name: "Default Take Profit".to_string(),
                    value: format!("{:.1}%", self.take_profit_pct),
                    inline: true,
                },
                EmbedField {
                    name: "Started At".to_string(),
                    value: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    inline: true,
                },
            ],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify that the bot has stopped
    pub async fn notify_bot_stopped(&self, reason: &str) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        info!("Sending bot stopped notification to Discord");

        let embed = DiscordEmbed {
            title: "🛑 Trading Bot Stopped".to_string(),
            description: Some(format!("Reason: {}", reason)),
            color: 16711680, // Red
            fields: vec![EmbedField {
                name: "Stopped At".to_string(),
                value: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                inline: false,
            }],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify about a warning (non-critical issue)
    pub async fn notify_warning(&self, warning_msg: &str, context: &str) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        warn!("Sending warning notification to Discord: {}", warning_msg);

        let embed = DiscordEmbed {
            title: "⚠️ Warning".to_string(),
            description: Some(format!("**Context:** {}\n\n{}", context, warning_msg)),
            color: 16750848, // Orange
            fields: vec![],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Notify about a balance update
    pub async fn notify_balance_update(
        &self,
        balance: Decimal,
        change_pct: Option<f64>,
    ) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        debug!("Sending balance update notification to Discord");

        let color = if let Some(pct) = change_pct {
            if pct > 0.0 {
                65280 // Green for profit
            } else if pct < 0.0 {
                16711680 // Red for loss
            } else {
                8421504 // Gray for neutral
            }
        } else {
            3447003 // Blue for info
        };

        let description = if let Some(pct) = change_pct {
            format!(
                "Current Balance: **${:.2}**\nChange: **{:+.2}%**",
                balance, pct
            )
        } else {
            format!("Current Balance: **${:.2}**", balance)
        };

        let embed = DiscordEmbed {
            title: "💰 Account Balance Update".to_string(),
            description: Some(description),
            color,
            fields: vec![],
            timestamp: Some(Utc::now()),
            footer: Some(EmbedFooter {
                text: self.bot_name.clone(),
            }),
        };

        self.send_webhook(vec![embed]).await
    }

    /// Send a webhook message with embeds
    async fn send_webhook(&self, embeds: Vec<DiscordEmbed>) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let payload = DiscordWebhook { embeds };

        match self
            .client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    debug!("Discord notification sent successfully");
                    Ok(())
                } else {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    error!(
                        "Discord webhook failed with status {}: {}",
                        status, error_text
                    );
                    Err(ExecutionError::Internal(format!(
                        "Discord webhook failed: {}",
                        status
                    )))
                }
            }
            Err(e) => {
                error!("Failed to send Discord notification: {}", e);
                Err(ExecutionError::Internal(format!(
                    "Failed to send Discord notification: {}",
                    e
                )))
            }
        }
    }

    /// Set the default stop-loss percentage
    pub fn set_stop_loss_pct(&mut self, pct: f64) {
        self.stop_loss_pct = pct;
    }

    /// Set the default take-profit percentage
    pub fn set_take_profit_pct(&mut self, pct: f64) {
        self.take_profit_pct = pct;
    }

    /// Set the bot name for footer
    pub fn set_bot_name(&mut self, name: String) {
        self.bot_name = name;
    }

    /// Get current stop-loss percentage
    pub fn stop_loss_pct(&self) -> f64 {
        self.stop_loss_pct
    }

    /// Get current take-profit percentage
    pub fn take_profit_pct(&self) -> f64 {
        self.take_profit_pct
    }
}

/// Discord webhook payload
#[derive(Debug, Serialize, Deserialize)]
struct DiscordWebhook {
    embeds: Vec<DiscordEmbed>,
}

/// Discord embed
#[derive(Debug, Serialize, Deserialize)]
struct DiscordEmbed {
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    color: u32,
    fields: Vec<EmbedField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    footer: Option<EmbedFooter>,
}

/// Discord embed field
#[derive(Debug, Serialize, Deserialize)]
struct EmbedField {
    name: String,
    value: String,
    inline: bool,
}

/// Discord embed footer
#[derive(Debug, Serialize, Deserialize)]
struct EmbedFooter {
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notifier_creation() {
        let notifier = DiscordNotifier::new(
            "https://discord.com/api/webhooks/test".to_string(),
            true,
            true,
            true,
            true,
        );

        assert!(notifier.is_enabled());
        assert_eq!(notifier.stop_loss_pct(), 2.0);
        assert_eq!(notifier.take_profit_pct(), 5.0);
    }

    #[test]
    fn test_notifier_with_risk_params() {
        let notifier = DiscordNotifier::with_risk_params(
            "https://discord.com/api/webhooks/test".to_string(),
            true,
            true,
            true,
            true,
            3.0,
            6.0,
        );

        assert!(notifier.is_enabled());
        assert_eq!(notifier.stop_loss_pct(), 3.0);
        assert_eq!(notifier.take_profit_pct(), 6.0);
    }

    #[test]
    fn test_notifier_disabled() {
        let notifier = DiscordNotifier::new(String::new(), false, true, true, true);

        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_notifier_no_webhook() {
        let notifier = DiscordNotifier::new(String::new(), true, true, true, true);

        assert!(!notifier.is_enabled());
    }

    #[test]
    fn test_set_risk_params() {
        let mut notifier = DiscordNotifier::new(
            "https://discord.com/api/webhooks/test".to_string(),
            true,
            true,
            true,
            true,
        );

        notifier.set_stop_loss_pct(1.5);
        notifier.set_take_profit_pct(4.5);

        assert_eq!(notifier.stop_loss_pct(), 1.5);
        assert_eq!(notifier.take_profit_pct(), 4.5);
    }

    #[test]
    fn test_set_bot_name() {
        let mut notifier = DiscordNotifier::new(
            "https://discord.com/api/webhooks/test".to_string(),
            true,
            true,
            true,
            true,
        );

        notifier.set_bot_name("My Custom Bot".to_string());
        // Bot name is private, but we can verify it doesn't panic
    }
}
