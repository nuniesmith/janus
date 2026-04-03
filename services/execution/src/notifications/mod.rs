//! Notifications Module
//!
//! This module provides notification integrations for the execution service.
//! Currently supports Discord webhooks, with plans to add email, SMS, and other channels.

pub mod discord;

pub use discord::DiscordNotifier;

use crate::error::Result;
use crate::types::{Fill, Order, Position};

/// Notification manager that can send to multiple channels
pub struct NotificationManager {
    discord: Option<DiscordNotifier>,
}

impl NotificationManager {
    /// Create a new notification manager
    pub fn new(discord: Option<DiscordNotifier>) -> Self {
        Self { discord }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        let discord = if std::env::var("DISCORD_ENABLE_NOTIFICATIONS")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false)
        {
            Some(DiscordNotifier::from_env())
        } else {
            None
        };

        Self::new(discord)
    }

    /// Notify that a trading signal was received
    #[allow(clippy::too_many_arguments)]
    pub async fn notify_signal_received(
        &self,
        signal_id: &str,
        symbol: &str,
        side: crate::types::OrderSide,
        quantity: f64,
        price: Option<f64>,
        confidence: f64,
        strategy_id: &str,
    ) -> Result<()> {
        if let Some(discord) = &self.discord {
            discord
                .notify_signal_received(
                    signal_id,
                    symbol,
                    side,
                    quantity,
                    price,
                    confidence,
                    strategy_id,
                )
                .await?;
        }
        Ok(())
    }

    /// Notify that an order was filled
    pub async fn notify_order_filled(
        &self,
        order: &Order,
        fill: &Fill,
        position: Option<&Position>,
    ) -> Result<()> {
        if let Some(discord) = &self.discord {
            discord.notify_order_filled(order, fill, position).await?;
        }
        Ok(())
    }

    /// Notify about a position update
    pub async fn notify_position_update(&self, position: &Position, reason: &str) -> Result<()> {
        if let Some(discord) = &self.discord {
            discord.notify_position_update(position, reason).await?;
        }
        Ok(())
    }

    /// Notify about an error
    pub async fn notify_error(
        &self,
        error_type: &str,
        error_message: &str,
        context: &str,
    ) -> Result<()> {
        if let Some(discord) = &self.discord {
            discord
                .notify_error(error_type, error_message, context)
                .await?;
        }
        Ok(())
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
        if let Some(discord) = &self.discord {
            discord
                .notify_daily_summary(
                    total_trades,
                    winning_trades,
                    losing_trades,
                    total_pnl,
                    win_rate,
                )
                .await?;
        }
        Ok(())
    }

    /// Check if any notification channel is enabled
    pub fn is_enabled(&self) -> bool {
        self.discord.as_ref().is_some_and(|d| d.is_enabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_manager_creation() {
        let manager = NotificationManager::new(None);
        assert!(!manager.is_enabled());
    }
}
