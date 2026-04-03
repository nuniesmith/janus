//! # WebSocket Signal Broadcaster
//!
//! Handles broadcasting of signals, risk alerts, portfolio updates, and market data
//! to WebSocket clients with subscription filtering and metrics tracking.

use crate::metrics::JanusMetrics;
use crate::websocket::{ClientManager, WebSocketMessage};
use std::sync::Arc;
use tracing::debug;

/// Signal broadcaster for WebSocket messages
pub struct SignalBroadcaster {
    client_manager: Arc<ClientManager>,
    metrics: Arc<JanusMetrics>,
}

impl SignalBroadcaster {
    /// Create a new signal broadcaster
    pub fn new(client_manager: Arc<ClientManager>, metrics: Arc<JanusMetrics>) -> Self {
        Self {
            client_manager,
            metrics,
        }
    }

    /// Broadcast a signal update to all subscribed clients
    pub async fn broadcast_signal(
        &self,
        signal: crate::websocket::SignalUpdate,
    ) -> Result<(), BroadcasterError> {
        debug!(
            "Broadcasting signal update: {} - {}",
            signal.symbol, signal.signal_type
        );

        let message = WebSocketMessage::SignalUpdate(signal.clone());
        let sent_count = self.broadcast_all(message).await?;

        // Update metrics
        self.metrics
            .websocket_metrics()
            .ws_messages_sent_total
            .inc_by(sent_count as u64);
        self.metrics
            .websocket_metrics()
            .signal_broadcast_count
            .inc();

        debug!("Signal broadcast sent to {} client(s)", sent_count);
        Ok(())
    }

    /// Broadcast a risk alert to all subscribed clients
    pub async fn broadcast_risk_alert(
        &self,
        alert: crate::websocket::RiskAlert,
    ) -> Result<(), BroadcasterError> {
        debug!(
            "Broadcasting risk alert: {:?} - {}",
            alert.alert_type, alert.severity
        );

        let message = WebSocketMessage::RiskAlert(alert.clone());
        let sent_count = self.broadcast_all(message).await?;

        // Update metrics
        self.metrics
            .websocket_metrics()
            .ws_messages_sent_total
            .inc_by(sent_count as u64);
        self.metrics.websocket_metrics().risk_alert_count.inc();

        debug!("Risk alert broadcast sent to {} client(s)", sent_count);
        Ok(())
    }

    /// Broadcast a portfolio update to all subscribed clients
    pub async fn broadcast_portfolio_update(
        &self,
        update: crate::websocket::PortfolioUpdate,
    ) -> Result<(), BroadcasterError> {
        debug!(
            "Broadcasting portfolio update: {} positions",
            update.positions.len()
        );

        let message = WebSocketMessage::PortfolioUpdate(update.clone());
        let sent_count = self.broadcast_all(message).await?;

        // Update metrics
        self.metrics
            .websocket_metrics()
            .ws_messages_sent_total
            .inc_by(sent_count as u64);

        debug!(
            "Portfolio update broadcast sent to {} client(s)",
            sent_count
        );
        Ok(())
    }

    /// Broadcast a performance update to all subscribed clients
    pub async fn broadcast_performance_update(
        &self,
        update: crate::websocket::PerformanceUpdate,
    ) -> Result<(), BroadcasterError> {
        debug!("Broadcasting performance update");

        let message = WebSocketMessage::PerformanceUpdate(update.clone());
        let sent_count = self.broadcast_all(message).await?;

        // Update metrics
        self.metrics
            .websocket_metrics()
            .ws_messages_sent_total
            .inc_by(sent_count as u64);

        debug!(
            "Performance update broadcast sent to {} client(s)",
            sent_count
        );
        Ok(())
    }

    /// Broadcast market data update to subscribed clients
    pub async fn broadcast_market_data(
        &self,
        data: crate::websocket::MarketDataUpdate,
    ) -> Result<(), BroadcasterError> {
        debug!(
            "Broadcasting market data: {} - {:?}",
            data.symbol, data.data_type
        );

        let message = WebSocketMessage::MarketData(data.clone());
        let sent_count = self.broadcast_all(message).await?;

        // Update metrics
        self.metrics
            .websocket_metrics()
            .ws_messages_sent_total
            .inc_by(sent_count as u64);

        Ok(())
    }

    /// Broadcast to clients matching a symbol
    pub async fn broadcast_to_symbol(
        &self,
        symbol: &str,
        message: WebSocketMessage,
    ) -> Result<(), BroadcasterError> {
        debug!("Broadcasting message to symbol subscribers: {}", symbol);

        self.client_manager
            .send_to_symbol_subscribers(symbol, message)
            .await;

        Ok(())
    }

    /// Broadcast message to all clients (no filtering)
    async fn broadcast_all(&self, message: WebSocketMessage) -> Result<usize, BroadcasterError> {
        let client_count = self.client_manager.client_count().await;
        debug!("Broadcasting message to all {} client(s)", client_count);

        self.client_manager.broadcast(message).await;

        Ok(client_count)
    }

    /// Get broadcaster statistics
    pub async fn get_stats(&self) -> BroadcasterStats {
        BroadcasterStats {
            active_clients: self.client_manager.client_count().await,
            total_messages_sent: self
                .metrics
                .websocket_metrics()
                .ws_messages_sent_total
                .get(),
            signals_broadcast: self
                .metrics
                .websocket_metrics()
                .signal_broadcast_count
                .get(),
            risk_alerts_broadcast: self.metrics.websocket_metrics().risk_alert_count.get(),
        }
    }
}

/// Broadcaster statistics
#[derive(Debug, Clone)]
pub struct BroadcasterStats {
    pub active_clients: usize,
    pub total_messages_sent: u64,
    pub signals_broadcast: u64,
    pub risk_alerts_broadcast: u64,
}

/// Broadcaster errors
#[derive(Debug, thiserror::Error)]
pub enum BroadcasterError {
    #[error("Broadcast failed: {0}")]
    BroadcastFailed(String),

    #[error("Client error: {0}")]
    ClientError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_broadcaster() -> SignalBroadcaster {
        let client_manager = Arc::new(ClientManager::new());
        let metrics = Arc::new(JanusMetrics::new().unwrap());
        SignalBroadcaster::new(client_manager, metrics)
    }

    #[tokio::test]
    async fn test_broadcaster_creation() {
        let broadcaster = create_test_broadcaster();
        let stats = broadcaster.get_stats().await;
        assert_eq!(stats.active_clients, 0);
    }

    #[tokio::test]
    async fn test_broadcast_all() {
        let broadcaster = create_test_broadcaster();
        let message = WebSocketMessage::Ping;

        let result = broadcaster.broadcast_all(message).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_broadcaster_stats() {
        let broadcaster = create_test_broadcaster();
        let stats = broadcaster.get_stats().await;

        assert_eq!(stats.active_clients, 0);
        assert_eq!(stats.total_messages_sent, 0);
        assert_eq!(stats.signals_broadcast, 0);
        assert_eq!(stats.risk_alerts_broadcast, 0);
    }
}
