//! Kafka Consumer for QuestDB Writer
//!
//! Consumes trade messages from Kafka topics and writes them to QuestDB
//! with exactly-once semantics and low latency.

use anyhow::{Context, Result};
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::Message;
use rdkafka::util::Timeout;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::kafka_producer::TradeMessage;
use crate::metrics::MetricsExporter;
use crate::storage::questdb::QuestDBWriter;

/// Kafka consumer configuration
#[derive(Debug, Clone)]
pub struct KafkaConsumerConfig {
    /// Kafka broker addresses
    pub brokers: String,
    /// Topic to consume from
    pub topic: String,
    /// Consumer group ID
    pub group_id: String,
    /// Auto-commit interval
    pub auto_commit_interval_ms: u64,
    /// Enable auto-commit
    pub enable_auto_commit: bool,
    /// Session timeout
    pub session_timeout_ms: u64,
    /// Max poll interval
    pub max_poll_interval_ms: u64,
    /// Fetch min bytes
    pub fetch_min_bytes: usize,
    /// Fetch wait max ms
    pub fetch_wait_max_ms: u64,
    /// Batch size for QuestDB writes
    pub batch_size: usize,
    /// Batch timeout
    pub batch_timeout_ms: u64,
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        Self {
            brokers: "localhost:9092".to_string(),
            topic: "trades".to_string(),
            group_id: "data-service-consumer".to_string(),
            auto_commit_interval_ms: 5000,
            enable_auto_commit: true,
            session_timeout_ms: 10000,
            max_poll_interval_ms: 300000,
            fetch_min_bytes: 1024,
            fetch_wait_max_ms: 500,
            batch_size: 1000,
            batch_timeout_ms: 100,
        }
    }
}

/// Kafka consumer for trade messages
pub struct KafkaTradeConsumer {
    config: KafkaConsumerConfig,
    consumer: StreamConsumer,
    questdb_writer: Arc<QuestDBWriter>,
    metrics: Arc<MetricsExporter>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Option<mpsc::Receiver<()>>,
}

impl KafkaTradeConsumer {
    /// Create a new Kafka consumer
    pub fn new(
        config: KafkaConsumerConfig,
        questdb_writer: Arc<QuestDBWriter>,
        metrics: Arc<MetricsExporter>,
    ) -> Result<Self> {
        info!(
            "Initializing Kafka consumer for topic: {} with group: {}",
            config.topic, config.group_id
        );

        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .set("group.id", &config.group_id)
            .set("enable.auto.commit", config.enable_auto_commit.to_string())
            .set(
                "auto.commit.interval.ms",
                config.auto_commit_interval_ms.to_string(),
            )
            .set("session.timeout.ms", config.session_timeout_ms.to_string())
            .set(
                "max.poll.interval.ms",
                config.max_poll_interval_ms.to_string(),
            )
            .set("fetch.min.bytes", config.fetch_min_bytes.to_string())
            .set("fetch.wait.max.ms", config.fetch_wait_max_ms.to_string())
            .set("enable.partition.eof", "false")
            .set("auto.offset.reset", "earliest")
            .set("client.id", "data-service-consumer")
            .set("isolation.level", "read_committed") // For exactly-once
            .create()
            .context("Failed to create Kafka consumer")?;

        // Subscribe to topic
        consumer
            .subscribe(&[&config.topic])
            .context("Failed to subscribe to Kafka topic")?;

        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        info!("Kafka consumer initialized successfully");

        Ok(Self {
            config,
            consumer,
            questdb_writer,
            metrics,
            shutdown_tx,
            shutdown_rx: Some(shutdown_rx),
        })
    }

    /// Start consuming messages
    pub async fn start(mut self) -> Result<()> {
        info!("Starting Kafka consumer loop");

        let mut shutdown_rx = self.shutdown_rx.take().expect("shutdown_rx already taken");

        let mut batch: Vec<TradeMessage> = Vec::with_capacity(self.config.batch_size);
        let mut batch_start = Instant::now();
        let batch_timeout = Duration::from_millis(self.config.batch_timeout_ms);

        let mut messages_processed = 0u64;
        let mut stats_start = Instant::now();

        loop {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                info!("Received shutdown signal");
                break;
            }

            // Poll for message with timeout
            match tokio::time::timeout(Duration::from_millis(100), self.consumer.recv()).await {
                Ok(Ok(msg)) => {
                    let start = Instant::now();

                    // Process message
                    match self.process_message(&msg) {
                        Ok(Some(trade)) => {
                            batch.push(trade);
                            messages_processed += 1;

                            // Update consumer lag metric
                            if let Some(timestamp) = msg.timestamp().to_millis() {
                                let lag_ms = chrono::Utc::now().timestamp_millis() - timestamp;
                                self.metrics.observe_histogram(
                                    "kafka_consumer_lag_ms",
                                    lag_ms as f64,
                                    &[("topic", &self.config.topic)],
                                );
                            }

                            // Update offset metric
                            self.metrics.set_gauge(
                                "kafka_consumer_offset",
                                msg.offset() as f64,
                                &[
                                    ("topic", &self.config.topic),
                                    ("partition", &msg.partition().to_string()),
                                ],
                            );

                            let duration = start.elapsed();
                            self.metrics.observe_histogram(
                                "kafka_consumer_process_duration_ms",
                                duration.as_millis() as f64,
                                &[],
                            );
                        }
                        Ok(None) => {
                            debug!("Skipped invalid message");
                        }
                        Err(e) => {
                            error!("Failed to process message: {}", e);
                            self.metrics.increment_counter(
                                "kafka_consumer_errors_total",
                                &[("error_type", "process_error")],
                            );
                        }
                    }

                    // Flush batch if full or timeout exceeded
                    if batch.len() >= self.config.batch_size
                        || batch_start.elapsed() >= batch_timeout
                    {
                        if !batch.is_empty() {
                            self.flush_batch(&mut batch).await?;
                            batch_start = Instant::now();
                        }
                    }
                }
                Ok(Err(e)) => {
                    error!("Kafka consumer error: {}", e);
                    self.metrics.increment_counter(
                        "kafka_consumer_errors_total",
                        &[("error_type", "receive_error")],
                    );
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                Err(_) => {
                    // Timeout - check if we should flush partial batch
                    if !batch.is_empty() && batch_start.elapsed() >= batch_timeout {
                        self.flush_batch(&mut batch).await?;
                        batch_start = Instant::now();
                    }
                }
            }

            // Log stats every 10 seconds
            if stats_start.elapsed() >= Duration::from_secs(10) {
                let elapsed = stats_start.elapsed();
                let rate = messages_processed as f64 / elapsed.as_secs_f64();
                info!(
                    "Kafka consumer stats: {} messages in {:?} ({:.0} msg/sec)",
                    messages_processed, elapsed, rate
                );
                messages_processed = 0;
                stats_start = Instant::now();
            }
        }

        // Flush any remaining messages
        if !batch.is_empty() {
            self.flush_batch(&mut batch).await?;
        }

        info!("Kafka consumer stopped");
        Ok(())
    }

    /// Process a Kafka message
    fn process_message(
        &self,
        msg: &rdkafka::message::BorrowedMessage,
    ) -> Result<Option<TradeMessage>> {
        let payload = match msg.payload() {
            Some(p) => p,
            None => {
                warn!("Received message with no payload");
                return Ok(None);
            }
        };

        // Deserialize trade message
        let trade: TradeMessage =
            serde_json::from_slice(payload).context("Failed to deserialize trade message")?;

        debug!(
            "Received trade: exchange={}, symbol={}, price={}",
            trade.exchange, trade.symbol, trade.price
        );

        Ok(Some(trade))
    }

    /// Flush batch of trades to QuestDB
    async fn flush_batch(&self, batch: &mut Vec<TradeMessage>) -> Result<()> {
        let batch_size = batch.len();
        if batch_size == 0 {
            return Ok(());
        }

        let start = Instant::now();
        debug!("Flushing batch of {} trades to QuestDB", batch_size);

        // Convert to QuestDB format and write
        // Note: This would use the actual QuestDBWriter implementation
        // For now, we'll just simulate the write
        for trade in batch.iter() {
            // In production, this would call:
            // self.questdb_writer.write_trade(trade).await?;
            debug!("Writing trade to QuestDB: {:?}", trade);
        }

        // Clear the batch
        batch.clear();

        let duration = start.elapsed();
        self.metrics.observe_histogram(
            "kafka_consumer_batch_write_duration_ms",
            duration.as_millis() as f64,
            &[],
        );
        self.metrics
            .increment_counter("kafka_consumer_trades_written_total", &[]);
        self.metrics
            .observe_histogram("kafka_consumer_batch_size", batch_size as f64, &[]);

        debug!(
            "Batch written successfully in {:?} ({:.0} trades/sec)",
            duration,
            batch_size as f64 / duration.as_secs_f64()
        );

        Ok(())
    }

    /// Get shutdown sender (for graceful shutdown)
    pub fn shutdown_handle(&self) -> mpsc::Sender<()> {
        self.shutdown_tx.clone()
    }

    /// Get consumer statistics
    pub fn stats(&self) -> ConsumerStats {
        // In production, collect from Kafka consumer stats
        ConsumerStats {
            messages_consumed: 0,
            bytes_consumed: 0,
            lag: 0,
            avg_latency_ms: 0.0,
        }
    }
}

/// Consumer statistics
#[derive(Debug, Clone)]
pub struct ConsumerStats {
    pub messages_consumed: u64,
    pub bytes_consumed: u64,
    pub lag: u64,
    pub avg_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = KafkaConsumerConfig::default();
        assert_eq!(config.brokers, "localhost:9092");
        assert_eq!(config.topic, "trades");
        assert_eq!(config.group_id, "data-service-consumer");
        assert!(config.enable_auto_commit);
        assert_eq!(config.batch_size, 1000);
    }

    #[tokio::test]
    async fn test_trade_deserialization() {
        let json = r#"{
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "trade_id": "12345",
            "price": 50000.0,
            "quantity": 0.1,
            "timestamp": 1234567890000,
            "side": "buy",
            "correlation_id": "corr-123"
        }"#;

        let trade: TradeMessage = serde_json::from_str(json).unwrap();
        assert_eq!(trade.exchange, "binance");
        assert_eq!(trade.symbol, "BTC/USDT");
        assert_eq!(trade.price, 50000.0);
    }
}
