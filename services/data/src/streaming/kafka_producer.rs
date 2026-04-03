//! Kafka Producer for Streaming Trade Ingestion
//!
//! This module provides a high-performance Kafka producer for streaming
//! trade data from exchanges to QuestDB with low latency (<100ms).

use anyhow::{Context, Result};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::metrics::PrometheusExporter;

/// Trade message for Kafka serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMessage {
    pub exchange: String,
    pub symbol: String,
    pub trade_id: String,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: i64,
    pub side: String,
    pub correlation_id: Option<String>,
}

/// Kafka producer configuration
#[derive(Debug, Clone)]
pub struct KafkaProducerConfig {
    /// Kafka broker addresses
    pub brokers: String,
    /// Topic name for trades
    pub topic: String,
    /// Producer timeout
    pub timeout_ms: u64,
    /// Enable idempotence for exactly-once semantics
    pub enable_idempotence: bool,
    /// Compression type (none, gzip, snappy, lz4, zstd)
    pub compression_type: String,
    /// Batch size in bytes
    pub batch_size: usize,
    /// Linger time in milliseconds
    pub linger_ms: u64,
    /// Max in-flight requests per connection
    pub max_in_flight: usize,
    /// Acks required (0, 1, all)
    pub acks: String,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        Self {
            brokers: "localhost:9092".to_string(),
            topic: "trades".to_string(),
            timeout_ms: 5000,
            enable_idempotence: true,
            compression_type: "lz4".to_string(),
            batch_size: 16384,
            linger_ms: 10,
            max_in_flight: 5,
            acks: "all".to_string(),
        }
    }
}

/// Kafka producer for streaming trades
pub struct KafkaTradeProducer {
    config: KafkaProducerConfig,
    producer: FutureProducer,
    metrics: Arc<PrometheusExporter>,
    sender: mpsc::Sender<ProducerMessage>,
}

/// Internal producer message
enum ProducerMessage {
    Trade(TradeMessage),
    Flush,
    Shutdown,
}

impl KafkaTradeProducer {
    /// Create a new Kafka producer
    pub fn new(config: KafkaProducerConfig, metrics: Arc<PrometheusExporter>) -> Result<Self> {
        info!(
            "Initializing Kafka producer with brokers: {}",
            config.brokers
        );

        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .set("message.timeout.ms", config.timeout_ms.to_string())
            .set("enable.idempotence", config.enable_idempotence.to_string())
            .set("compression.type", &config.compression_type)
            .set("batch.size", config.batch_size.to_string())
            .set("linger.ms", config.linger_ms.to_string())
            .set(
                "max.in.flight.requests.per.connection",
                config.max_in_flight.to_string(),
            )
            .set("acks", &config.acks)
            .set("retries", "2147483647") // Max retries
            .set("max.in.flight.requests.per.connection", "5")
            .set("client.id", "data-service-producer")
            .set("queue.buffering.max.messages", "1000000")
            .set("queue.buffering.max.kbytes", "1048576") // 1GB
            .create()
            .context("Failed to create Kafka producer")?;

        let (sender, receiver) = mpsc::channel(10000);

        let mut producer_instance = Self {
            config: config.clone(),
            producer,
            metrics,
            sender,
        };

        // Spawn background worker
        producer_instance.spawn_worker(receiver);

        info!("Kafka producer initialized successfully");
        Ok(producer_instance)
    }

    /// Spawn background worker to process messages
    fn spawn_worker(&mut self, mut receiver: mpsc::Receiver<ProducerMessage>) {
        let producer = self.producer.clone();
        let topic = self.config.topic.clone();
        let metrics = self.metrics.clone();
        let timeout = Duration::from_millis(self.config.timeout_ms);

        tokio::spawn(async move {
            info!("Kafka producer worker started");
            let mut batch_count = 0;
            let mut batch_start = Instant::now();

            while let Some(msg) = receiver.recv().await {
                match msg {
                    ProducerMessage::Trade(trade) => {
                        batch_count += 1;

                        let start = Instant::now();
                        if let Err(e) =
                            Self::send_trade(&producer, &topic, trade, timeout, &metrics).await
                        {
                            error!("Failed to send trade to Kafka: {}", e);
                            metrics.increment_counter(
                                "kafka_producer_errors_total",
                                &[("error_type", "send_failure")],
                            );
                        } else {
                            let duration = start.elapsed();
                            metrics.observe_histogram(
                                "kafka_producer_send_duration_ms",
                                duration.as_millis() as f64,
                                &[],
                            );
                            metrics.increment_counter("kafka_producer_messages_sent_total", &[]);
                        }

                        // Log batch stats every 1000 messages
                        if batch_count >= 1000 {
                            let elapsed = batch_start.elapsed();
                            let rate = batch_count as f64 / elapsed.as_secs_f64();
                            debug!(
                                "Kafka producer: {} messages in {:?} ({:.0} msg/sec)",
                                batch_count, elapsed, rate
                            );
                            batch_count = 0;
                            batch_start = Instant::now();
                        }
                    }
                    ProducerMessage::Flush => {
                        debug!("Flushing Kafka producer");
                        producer.flush(Timeout::After(timeout));
                    }
                    ProducerMessage::Shutdown => {
                        info!("Shutting down Kafka producer worker");
                        producer.flush(Timeout::After(timeout));
                        break;
                    }
                }
            }

            info!("Kafka producer worker stopped");
        });
    }

    /// Send a trade message to Kafka
    async fn send_trade(
        producer: &FutureProducer,
        topic: &str,
        trade: TradeMessage,
        timeout: Duration,
        metrics: &Arc<MetricsExporter>,
    ) -> Result<()> {
        // Serialize trade to JSON
        let payload = serde_json::to_vec(&trade).context("Failed to serialize trade")?;

        // Use exchange + symbol as partition key for ordering
        let key = format!("{}:{}", trade.exchange, trade.symbol);

        // Create Kafka record
        let record = FutureRecord::to(topic)
            .payload(&payload)
            .key(&key)
            .timestamp(trade.timestamp);

        // Send with timeout
        match producer.send(record, Timeout::After(timeout)).await {
            Ok((partition, offset)) => {
                debug!(
                    "Trade sent to Kafka: partition={}, offset={}, exchange={}, symbol={}",
                    partition, offset, trade.exchange, trade.symbol
                );
                metrics.set_gauge(
                    "kafka_producer_last_offset",
                    offset as f64,
                    &[("partition", &partition.to_string())],
                );
                Ok(())
            }
            Err((err, _)) => {
                error!("Failed to send trade to Kafka: {}", err);
                Err(anyhow::anyhow!("Kafka send error: {}", err))
            }
        }
    }

    /// Produce a trade message (non-blocking)
    pub async fn produce(&self, trade: TradeMessage) -> Result<()> {
        self.sender
            .send(ProducerMessage::Trade(trade))
            .await
            .context("Failed to queue trade for Kafka")?;
        Ok(())
    }

    /// Flush all pending messages
    pub async fn flush(&self) -> Result<()> {
        self.sender
            .send(ProducerMessage::Flush)
            .await
            .context("Failed to send flush command")?;
        Ok(())
    }

    /// Shutdown the producer
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Kafka producer");
        self.sender
            .send(ProducerMessage::Shutdown)
            .await
            .context("Failed to send shutdown command")?;

        // Give worker time to flush and cleanup
        tokio::time::sleep(Duration::from_secs(2)).await;

        Ok(())
    }

    /// Get producer statistics
    pub fn stats(&self) -> ProducerStats {
        // In production, collect from Kafka producer stats
        ProducerStats {
            messages_sent: 0,
            bytes_sent: 0,
            errors: 0,
            avg_latency_ms: 0.0,
        }
    }
}

/// Producer statistics
#[derive(Debug, Clone)]
pub struct ProducerStats {
    pub messages_sent: u64,
    pub bytes_sent: u64,
    pub errors: u64,
    pub avg_latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trade_message_serialization() {
        let trade = TradeMessage {
            exchange: "binance".to_string(),
            symbol: "BTC/USDT".to_string(),
            trade_id: "12345".to_string(),
            price: 50000.0,
            quantity: 0.1,
            timestamp: 1234567890000,
            side: "buy".to_string(),
            correlation_id: Some("corr-123".to_string()),
        };

        let json = serde_json::to_string(&trade).unwrap();
        assert!(json.contains("binance"));
        assert!(json.contains("BTC/USDT"));

        let deserialized: TradeMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.exchange, "binance");
        assert_eq!(deserialized.price, 50000.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = KafkaProducerConfig::default();
        assert_eq!(config.brokers, "localhost:9092");
        assert_eq!(config.topic, "trades");
        assert_eq!(config.compression_type, "lz4");
        assert!(config.enable_idempotence);
    }
}
