//! # Alerts Module
//!
//! Implements alerting integration with Slack, PagerDuty, and other notification systems.
//! This module enables the CNS to send alerts when critical conditions are detected.

use crate::Result;
use crate::reflexes::AlertSeverity;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{error, info, warn};

/// Alert destination configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Slack webhook URL (optional)
    pub slack_webhook_url: Option<String>,

    /// PagerDuty integration key (optional)
    pub pagerduty_integration_key: Option<String>,

    /// Email SMTP configuration (optional)
    pub email_config: Option<EmailConfig>,

    /// Custom webhook URLs
    pub custom_webhooks: Vec<String>,

    /// Timeout for alert requests
    pub timeout: Duration,

    /// Retry attempts on failure
    pub retry_attempts: u32,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            slack_webhook_url: None,
            pagerduty_integration_key: None,
            email_config: None,
            custom_webhooks: Vec::new(),
            timeout: Duration::from_secs(5),
            retry_attempts: 3,
        }
    }
}

/// Email configuration for alerts
#[derive(Debug, Clone)]
pub struct EmailConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_address: String,
    pub to_addresses: Vec<String>,
}

/// Alert notification manager
#[derive(Clone)]
pub struct AlertManager {
    config: AlertConfig,
    client: reqwest::Client,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertConfig) -> Self {
        let client = match reqwest::Client::builder().timeout(config.timeout).build() {
            Ok(c) => c,
            Err(e) => {
                error!(
                    "Failed to create HTTP client for alerts with custom timeout: {}. \
                     Falling back to default client.",
                    e
                );
                reqwest::Client::new()
            }
        };

        Self { config, client }
    }

    /// Send an alert to all configured destinations
    pub async fn send_alert(
        &self,
        severity: AlertSeverity,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<AlertResult> {
        let title = title.into();
        let message = message.into();

        let mut result = AlertResult::default();

        // Send to Slack
        if let Some(ref webhook_url) = self.config.slack_webhook_url {
            match self
                .send_slack_alert(webhook_url, &severity, &title, &message)
                .await
            {
                Ok(_) => {
                    info!("Alert sent to Slack: {}", title);
                    result.slack_success = true;
                }
                Err(e) => {
                    error!("Failed to send Slack alert: {}", e);
                    result.slack_error = Some(e.to_string());
                }
            }
        }

        // Send to PagerDuty
        if let Some(ref integration_key) = self.config.pagerduty_integration_key {
            match self
                .send_pagerduty_alert(integration_key, &severity, &title, &message)
                .await
            {
                Ok(_) => {
                    info!("Alert sent to PagerDuty: {}", title);
                    result.pagerduty_success = true;
                }
                Err(e) => {
                    error!("Failed to send PagerDuty alert: {}", e);
                    result.pagerduty_error = Some(e.to_string());
                }
            }
        }

        // Send to custom webhooks
        for webhook_url in &self.config.custom_webhooks {
            match self
                .send_webhook_alert(webhook_url, &severity, &title, &message)
                .await
            {
                Ok(_) => {
                    info!("Alert sent to webhook: {}", webhook_url);
                    result.webhook_success_count += 1;
                }
                Err(e) => {
                    error!("Failed to send webhook alert to {}: {}", webhook_url, e);
                    result.webhook_errors.push(e.to_string());
                }
            }
        }

        Ok(result)
    }

    /// Send alert to Slack
    async fn send_slack_alert(
        &self,
        webhook_url: &str,
        severity: &AlertSeverity,
        title: &str,
        message: &str,
    ) -> Result<()> {
        let color = match severity {
            AlertSeverity::Info => "#36a64f",     // Green
            AlertSeverity::Warning => "#ff9900",  // Orange
            AlertSeverity::Error => "#ff0000",    // Red
            AlertSeverity::Critical => "#8b0000", // Dark Red
        };

        let emoji = match severity {
            AlertSeverity::Info => ":information_source:",
            AlertSeverity::Warning => ":warning:",
            AlertSeverity::Error => ":x:",
            AlertSeverity::Critical => ":rotating_light:",
        };

        let payload = SlackWebhookPayload {
            text: format!("{} *{}*", emoji, title),
            attachments: vec![SlackAttachment {
                color: color.to_string(),
                fields: vec![
                    SlackField {
                        title: "Severity".to_string(),
                        value: format!("{:?}", severity),
                        short: true,
                    },
                    SlackField {
                        title: "Message".to_string(),
                        value: message.to_string(),
                        short: false,
                    },
                    SlackField {
                        title: "Timestamp".to_string(),
                        value: chrono::Utc::now().to_rfc3339(),
                        short: true,
                    },
                ],
            }],
        };

        self.send_with_retry(webhook_url, &payload).await?;
        Ok(())
    }

    /// Send alert to PagerDuty
    async fn send_pagerduty_alert(
        &self,
        integration_key: &str,
        severity: &AlertSeverity,
        title: &str,
        message: &str,
    ) -> Result<()> {
        let pd_severity = match severity {
            AlertSeverity::Info => "info",
            AlertSeverity::Warning => "warning",
            AlertSeverity::Error => "error",
            AlertSeverity::Critical => "critical",
        };

        let payload = PagerDutyPayload {
            routing_key: integration_key.to_string(),
            event_action: "trigger".to_string(),
            payload: PagerDutyEventPayload {
                summary: title.to_string(),
                severity: pd_severity.to_string(),
                source: "fks-cns".to_string(),
                custom_details: {
                    let mut details = HashMap::new();
                    details.insert("message".to_string(), message.to_string());
                    details.insert("timestamp".to_string(), chrono::Utc::now().to_rfc3339());
                    details
                },
            },
        };

        let url = "https://events.pagerduty.com/v2/enqueue";
        self.send_with_retry(url, &payload).await?;
        Ok(())
    }

    /// Send alert to generic webhook
    async fn send_webhook_alert(
        &self,
        webhook_url: &str,
        severity: &AlertSeverity,
        title: &str,
        message: &str,
    ) -> Result<()> {
        let payload = GenericWebhookPayload {
            severity: format!("{:?}", severity),
            title: title.to_string(),
            message: message.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        self.send_with_retry(webhook_url, &payload).await?;
        Ok(())
    }

    /// Send HTTP request with retry logic
    async fn send_with_retry<T: Serialize>(&self, url: &str, payload: &T) -> Result<()> {
        let mut last_error = None;

        for attempt in 1..=self.config.retry_attempts {
            match self.client.post(url).json(payload).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        return Ok(());
                    } else {
                        let status = response.status();
                        let body = response.text().await.unwrap_or_default();
                        last_error = Some(anyhow::anyhow!("HTTP {} - {}", status, body));
                    }
                }
                Err(e) => {
                    last_error = Some(e.into());
                }
            }

            if attempt < self.config.retry_attempts {
                warn!("Alert send attempt {} failed, retrying...", attempt);
                tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
            }
        }

        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("All retry attempts failed"))
            .into())
    }
}

/// Result of sending an alert
#[derive(Debug, Default)]
pub struct AlertResult {
    pub slack_success: bool,
    pub slack_error: Option<String>,
    pub pagerduty_success: bool,
    pub pagerduty_error: Option<String>,
    pub webhook_success_count: usize,
    pub webhook_errors: Vec<String>,
}

impl AlertResult {
    /// Check if any alert was successfully sent
    pub fn any_success(&self) -> bool {
        self.slack_success || self.pagerduty_success || self.webhook_success_count > 0
    }

    /// Check if all configured alerts were successful
    pub fn all_success(&self) -> bool {
        let slack_ok = self.slack_error.is_none();
        let pagerduty_ok = self.pagerduty_error.is_none();
        let webhooks_ok = self.webhook_errors.is_empty();

        slack_ok && pagerduty_ok && webhooks_ok
    }
}

// ============================================================================
// Payload Structures
// ============================================================================

#[derive(Debug, Serialize)]
struct SlackWebhookPayload {
    text: String,
    attachments: Vec<SlackAttachment>,
}

#[derive(Debug, Serialize)]
struct SlackAttachment {
    color: String,
    fields: Vec<SlackField>,
}

#[derive(Debug, Serialize)]
struct SlackField {
    title: String,
    value: String,
    short: bool,
}

#[derive(Debug, Serialize)]
struct PagerDutyPayload {
    routing_key: String,
    event_action: String,
    payload: PagerDutyEventPayload,
}

#[derive(Debug, Serialize)]
struct PagerDutyEventPayload {
    summary: String,
    severity: String,
    source: String,
    custom_details: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct GenericWebhookPayload {
    severity: String,
    title: String,
    message: String,
    timestamp: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alert_manager_creation() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);
        assert!(manager.config.slack_webhook_url.is_none());
    }

    #[tokio::test]
    async fn test_alert_result_any_success() {
        let mut result = AlertResult::default();
        assert!(!result.any_success());

        result.slack_success = true;
        assert!(result.any_success());
    }

    #[tokio::test]
    async fn test_alert_result_all_success() {
        let result = AlertResult::default();
        assert!(result.all_success()); // No errors means all success

        let result = AlertResult {
            slack_error: Some("error".to_string()),
            ..Default::default()
        };
        assert!(!result.all_success());
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.retry_attempts, 3);
        assert!(config.custom_webhooks.is_empty());
    }
}
