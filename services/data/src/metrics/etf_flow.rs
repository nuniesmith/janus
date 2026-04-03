//! ETF Net Flows Poller
//!
//! Scrapes ETF net flow data from Farside Investors or similar sources.
//!
//! ## Data Sources:
//! - Primary: Farside Investors (https://farside.co.uk/btc/)
//! - Alternative: Coinglass API (rate-limited)
//!
//! ## Flow Data:
//! - Bitcoin ETFs: IBIT, FBTC, ARKB, BITB, etc.
//! - Ethereum ETFs: ETHE, ETH, etc.
//! - Updated daily (T+1 reporting)
//!
//! ## Scraping Strategy:
//! Since Farside publishes data in HTML tables, we use the `scraper` crate
//! to parse the table rows and extract flow values.

use anyhow::{Context, Result};
use tracing::{debug, info};

use crate::actors::MetricData;

/// ETF Net Flows poller
#[derive(Clone)]
pub struct EtfFlowPoller {
    url: String,
    client: reqwest::Client,
}

impl EtfFlowPoller {
    /// Create a new ETF Flow poller
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .user_agent("Mozilla/5.0 (compatible; DataFactory/1.0)")
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Poll ETF net flows
    pub async fn poll(&self) -> Result<Vec<MetricData>> {
        debug!("EtfFlowPoller: Fetching ETF net flows from {}", self.url);

        // Fetch the HTML page
        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .context("Failed to fetch ETF flows page")?;

        if !response.status().is_success() {
            anyhow::bail!("ETF flows page returned status: {}", response.status());
        }

        let html = response
            .text()
            .await
            .context("Failed to read ETF flows HTML")?;

        // Parse the HTML and extract flow data
        let flows = self.parse_etf_flows(&html)?;

        info!(
            "EtfFlowPoller: Fetched {} ETF flow data points",
            flows.len()
        );

        Ok(flows)
    }

    /// Parse ETF flows from HTML
    fn parse_etf_flows(&self, html: &str) -> Result<Vec<MetricData>> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(html);

        // Farside Investors uses a table with class "etfs" or similar
        // We'll try multiple selectors to be robust
        let table_selectors = vec!["table.etfs", "table#etf-table", "table.table", "table"];

        let row_selector = Selector::parse("tr").unwrap();
        let cell_selector = Selector::parse("td").unwrap();
        let header_selector = Selector::parse("th").unwrap();

        let mut metrics = Vec::new();
        let timestamp = chrono::Utc::now().timestamp_millis();

        // Try to find the ETF table
        let mut table_found = false;
        for table_sel_str in table_selectors {
            if let Ok(table_selector) = Selector::parse(table_sel_str)
                && let Some(table) = document.select(&table_selector).next()
            {
                table_found = true;
                debug!(
                    "EtfFlowPoller: Found table with selector: {}",
                    table_sel_str
                );

                // Find the date column index by looking at headers
                let mut date_col_idx = None;
                let mut ticker_col_idx = None;
                let mut flow_col_indices = Vec::new();

                // Parse header row to find column indices
                if let Some(header_row) = table.select(&row_selector).next() {
                    for (idx, header) in header_row.select(&header_selector).enumerate() {
                        let header_text = header.text().collect::<String>().to_lowercase();
                        if header_text.contains("date") {
                            date_col_idx = Some(idx);
                        } else if header_text.contains("fund") || header_text.contains("ticker") {
                            ticker_col_idx = Some(idx);
                        } else if !header_text.is_empty() && header_text != "total" {
                            // Assume other columns are flow data (dates)
                            flow_col_indices.push(idx);
                        }
                    }
                }

                // Parse data rows
                for (_row_idx, row) in table.select(&row_selector).enumerate().skip(1) {
                    let cells: Vec<_> = row.select(&cell_selector).collect();

                    if cells.len() < 2 {
                        continue; // Skip rows with too few cells
                    }

                    // Extract ticker/fund name (usually first or second column)
                    let ticker_idx = ticker_col_idx.unwrap_or(0);
                    if ticker_idx >= cells.len() {
                        continue;
                    }

                    let ticker = cells[ticker_idx]
                        .text()
                        .collect::<String>()
                        .trim()
                        .to_string();

                    if ticker.is_empty() || ticker.to_lowercase().contains("total") {
                        continue; // Skip empty or total rows
                    }

                    // Parse flow values from remaining columns
                    for (cell_idx, cell) in cells.iter().enumerate() {
                        if cell_idx == ticker_idx || cell_idx == date_col_idx.unwrap_or(usize::MAX)
                        {
                            continue; // Skip ticker and date columns
                        }

                        let flow_text = cell.text().collect::<String>().trim().to_string();

                        if flow_text.is_empty() || flow_text == "-" || flow_text == "—" {
                            continue; // Skip empty or placeholder values
                        }

                        // Try to parse the flow value
                        if let Ok(value) = Self::parse_flow_value(&flow_text) {
                            metrics.push(MetricData {
                                metric_type: "etf_inflow".to_string(),
                                asset: "BTC".to_string(), // Default to BTC
                                source: "farside".to_string(),
                                value,
                                meta: Some(ticker.clone()),
                                timestamp,
                            });
                        }
                    }
                }

                break; // Found and processed table
            }
        }

        if !table_found {
            debug!("EtfFlowPoller: No table found, returning empty metrics");
        } else {
            debug!("EtfFlowPoller: Parsed {} flow data points", metrics.len());
        }

        // If no metrics were parsed, return a single zero-value metric to indicate the poll succeeded
        if metrics.is_empty() {
            metrics.push(MetricData {
                metric_type: "etf_inflow".to_string(),
                asset: "BTC".to_string(),
                source: "farside".to_string(),
                value: 0.0,
                meta: Some("no_data".to_string()),
                timestamp,
            });
        }

        Ok(metrics)
    }

    /// Parse flow value from string (handles formats like "$10.5M", "($5.2M)")
    fn parse_flow_value(s: &str) -> Result<f64> {
        let cleaned = s.trim().replace(['$', ',', 'M', 'B'], "");

        // Handle negative values in parentheses
        let (value_str, is_negative) = if cleaned.starts_with('(') && cleaned.ends_with(')') {
            (cleaned.trim_start_matches('(').trim_end_matches(')'), true)
        } else {
            (cleaned.as_str(), false)
        };

        let mut value: f64 = value_str.parse().context("Failed to parse flow value")?;

        // Multiply by millions
        if s.contains('M') {
            value *= 1_000_000.0;
        } else if s.contains('B') {
            value *= 1_000_000_000.0;
        }

        if is_negative {
            value = -value;
        }

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_flow_value_positive() {
        let value = EtfFlowPoller::parse_flow_value("$10.5M").unwrap();
        assert_eq!(value, 10_500_000.0);
    }

    #[test]
    fn test_parse_flow_value_negative() {
        let value = EtfFlowPoller::parse_flow_value("($5.2M)").unwrap();
        assert_eq!(value, -5_200_000.0);
    }

    #[test]
    fn test_parse_flow_value_billions() {
        let value = EtfFlowPoller::parse_flow_value("$1.5B").unwrap();
        assert_eq!(value, 1_500_000_000.0);
    }

    #[test]
    fn test_parse_flow_value_with_commas() {
        let value = EtfFlowPoller::parse_flow_value("$1,234.56M").unwrap();
        assert_eq!(value, 1_234_560_000.0);
    }

    #[tokio::test]
    async fn test_etf_flow_poller_creation() {
        let poller = EtfFlowPoller::new("https://farside.co.uk/btc/".to_string());
        assert_eq!(poller.url, "https://farside.co.uk/btc/");
    }
}
