//! Prometheus Metrics HTTP Server
//!
//! This example demonstrates how to expose execution metrics via an HTTP endpoint
//! for Prometheus to scrape. This is a production-ready metrics exporter that can
//! be integrated into your live trading system.
//!
//! The server provides:
//! - `/metrics` - Prometheus metrics endpoint
//! - `/health` - Health check endpoint
//! - `/status` - Human-readable status page
//!
//! Run with:
//! ```bash
//! cargo run --example metrics_server --release
//! ```
//!
//! Then access:
//! - http://localhost:9090/metrics - Prometheus format
//! - http://localhost:9090/health - Health check
//! - http://localhost:9090/status - Status page

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use vision::execution::{ExecutionStrategy, InstrumentedExecutionManager, OrderRequest, Side};
use vision::{LivePipeline, MarketData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           Prometheus Metrics Server Example                  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Create shared execution manager
    let exec_manager = Arc::new(Mutex::new(InstrumentedExecutionManager::new()));

    // Start background trading simulation
    let exec_manager_clone = Arc::clone(&exec_manager);
    thread::spawn(move || {
        if let Err(e) = run_trading_simulation(exec_manager_clone) {
            eprintln!("Trading simulation error: {}", e);
        }
    });

    // Start HTTP server
    println!("Starting HTTP metrics server on http://localhost:9090");
    println!("\nAvailable endpoints:");
    println!("  • http://localhost:9090/metrics - Prometheus metrics");
    println!("  • http://localhost:9090/health  - Health check");
    println!("  • http://localhost:9090/status  - Status page");
    println!("\nPress Ctrl+C to stop\n");

    let listener = TcpListener::bind("127.0.0.1:9090")?;
    listener.set_nonblocking(false)?;

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let exec_manager_clone = Arc::clone(&exec_manager);
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, exec_manager_clone) {
                        eprintln!("Error handling client: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    Ok(())
}

/// Run a background trading simulation to generate metrics
fn run_trading_simulation(
    exec_manager: Arc<Mutex<InstrumentedExecutionManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = LivePipeline::default();
    pipeline.initialize()?;
    pipeline.warmup()?;

    println!("✓ Trading simulation started\n");

    let mut tick = 0;
    loop {
        // Generate market data
        let trend = (tick as f64 / 10.0).sin() * 2.0;
        let noise = ((tick * 7) % 11) as f64 / 10.0 - 0.5;
        let price = 100.0 + trend + noise;

        let data = MarketData::new(
            tick as i64,
            price - 0.1,
            price + 0.5,
            price - 0.5,
            price,
            1000.0 + tick as f64 * 50.0,
        );

        // Process prediction
        if let Some(prediction) = pipeline.process_tick(data)? {
            if prediction.meets_confidence(0.75) {
                let side = if prediction.is_bullish() {
                    Side::Buy
                } else if prediction.is_bearish() {
                    Side::Sell
                } else {
                    tick += 1;
                    thread::sleep(Duration::from_millis(500));
                    continue;
                };

                // Submit order
                if let Ok(mut manager) = exec_manager.lock() {
                    manager.submit_order(OrderRequest {
                        symbol: "DEMO".to_string(),
                        quantity: 100.0,
                        side,
                        strategy: ExecutionStrategy::TWAP {
                            duration: Duration::from_secs(5),
                            num_slices: 5,
                        },
                        limit_price: None,
                        venues: None,
                    });
                }
            }
        }

        // Process executions
        if let Ok(mut manager) = exec_manager.lock() {
            manager.process();
        }

        tick += 1;
        thread::sleep(Duration::from_millis(500));
    }
}

/// Handle HTTP client connection
fn handle_client(
    mut stream: TcpStream,
    exec_manager: Arc<Mutex<InstrumentedExecutionManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let buf_reader = BufReader::new(&stream);
    let request_line = buf_reader.lines().next().unwrap()?;

    // Parse request
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        send_response(&mut stream, 400, "Bad Request", "text/plain")?;
        return Ok(());
    }

    let path = parts[1];

    match path {
        "/metrics" => handle_metrics(&mut stream, exec_manager)?,
        "/health" => handle_health(&mut stream, exec_manager)?,
        "/status" => handle_status(&mut stream, exec_manager)?,
        "/" => handle_index(&mut stream)?,
        _ => send_response(&mut stream, 404, "Not Found", "text/plain")?,
    }

    Ok(())
}

/// Handle /metrics endpoint (Prometheus format)
fn handle_metrics(
    stream: &mut TcpStream,
    exec_manager: Arc<Mutex<InstrumentedExecutionManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = if let Ok(manager) = exec_manager.lock() {
        manager.export_metrics()?
    } else {
        return send_response(stream, 503, "Service Unavailable", "text/plain");
    };

    send_response(stream, 200, &metrics, "text/plain; version=0.0.4")?;
    Ok(())
}

/// Handle /health endpoint
fn handle_health(
    stream: &mut TcpStream,
    exec_manager: Arc<Mutex<InstrumentedExecutionManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (is_healthy, status_json) = if let Ok(manager) = exec_manager.lock() {
        let health = manager.health_status();
        let json = format!(
            r#"{{
  "status": "{}",
  "healthy": {},
  "total_orders": {},
  "completed_orders": {},
  "failed_orders": {},
  "quality_score": {:.2},
  "uptime_seconds": {}
}}"#,
            if health.healthy {
                "healthy"
            } else {
                "unhealthy"
            },
            health.healthy,
            health.total_orders,
            health.completed_orders,
            health.failed_orders,
            health.avg_quality_score,
            health.last_update.elapsed().as_secs()
        );
        (health.healthy, json)
    } else {
        (
            false,
            r#"{"status": "error", "healthy": false}"#.to_string(),
        )
    };

    let status_code = if is_healthy { 200 } else { 503 };
    send_response(stream, status_code, &status_json, "application/json")?;
    Ok(())
}

/// Handle /status endpoint (human-readable)
fn handle_status(
    stream: &mut TcpStream,
    exec_manager: Arc<Mutex<InstrumentedExecutionManager>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let html = if let Ok(manager) = exec_manager.lock() {
        let health = manager.health_status();
        let stats = manager.strategy_stats();
        let report = manager.execution_report();

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Execution Manager Status</title>
    <style>
        body {{ font-family: monospace; margin: 40px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #4ec9b0; }}
        h2 {{ color: #569cd6; margin-top: 30px; }}
        .status {{ font-size: 24px; margin: 20px 0; }}
        .healthy {{ color: #4ec9b0; }}
        .unhealthy {{ color: #f48771; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #444; }}
        th {{ background: #2d2d30; color: #4ec9b0; }}
        .metric {{ color: #ce9178; }}
        .value {{ color: #b5cea8; }}
        .links {{ margin-top: 40px; }}
        .links a {{ color: #569cd6; margin-right: 20px; }}
    </style>
</head>
<body>
    <h1>🎯 Execution Manager Status</h1>

    <div class="status">
        Status: <span class="{}">{}</span>
    </div>

    <h2>📊 Overview</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Orders</td><td class="value">{}</td></tr>
        <tr><td>Completed</td><td class="value">{}</td></tr>
        <tr><td>Failed</td><td class="value">{}</td></tr>
        <tr><td>Success Rate</td><td class="value">{:.1}%</td></tr>
        <tr><td>Quality Score</td><td class="value">{:.1}/100</td></tr>
    </table>

    <h2>🎲 Strategy Distribution</h2>
    <table>
        <tr><th>Strategy</th><th>Count</th></tr>
        <tr><td>Market</td><td class="value">{}</td></tr>
        <tr><td>Limit</td><td class="value">{}</td></tr>
        <tr><td>TWAP</td><td class="value">{}</td></tr>
        <tr><td>VWAP</td><td class="value">{}</td></tr>
    </table>

    <h2>📈 Execution Analytics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Executions</td><td class="value">{}</td></tr>
        <tr><td>Total Quantity</td><td class="value">{:.0}</td></tr>
        <tr><td>Avg Slippage</td><td class="value">{:.2} bps</td></tr>
        <tr><td>Total Cost</td><td class="value">${:.2}</td></tr>
        <tr><td>Quality Score</td><td class="value">{:.1}/100</td></tr>
    </table>

    <div class="links">
        <a href="/metrics">📊 Prometheus Metrics</a>
        <a href="/health">❤️ Health Check</a>
        <a href="/status">🔄 Refresh</a>
    </div>

    <p style="margin-top: 40px; color: #858585;">
        Last updated: <span class="value">{} seconds ago</span>
    </p>
</body>
</html>"#,
            if health.healthy {
                "healthy"
            } else {
                "unhealthy"
            },
            if health.healthy {
                "✓ Healthy"
            } else {
                "✗ Unhealthy"
            },
            health.total_orders,
            health.completed_orders,
            health.failed_orders,
            if health.total_orders > 0 {
                100.0 * health.completed_orders as f64 / health.total_orders as f64
            } else {
                0.0
            },
            health.avg_quality_score,
            stats.market_orders,
            stats.limit_orders,
            stats.twap_orders,
            stats.vwap_orders,
            report.total_executions,
            report.total_quantity,
            report.average_slippage_bps,
            report.total_cost,
            report.quality_score,
            health.last_update.elapsed().as_secs()
        )
    } else {
        "<html><body><h1>Service Unavailable</h1></body></html>".to_string()
    };

    send_response(stream, 200, &html, "text/html")?;
    Ok(())
}

/// Handle / endpoint (index)
fn handle_index(stream: &mut TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    let html = r#"<!DOCTYPE html>
<html>
<head>
    <title>Vision Execution Metrics</title>
    <style>
        body { font-family: monospace; margin: 40px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        .links { margin: 30px 0; }
        .links a { display: block; color: #569cd6; margin: 10px 0; font-size: 18px; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        code { background: #2d2d30; padding: 2px 6px; color: #ce9178; }
    </style>
</head>
<body>
    <h1>🎯 Vision Execution Metrics Server</h1>

    <p>Welcome to the Vision execution metrics exporter!</p>

    <div class="links">
        <a href="/metrics">📊 Prometheus Metrics</a>
        <a href="/health">❤️ Health Check (JSON)</a>
        <a href="/status">📈 Status Dashboard</a>
    </div>

    <h2>Prometheus Configuration</h2>
    <p>Add this to your <code>prometheus.yml</code>:</p>
    <pre style="background: #2d2d30; padding: 20px; border-radius: 4px; overflow-x: auto;">
scrape_configs:
  - job_name: 'vision_execution'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    </pre>

    <h2>Example Queries</h2>
    <ul>
        <li><code>rate(vision_execution_total[1m])</code> - Execution rate</li>
        <li><code>histogram_quantile(0.95, vision_execution_slippage_bps)</code> - 95th percentile slippage</li>
        <li><code>vision_execution_quality_score</code> - Current quality score</li>
    </ul>
</body>
</html>"#;

    send_response(stream, 200, html, "text/html")?;
    Ok(())
}

/// Send HTTP response
fn send_response(
    stream: &mut TcpStream,
    status_code: u16,
    body: &str,
    content_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let status_text = match status_code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "Unknown",
    };

    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
        status_code,
        status_text,
        content_type,
        body.len(),
        body
    );

    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}
