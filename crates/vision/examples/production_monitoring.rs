//! Production Monitoring Example
//!
//! Demonstrates production deployment features for the vision pipeline:
//! - Health checks and status monitoring
//! - Metrics collection and Prometheus export
//! - Circuit breaker pattern for fault tolerance
//! - Retry logic with exponential backoff
//! - Error rate tracking
//!
//! Run with:
//! ```bash
//! cargo run --example production_monitoring --release
//! ```

use std::time::Duration;
use vision::production::{
    CircuitBreaker, CircuitBreakerConfig, ComponentHealth, HealthCheckConfig, HealthStatus,
    ProductionConfig, ProductionMonitor, ResourceMetrics, RetryConfig, RetryExecutor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Production Monitoring Example ===\n");

    // Example 1: Basic health monitoring
    println!("Example 1: Health Monitoring");
    println!("-----------------------------");
    health_monitoring_example()?;
    println!();

    // Example 2: Metrics collection
    println!("Example 2: Metrics Collection");
    println!("------------------------------");
    metrics_example()?;
    println!();

    // Example 3: Circuit breaker
    println!("Example 3: Circuit Breaker Pattern");
    println!("-----------------------------------");
    circuit_breaker_example()?;
    println!();

    // Example 4: Retry logic
    println!("Example 4: Retry with Exponential Backoff");
    println!("------------------------------------------");
    retry_example()?;
    println!();

    // Example 5: Full production system
    println!("Example 5: Complete Production System");
    println!("--------------------------------------");
    production_system_example()?;
    println!();

    Ok(())
}

/// Example 1: Health monitoring
fn health_monitoring_example() -> Result<(), Box<dyn std::error::Error>> {
    let monitor = ProductionMonitor::new(ProductionConfig::default());
    monitor.start()?;

    // Update component health
    monitor.update_component_health("database", HealthStatus::Healthy, None);
    monitor.update_component_health("cache", HealthStatus::Healthy, None);
    monitor.update_component_health(
        "api",
        HealthStatus::Degraded,
        Some("High latency detected".to_string()),
    );

    // Update resource metrics
    let resources = ResourceMetrics {
        timestamp: std::time::Instant::now(),
        memory_used_mb: 512.0,
        memory_available_mb: 1024.0,
        cpu_usage_percent: 45.0,
        thread_count: 8,
    };
    monitor.update_resources(resources);

    // Get health report
    let report = monitor.health_report();
    report.print_summary();

    println!("\nLiveness: {}", if monitor.is_alive() { "✓" } else { "✗" });
    println!("Readiness: {}", if monitor.is_ready() { "✓" } else { "✗" });
    println!("Uptime: {} seconds", monitor.uptime_seconds());

    Ok(())
}

/// Example 2: Metrics collection and export
fn metrics_example() -> Result<(), Box<dyn std::error::Error>> {
    let monitor = ProductionMonitor::new(ProductionConfig::default());

    // Simulate predictions
    for i in 0..100 {
        let latency = 0.001 + (i as f64 % 10.0) * 0.0001;
        let success = i % 10 != 0; // 10% failure rate
        monitor.record_prediction(latency, success);

        // Simulate cache behavior
        if i % 3 == 0 {
            monitor.record_cache_hit();
        } else {
            monitor.record_cache_miss();
        }
    }

    // Get metrics
    let metrics = monitor.metrics();
    println!("Predictions: {}", metrics.predictions_total.get());
    println!("Cache hit rate: {:.2}%", metrics.cache_hit_rate() * 100.0);
    println!("Error rate: {:.2}%", monitor.error_rate() * 100.0);

    // Export to Prometheus format
    println!("\n--- Prometheus Metrics (sample) ---");
    let prometheus = monitor.export_metrics();
    for line in prometheus.lines().take(10) {
        println!("{}", line);
    }
    println!("...");

    Ok(())
}

/// Example 3: Circuit breaker pattern
fn circuit_breaker_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        success_threshold: 2,
        timeout: Duration::from_secs(2),
        window_size: 10,
    };
    let breaker = CircuitBreaker::new(config);

    println!("Initial state: {}", breaker.state().as_str());

    // Simulate failing service
    println!("\nSimulating service failures...");
    for i in 1..=5 {
        let result = breaker.call(|| {
            if i <= 3 {
                Err::<(), _>("Service unavailable")
            } else {
                Ok(())
            }
        });

        match result {
            Ok(_) => println!("Call {}: Success", i),
            Err(e) => {
                let msg = match e {
                    vision::production::CircuitBreakerError::CircuitOpen => {
                        "Rejected (circuit open)"
                    }
                    vision::production::CircuitBreakerError::CallFailed(_) => "Failed",
                };
                println!("Call {}: {}", i, msg);
            }
        }

        println!("  Circuit state: {}", breaker.state().as_str());
    }

    // Print statistics
    let stats = breaker.stats();
    println!("\nCircuit Breaker Statistics:");
    println!("  Total calls: {}", stats.total_calls);
    println!("  Successful: {}", stats.successful_calls);
    println!("  Failed: {}", stats.failed_calls);
    println!("  Rejected: {}", stats.rejected_calls);
    println!("  Failure rate: {:.2}%", stats.failure_rate() * 100.0);
    println!("  State changes: {}", stats.state_changes);

    Ok(())
}

/// Example 4: Retry logic
fn retry_example() -> Result<(), Box<dyn std::error::Error>> {
    let config = RetryConfig {
        max_attempts: 4,
        initial_delay: Duration::from_millis(50),
        max_delay: Duration::from_secs(5),
        multiplier: 2.0,
    };
    let executor = RetryExecutor::new(config.clone());

    println!("Retry configuration:");
    println!("  Max attempts: {}", config.max_attempts);
    println!("  Initial delay: {}ms", config.initial_delay.as_millis());
    println!("  Multiplier: {}", config.multiplier);

    // Simulate flaky operation that succeeds on 3rd attempt
    println!("\nAttempting flaky operation...");
    let mut attempt_count = 0;
    let result = executor.execute(|| {
        attempt_count += 1;
        println!("  Attempt {}", attempt_count);

        if attempt_count < 3 {
            Err("Temporary failure")
        } else {
            Ok("Success!")
        }
    });

    match result {
        Ok(msg) => println!("Result: {} (after {} attempts)", msg, attempt_count),
        Err(e) => println!("Failed: {}", e),
    }

    // Show retry delays
    println!("\nRetry delay schedule:");
    for i in 0..config.max_attempts {
        let delay = config.delay_for_attempt(i);
        println!("  Attempt {}: {}ms delay", i, delay.as_millis());
    }

    Ok(())
}

/// Example 5: Complete production system
fn production_system_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create production monitor with strict configuration
    let config = ProductionConfig::strict();
    let monitor = ProductionMonitor::new(config);
    monitor.start()?;

    println!("Production system started with strict configuration");
    println!();

    // Simulate a production workload
    println!("Simulating production workload...");

    // Set up initial components
    monitor.update_component_health("model", HealthStatus::Healthy, None);
    monitor.update_component_health("cache", HealthStatus::Healthy, None);
    monitor.update_component_health("database", HealthStatus::Healthy, None);

    // Simulate 50 predictions
    for i in 0..50 {
        // Vary latency
        let latency = 0.001 + (i as f64 % 5.0) * 0.0005;

        // Occasional failures
        let success = i % 15 != 0;

        // Record metrics
        monitor.record_prediction(latency, success);

        // Cache behavior
        if i % 4 == 0 {
            monitor.record_cache_miss();
        } else {
            monitor.record_cache_hit();
        }

        // Update resource usage periodically
        if i % 10 == 0 {
            let resources = ResourceMetrics {
                timestamp: std::time::Instant::now(),
                memory_used_mb: 400.0 + (i as f64 * 2.0),
                memory_available_mb: 1024.0,
                cpu_usage_percent: 30.0 + (i as f64 % 20.0),
                thread_count: 8,
            };
            monitor.update_resources(resources);
        }
    }

    // Perform health check
    monitor.perform_health_check();

    // Print comprehensive status
    println!();
    monitor.print_status();

    // Check SLA compliance
    println!("\n=== SLA Compliance ===");
    let error_rate_healthy = monitor.is_error_rate_healthy();
    println!(
        "Error rate threshold: {} (healthy: {})",
        if error_rate_healthy { "✓" } else { "✗" },
        error_rate_healthy
    );

    let circuit_healthy =
        monitor.circuit_breaker().state() == vision::production::CircuitState::Closed;
    println!(
        "Circuit breaker: {} (closed: {})",
        if circuit_healthy { "✓" } else { "✗" },
        circuit_healthy
    );

    // Export metrics for external monitoring
    println!("\n=== Metrics Export ===");
    let prometheus = monitor.export_metrics();
    let metric_count = prometheus.lines().filter(|l| !l.starts_with('#')).count();
    println!("Exported {} metric values", metric_count);
    println!("Format: Prometheus");
    println!("Ready for scraping at /metrics endpoint");

    // Demonstrate alerting
    println!("\n=== Alerting ===");
    if monitor.error_rate() > 0.05 {
        println!(
            "⚠️  ALERT: High error rate detected: {:.2}%",
            monitor.error_rate() * 100.0
        );
    }

    let report = monitor.health_report();
    if !report.is_healthy() {
        println!("⚠️  ALERT: System health degraded");
        for component in report.unhealthy_components() {
            println!(
                "   - {}: {}",
                component.name,
                component.message.as_ref().unwrap_or(&"unknown".to_string())
            );
        }
    }

    if monitor.is_ready() {
        println!("✓ System is ready to serve traffic");
    } else {
        println!("⚠️  System is not ready");
    }

    Ok(())
}
