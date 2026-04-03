//! Integration tests for backfill locking
//!
//! These tests validate the distributed locking mechanism with real Redis instances
//! and concurrent worker scenarios.

use janus_data::backfill::{BackfillLock, LockConfig, LockMetrics};
use prometheus::Registry;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

/// Helper to create a test Redis client
/// Skips tests if Redis is not available
async fn test_redis_client() -> Option<redis::Client> {
    let client = redis::Client::open("redis://127.0.0.1:6379/").ok()?;

    // Test connection
    match client.get_multiplexed_async_connection().await {
        Ok(_) => Some(client),
        Err(_) => None,
    }
}

/// Helper to create a BackfillLock instance for testing
/// Each call creates a unique registry to avoid test interference
async fn create_test_lock() -> Option<(BackfillLock, Arc<LockMetrics>)> {
    create_test_lock_with_prefix(format!("test:{}:", Uuid::new_v4())).await
}

/// Helper to create a BackfillLock instance with a specific key prefix
async fn create_test_lock_with_prefix(prefix: String) -> Option<(BackfillLock, Arc<LockMetrics>)> {
    let client = test_redis_client().await?;
    let registry = Registry::new();
    let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
    let config = LockConfig {
        key_prefix: prefix, // Unique prefix to avoid key collisions between tests
        ..Default::default()
    };
    let lock = BackfillLock::new(client, config, metrics.clone());
    Some((lock, metrics))
}

#[tokio::test]
async fn test_concurrent_workers_only_one_succeeds() {
    let (lock, metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let gap_id = format!("concurrent_test_{}", Uuid::new_v4());
    let lock = Arc::new(lock);

    // Spawn 5 concurrent workers trying to acquire the same lock
    let mut handles = vec![];
    for worker_id in 0..5 {
        let lock = lock.clone();
        let gap_id = gap_id.clone();

        let handle = tokio::spawn(async move {
            match lock.acquire(&gap_id).await {
                Ok(Some(_guard)) => {
                    // Lock acquired - simulate work
                    sleep(Duration::from_millis(100)).await;
                    Some(worker_id)
                }
                Ok(None) => {
                    // Lock contention - another worker has it
                    None
                }
                Err(e) => {
                    panic!("Redis error: {}", e);
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all workers to complete
    let mut successful_workers = vec![];
    for handle in handles {
        if let Some(worker_id) = handle.await.unwrap() {
            successful_workers.push(worker_id);
        }
    }

    // Exactly one worker should have acquired the lock
    assert_eq!(
        successful_workers.len(),
        1,
        "Expected exactly 1 worker to acquire lock, but {} succeeded",
        successful_workers.len()
    );

    // Verify metrics
    assert_eq!(metrics.locks_acquired.get(), 1);
    assert_eq!(metrics.locks_contended.get(), 4); // 4 workers failed
}

#[tokio::test]
async fn test_sequential_acquisition_after_release() {
    let (lock, metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let gap_id = "sequential_test_gap";

    // First worker acquires and releases
    let mut guard = lock.acquire(gap_id).await.unwrap().unwrap();
    sleep(Duration::from_millis(50)).await;
    guard.release().await.unwrap();
    drop(guard); // Explicitly drop the guard

    // Ensure release completed and Redis processed it
    sleep(Duration::from_millis(100)).await;

    // Second worker should be able to acquire
    let mut guard2 = lock.acquire(gap_id).await.unwrap().unwrap();
    sleep(Duration::from_millis(50)).await;
    guard2.release().await.unwrap();
    drop(guard2);

    // Allow time for async cleanup
    sleep(Duration::from_millis(100)).await;

    // Verify both acquisitions succeeded (each test has fresh metrics)
    assert_eq!(metrics.locks_acquired.get(), 2, "Expected 2 acquisitions");
    assert_eq!(metrics.locks_released.get(), 2, "Expected 2 releases");
}

#[tokio::test]
async fn test_lock_expires_on_worker_crash() {
    let client = match test_redis_client().await {
        Some(c) => c,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let registry = Registry::new();
    let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
    let config = LockConfig {
        ttl: Duration::from_secs(2), // Short TTL for testing
        ..Default::default()
    };
    let lock = BackfillLock::new(client, config, metrics);

    let gap_id = format!("crash_test_{}", Uuid::new_v4());

    // Worker 1 acquires lock and "crashes" (doesn't release)
    {
        let _guard = lock.acquire(&gap_id).await.unwrap().unwrap();
        // Simulate crash by not releasing and dropping guard without awaiting
        std::mem::forget(_guard);
    }

    // Worker 2 tries immediately and fails
    let result = lock.acquire(&gap_id).await.unwrap();
    assert!(result.is_none(), "Should not acquire while lock is held");

    // Wait for TTL expiration
    sleep(Duration::from_secs(3)).await;

    // Worker 2 tries again and succeeds
    let guard = lock.acquire(&gap_id).await.unwrap();
    assert!(guard.is_some(), "Should acquire after TTL expiration");
}

#[tokio::test]
async fn test_lock_extension_for_long_running_work() {
    let client = match test_redis_client().await {
        Some(c) => c,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let registry = Registry::new();
    let metrics = Arc::new(LockMetrics::new(&registry).unwrap());
    let config = LockConfig {
        ttl: Duration::from_secs(5),
        extension_interval: Duration::from_secs(2),
        ..Default::default()
    };
    let lock = BackfillLock::new(client, config, metrics.clone());

    let gap_id = format!("extension_test_{}", Uuid::new_v4());

    // Acquire lock
    let mut guard = lock.acquire(&gap_id).await.unwrap().unwrap();

    // Simulate long-running work with periodic extension
    for i in 0..3 {
        sleep(Duration::from_secs(2)).await;

        if guard.should_extend() {
            let extended = guard.extend().await.unwrap();
            assert!(extended, "Extension {} failed", i + 1);
        }
    }

    // Verify extensions occurred
    assert!(
        metrics.locks_extended.get() >= 1,
        "Should have extended at least once"
    );

    // Lock should still be valid
    assert!(lock.is_locked(&gap_id).await.unwrap());
}

#[tokio::test]
async fn test_two_workers_different_gaps() {
    let (lock, metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let lock = Arc::new(lock);
    let gap_id_1 = format!("gap_1_{}", Uuid::new_v4());
    let gap_id_2 = format!("gap_2_{}", Uuid::new_v4());

    // Two workers acquire locks for different gaps concurrently
    let lock1 = lock.clone();
    let gap1 = gap_id_1.clone();
    let handle1 = tokio::spawn(async move {
        let _guard = lock1.acquire(&gap1).await.unwrap().unwrap();
        sleep(Duration::from_millis(100)).await;
    });

    let lock2 = lock.clone();
    let gap2 = gap_id_2.clone();
    let handle2 = tokio::spawn(async move {
        let _guard = lock2.acquire(&gap2).await.unwrap().unwrap();
        sleep(Duration::from_millis(100)).await;
    });

    // Both should succeed
    handle1.await.unwrap();
    handle2.await.unwrap();

    // Verify both locks were acquired
    assert_eq!(metrics.locks_acquired.get(), 2);
    assert_eq!(metrics.locks_contended.get(), 0);
}

#[tokio::test]
async fn test_lock_metrics_accuracy() {
    let (lock, metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let gap_id = "metrics_test_gap";

    // Acquire lock
    let mut guard = lock.acquire(gap_id).await.unwrap().unwrap();
    assert_eq!(metrics.locks_acquired.get(), 1);
    assert_eq!(metrics.locks_held.get(), 1);

    // Contend for lock
    let result = lock.acquire(gap_id).await.unwrap();
    assert!(result.is_none());
    assert_eq!(metrics.locks_contended.get(), 1);

    // Release lock
    guard.release().await.unwrap();
    assert_eq!(metrics.locks_released.get(), 1);
    assert_eq!(metrics.locks_held.get(), 0);
}

#[tokio::test]
async fn test_realistic_backfill_scenario() {
    let (lock, metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let lock = Arc::new(lock);

    // Simulate 3 data-factory instances processing a shared queue of 10 gaps
    // Use a VecDeque as a shared queue that workers pull from
    use parking_lot::Mutex;
    use std::collections::{HashSet, VecDeque};

    let gap_queue: VecDeque<String> = (0..10).map(|i| format!("gap_{}", i)).collect();
    let gap_queue = Arc::new(Mutex::new(gap_queue));
    let completed_gaps = Arc::new(Mutex::new(HashSet::new()));
    let mut handles = vec![];

    // Spawn 3 worker instances
    for instance_id in 0..3 {
        let lock = lock.clone();
        let queue = gap_queue.clone();
        let completed = completed_gaps.clone();

        let handle = tokio::spawn(async move {
            let mut processed = 0;
            let mut skipped = 0;

            loop {
                // Pull a gap from the shared queue
                let gap_id = {
                    let mut q = queue.lock();
                    q.pop_front()
                };

                let Some(gap_id) = gap_id else {
                    // Queue is empty
                    break;
                };

                // Try to acquire lock for this gap
                match lock.acquire(&gap_id).await {
                    Ok(Some(mut guard)) => {
                        // Lock acquired - simulate backfill work
                        sleep(Duration::from_millis(50)).await;

                        // Periodically extend for long backfills
                        if gap_id.contains("gap_5") || gap_id.contains("gap_7") {
                            sleep(Duration::from_millis(50)).await;
                            guard.extend().await.ok();
                        }

                        // Mark as completed
                        {
                            let mut comp = completed.lock();
                            comp.insert(gap_id.clone());
                        }

                        processed += 1;
                        // Explicitly release to ensure metrics are updated
                        guard.release().await.ok();
                        drop(guard);
                    }
                    Ok(None) => {
                        // Another instance is processing this gap - put it back in queue
                        {
                            let mut q = queue.lock();
                            q.push_back(gap_id);
                        }
                        skipped += 1;
                        // Small delay before retrying queue
                        sleep(Duration::from_millis(10)).await;
                    }
                    Err(e) => {
                        panic!("Redis error in instance {}: {}", instance_id, e);
                    }
                }
            }

            (instance_id, processed, skipped)
        });

        handles.push(handle);
    }

    // Wait for all instances to complete
    for handle in handles {
        let (instance_id, processed, skipped) = handle.await.unwrap();
        println!(
            "Instance {} processed {} gaps, skipped {}",
            instance_id, processed, skipped
        );
    }

    // Allow time for async cleanup
    sleep(Duration::from_millis(100)).await;

    // Each gap should be processed exactly once
    let completed_count = completed_gaps.lock().len();
    assert_eq!(
        completed_count, 10,
        "Expected all 10 gaps to be processed exactly once, but got {}",
        completed_count
    );

    // Queue should be empty
    assert_eq!(gap_queue.lock().len(), 0, "Queue should be empty");

    // Verify metrics (each test has fresh metrics)
    let acquired = metrics.locks_acquired.get();
    let released = metrics.locks_released.get();
    assert_eq!(acquired, 10, "Should have acquired 10 locks");
    assert_eq!(released, 10, "Should have released 10 locks");

    println!("Locks acquired: {}, released: {}", acquired, released);
    println!("Total contention events: {}", metrics.locks_contended.get());
}

#[tokio::test]
async fn test_get_ttl_functionality() {
    let (lock, _metrics) = match create_test_lock().await {
        Some(l) => l,
        None => {
            eprintln!("Skipping test: Redis not available");
            return;
        }
    };

    let gap_id = "ttl_test_gap";

    // No TTL initially
    let ttl = lock.get_ttl(gap_id).await.unwrap();
    assert!(ttl.is_none(), "Should have no TTL before lock acquisition");

    // Acquire lock
    let _guard = lock.acquire(gap_id).await.unwrap().unwrap();

    // Should have TTL now
    let ttl = lock.get_ttl(gap_id).await.unwrap();
    assert!(ttl.is_some(), "Should have TTL after acquisition");

    let ttl_value = ttl.unwrap();
    assert!(
        ttl_value <= Duration::from_secs(300),
        "TTL should be at most 300 seconds"
    );
    assert!(
        ttl_value >= Duration::from_secs(290),
        "TTL should be close to 300 seconds initially"
    );
}
