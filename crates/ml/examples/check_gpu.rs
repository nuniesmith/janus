//! GPU detection example for JANUS ML
//!
//! This example checks if GPU acceleration is available via WGPU/Vulkan
//! and displays information about available compute devices.

use janus_ml::backend::{auto_backend, cpu_backend};

#[cfg(feature = "gpu")]
use janus_ml::backend::gpu_backend;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== JANUS ML GPU Detection ===\n");

    // Check CPU backend
    println!("✓ CPU Backend: Available");
    let cpu_config = cpu_backend();
    println!("  Type: {}", cpu_config.device().backend_type());

    // Check GPU backend
    #[cfg(feature = "gpu")]
    {
        println!("\n🔍 Checking GPU Backend...");
        match gpu_backend() {
            Ok(gpu_config) => {
                println!("✓ GPU Backend: Available");
                println!("  Type: {}", gpu_config.device().backend_type());

                // Try to enumerate WGPU devices
                println!("\n📊 Enumerating WGPU Adapters:");
                check_wgpu_adapters();
            }
            Err(e) => {
                println!("✗ GPU Backend: Not available");
                println!("  Error: {:?}", e);
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n⚠️  GPU support not compiled in");
        println!("   Rebuild with --features gpu to enable GPU support");
    }

    // Auto-detect best backend
    println!("\n🎯 Auto-detected Backend:");
    let auto_config = auto_backend();
    println!("  Selected: {}", auto_config.device().backend_type());
    println!("  Is GPU: {}", auto_config.device().is_gpu());

    println!("\n=== Detection Complete ===");
}

#[cfg(feature = "gpu")]
fn check_wgpu_adapters() {
    use pollster::FutureExt;

    let instance = wgpu::Instance::default();

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        println!("  ⚠️  No WGPU adapters found");
        return;
    }

    for (i, adapter) in adapters.iter().enumerate() {
        let info: wgpu::AdapterInfo = adapter.get_info();
        println!("\n  Adapter {}: {}", i, info.name);
        println!("    Type: {:?}", info.device_type);
        println!("    Backend: {:?}", info.backend);
        println!("    Driver: {}", info.driver);
        println!("    Driver Info: {}", info.driver_info);

        // Try to get device limits
        let features = adapter.features();
        println!("    Features: {:?}", features);

        // Check if we can create a device
        match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
        {
            Ok((device, _queue)) => {
                println!("    Status: ✓ Can create compute device");
                let limits: wgpu::Limits = device.limits();
                println!(
                    "    Max Compute Workgroup Size: {:?}",
                    limits.max_compute_workgroup_size_x
                );
                println!(
                    "    Max Buffer Size: {} MB",
                    limits.max_buffer_size / 1_048_576
                );
            }
            Err(e) => {
                println!("    Status: ✗ Cannot create device: {:?}", e);
            }
        }
    }
}
