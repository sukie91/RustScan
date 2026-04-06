# RustScan Development Guide

## Getting Started

### Prerequisites

- Rust 1.70+ (Edition 2021)
- Cargo (comes with Rust)
- Optional: OpenCV 4.x (for RustSLAM with opencv feature)
- Optional: PyTorch/LibTorch (for RustSLAM with deep-learning feature)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RustScan

# Build RustMesh
cd RustMesh
cargo build --release

# Build RustSLAM
cd ../RustSLAM
cargo build --release
```

## Building

### RustMesh

```bash
cd RustMesh

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# With parallel feature
cargo build --release --features parallel
```

### RustSLAM

```bash
cd RustSLAM

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# With optional features
cargo build --release --features opencv
cargo build --release --features image
cargo build --release --features deep-learning
```

## Testing

### RustMesh

```bash
cd RustMesh

# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### RustSLAM

```bash
cd RustSLAM

# Run all tests
cargo test

# Run library tests only
cargo test --lib

# Run integration tests (examples)
cargo test --example test_marching_cubes
cargo test --example test_video_loader
cargo test --example test_optimization_thread

# Run end-to-end pipeline test (uses test_data/video/sofa.MOV)
RUSTSCAN_E2E=1 cargo test -- test_end_to_end_pipeline_video

# Run E2E test with custom thresholds
RUSTSCAN_E2E=1 RUSTSCAN_E2E_MAX_FRAMES=100 RUSTSCAN_E2E_PSNR=25.0 cargo test -- test_end_to_end_pipeline_video
```

## Running Examples

### RustMesh

```bash
cd RustMesh

# List available examples
ls examples/

# Run an example
cargo run --example test_smart
cargo run --release --example openmesh_compare_decimation_trace -- 10
cargo run --example test_subdivision
```

### RustSLAM

```bash
cd RustSLAM

# Load a dataset example
cargo run --release --example load_tum_dataset -- --dataset path/to/dataset

# Run end-to-end mesh extraction example
cargo run --release --example e2e_slam_to_mesh

# Run full pipeline via CLI
cargo run --release -- --input ../test_data/video/sofa.MOV --output ./output

# Run example videos (from repo root)
./run_examples.sh
```

## Benchmarking

### RustMesh

```bash
cd RustMesh

# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench quadric_bench
```

## Code Organization

### RustMesh Structure

```
RustMesh/
├── src/
│   ├── Core/           # Core data structures
│   ├── Tools/          # Mesh algorithms
│   ├── Utils/          # Utilities
│   └── lib.rs          # Library root
├── examples/           # Example programs
├── benches/            # Benchmarks
└── Cargo.toml
```

### RustSLAM Structure

```
RustSLAM/
├── src/
│   ├── core/           # Core data structures
│   ├── features/       # Feature extraction
│   ├── tracker/        # Visual Odometry
│   ├── optimizer/      # Bundle Adjustment
│   ├── loop_closing/   # Loop detection
│   ├── fusion/         # 3D Gaussian Splatting
│   ├── mapping/        # Mapping
│   ├── pipeline/       # SLAM pipeline
│   ├── io/             # I/O utilities
│   ├── viewer/         # Visualization
│   └── lib.rs          # Library root
├── examples/           # Example programs
└── Cargo.toml
```

## Development Workflow

### 1. Feature Development

```bash
# Create a new branch
git checkout -b feature/my-feature

# Make changes
# ... edit files ...

# Run tests
cargo test

# Run clippy (linter)
cargo clippy

# Format code
cargo fmt

# Commit changes
git add .
git commit -m "feat: add my feature"
```

### 2. Adding Tests

#### Unit Tests

Add tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        // Test code
    }
}
```

#### Integration Tests

Create a new example file in `examples/`:

```rust
// examples/test_my_feature.rs

#[cfg(test)]
mod tests {
    #[test]
    fn test_integration() {
        // Integration test code
    }
}
```

### 3. Documentation

```bash
# Generate documentation
cargo doc --no-deps --open

# Generate documentation with private items
cargo doc --no-deps --document-private-items --open
```

## Common Tasks

### Adding a New Mesh Algorithm (RustMesh)

1. Create new file in `src/Tools/`
2. Implement the algorithm
3. Add tests
4. Add example in `examples/`
5. Update documentation

### Adding a New SLAM Module (RustSLAM)

1. Create new file in appropriate `src/` subdirectory
2. Implement the module
3. Add tests in `#[cfg(test)]` module
4. Add integration test in `examples/`
5. Update pipeline if needed
6. Update documentation

## Debugging

### Enable Logging (RustSLAM)

```bash
# Set log level
export RUST_LOG=debug
cargo run --example e2e_slam_to_mesh

# Or inline
RUST_LOG=debug cargo run --example e2e_slam_to_mesh
```

### Debug Build

```bash
# Build with debug symbols
cargo build

# Run with debugger (lldb on macOS)
rust-lldb target/debug/my_binary
```

## Performance Profiling

### Using cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Profile
cargo flamegraph --example e2e_slam_to_mesh
```

### Using Instruments (macOS)

```bash
# Build with release profile
cargo build --release

# Run with Instruments
instruments -t "Time Profiler" target/release/my_binary
```

## Continuous Integration

The project uses GitHub Actions for CI (if configured). Local checks:

```bash
# Run all checks
cargo test
cargo clippy -- -D warnings
cargo fmt -- --check
```

## Troubleshooting

### OpenCV Not Found

```bash
# macOS with Homebrew
brew install opencv

# Set environment variable
export OPENCV_LINK_LIBS=opencv4
export OPENCV_LINK_PATHS=/usr/local/lib
export OPENCV_INCLUDE_PATHS=/usr/local/include/opencv4
```

### Metal/GPU Issues

```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep Metal

# Disable GPU features if needed
cargo build --no-default-features
```

### Build Errors

```bash
# Clean build artifacts
cargo clean

# Update dependencies
cargo update

# Rebuild
cargo build
```

## Code Style

- Follow Rust standard style (enforced by `cargo fmt`)
- Use `cargo clippy` to catch common mistakes
- Write documentation comments for public APIs
- Keep functions focused and small
- Use descriptive variable names

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `cargo test`, `cargo clippy`, `cargo fmt`
6. Submit a pull request

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Clippy Lints](https://rust-lang.github.io/rust-clippy/)
