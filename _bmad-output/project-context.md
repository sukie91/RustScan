---
project_name: 'RustScan'
user_name: ' 飞哥'
date: '2026-03-01'
sections_completed: ['technology_stack', 'language_rules', 'architecture_rules', 'cli_config_rules', 'video_io_rules', 'mesh_extraction_rules', 'testing_rules', 'code_quality_rules', 'workflow_rules', 'critical_rules']
status: 'complete'
rule_count: 98
optimized_for_llm: true
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

**Language:**
- Rust Edition 2021

**Core Libraries:**
- glam 0.25 (SIMD math - shared across both libraries)
- nalgebra 0.33 (linear algebra - shared across both libraries)
- serde 1.0 (serialization)

**RustMesh Dependencies:**
- byteorder 1
- itertools 0.13
- criterion 0.5 (benchmarking)

**RustSLAM Dependencies:**
- apex-solver 1.0 (Bundle Adjustment)
- rayon 1.8 (parallelism)
- candle-core 0.9.2, candle-metal 0.27.1 (GPU acceleration via Apple MPS)
- kiddo 5.2.1 (KD-Tree for KNN matching)
- thiserror 1.0 (error handling)
- crossbeam-channel 0.5 (inter-thread communication)
- log 0.4, env_logger 0.11

**Optional Features:**
- opencv 0.98 (enable with feature "opencv")
- tch 0.5 (enable with feature "deep-learning")
- image 0.25 (enable with feature "image")

**Build Tools:**
- bindgen 0.69 (C/C++ bindings)

**Video I/O:**
- ffmpeg-next 8.0 (video decoding with VideoToolbox HW accel)
- lru 0.12 (frame cache)
- sysinfo 0.30 (system info)

**CLI:**
- clap 4.5 (derive macros for CLI args)
- serde_json 1.0 (JSON output)
- toml 0.8 (config file parsing)

## Critical Implementation Rules

### Language-Specific Rules (Rust)

**Handle System & Type Safety:**
- ALWAYS use `u32` for handle indices, NEVER `i32`
- Invalid handles MUST use `u32::MAX` as sentinel value
- Implement handle types as newtype wrappers around `BaseHandle`
- All handle types must implement: `Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash`
- Provide both `idx()` → `u32` and `idx_usize()` → `usize` methods for indexing

**Performance Annotations:**
- Mark hot-path functions with `#[inline]` attribute
- Use `#[inline]` for all handle accessor methods
- Profile configurations are critical:
  - Release: LTO enabled, codegen-units=1, opt-level=3, strip=true, panic="abort"
  - Dev: opt-level=1, debug="line-tables-only" for faster compilation
  - Test: opt-level=0

**Documentation Standards:**
- Module-level docs use `//!` at the top of files
- Public API docs use `///` with clear descriptions
- Include examples in doc comments for non-obvious APIs
- Document coordinate systems and conventions (e.g., "world from camera" for poses)

**Error Handling:**
- Use `thiserror` crate for custom error types
- Derive `Error` trait for all error enums
- Provide context in error messages

**SIMD & Math:**
- Use `glam` types (Vec3, Mat4, Quat) for all 3D math operations
- Prefer `glam` over manual array operations for SIMD benefits
- Use `nalgebra` only for linear algebra operations (matrix decomposition, solving)

**Optional Features:**
- Gate heavy dependencies behind feature flags
- Document feature requirements in module docs
- Examples: `opencv`, `deep-learning`, `image`

**Parallelism:**
- Use `rayon` for data parallelism (`.par_iter()`)
- Use `crossbeam-channel` for inter-thread communication
- Avoid `std::sync::mpsc` in favor of crossbeam for better performance

### Architecture & Library-Specific Rules

**RustMesh: Half-Edge Data Structure:**
- ALWAYS maintain half-edge connectivity invariants:
  - Each halfedge has exactly one opposite halfedge
  - Each halfedge belongs to exactly one face (or is boundary)
  - Halfedges form closed loops around faces
- Use circulators for traversing mesh topology (VertexCirculator, FaceCirculator)
- NEVER directly modify connectivity arrays - use provided methods
- Boundary halfedges have invalid face handles

**SoA (Structure of Arrays) Memory Layout:**
- Store mesh attributes in separate vectors (positions, normals, colors)
- Index all arrays using the same handle
- Benefits: Better cache locality, SIMD-friendly, easier to add/remove attributes
- When adding new attributes, extend `SoaKernel` or `AttribKernel`

**RustSLAM: Visual Odometry Pipeline:**
- Frame processing order: Feature extraction → Matching → Pose estimation → BA
- KeyFrame selection criteria: Sufficient parallax, feature count, time interval
- MapPoint lifecycle: Created from triangulation → Tracked → Culled if lost
- Use SE3 for all pose representations (not separate R and t)

**3D Gaussian Splatting:**
- Gaussian parameters: position (3D), covariance (6D), color (3D), opacity (1D)
- Rendering pipeline: Depth sort → Tile rasterization → Alpha blending
- Training: Forward render → Loss (SSIM + L1) → Backward → Densify/Prune
- Use tiled rendering for GPU efficiency
- Densification triggers: High gradient regions
- Pruning triggers: Low opacity, too large/small

**GPU Acceleration (Candle + Metal):**
- Use `candle-core` for tensor operations
- Use `candle-metal` for Apple MPS backend
- Keep data on GPU as long as possible (minimize CPU↔GPU transfers)
- Batch operations for better GPU utilization

**Bundle Adjustment:**
- Use `apex-solver` for optimization
- Parameterize poses as SE3 (not Euler angles)
- Use robust kernels (Huber) for outlier rejection
- Fix first pose to avoid gauge freedom

### CLI & Configuration Rules

**CLI Implementation (clap):**
- Use `#[derive(Parser)]` for CLI argument structs
- Provide `#[command(name, version, about)]` metadata
- Use `#[arg(long, value_name = "FILE")]` for named arguments
- Implement `ValueEnum` for enum arguments (OutputFormat, LogLevel)
- Always provide `Default` implementations for config structs

**Configuration File Handling:**
- Use TOML format for configuration files (`rustscan.toml`)
- Implement `serde::Deserialize` with `#[serde(default, deny_unknown_fields)]`
- CLI arguments MUST override config file settings
- Provide validation methods that return `ValidationErrors` (Vec<String>)

**Parameter Validation Pattern:**
```rust
impl TrackerParams {
    pub fn validate(&self) -> ValidationErrors {
        let mut errs = Vec::new();
        if self.max_features == 0 {
            errs.push("tracker.max_features must be > 0".into());
        }
        // ... more validations
        errs
    }
}
```

**Error Types for CLI:**
- Use `thiserror` for custom error types
- Provide `ExitCode` returns (0=success, 1=user error, 2=system error)
- Include recovery suggestions in error messages

### Video I/O Rules

**Video Decoder Configuration:**
- Default LRU cache capacity: 100 frames
- Prefer hardware acceleration (`prefer_hardware: true`)
- Support formats: MP4, MOV, HEVC (H.264/H.265)
- Hardware decoder: `h264_videotoolbox`, `hevc_videotoolbox` (macOS)

**Frame Cache Behavior:**
- Forward seeking: O(1) from cache or sequential decode
- Backward seeking: O(n) - requires full decoder reset (no keyframe seeking)
- Use `Arc<Vec<u8>>` for frame data to avoid copies
- Frame cache capacity configurable via `--video-cache-capacity`

**Error Handling for Video:**
- Define `VideoError` enum with `thiserror`
- Validate file existence, format, codec before decoding
- Provide clear error messages for unsupported formats
- Hardware fallback: Log when HW accel fails and SW decoder used

### Mesh Extraction Rules

**TSDF Volume Fusion:**
- Voxel size controls mesh resolution (default: 0.01m = 1cm)
- Truncation distance: typically 3-5x voxel size
- Division guards required: use `.max(1e-8)` for division operations
- Integrate depth frames from Gaussian rendering

**Marching Cubes:**
- Full 256-case lookup table implemented
- Color interpolation from TSDF volume
- Output: vertices with positions and colors, triangle indices

**Mesh Post-Processing:**
- Cluster filtering: Remove isolated triangles (floaters)
- Normal smoothing: Improve mesh appearance
- Export formats: OBJ (1-based indexing), PLY (0-based indexing)

**Mesh Metadata (JSON):**
- Bounding box (min, max)
- Vertex/face counts
- Processing timings
- TSDF configuration

### Testing Rules

**Test Organization:**
- Unit tests: Use `#[cfg(test)]` modules within source files
- Integration tests: Place in `examples/` directory with `test_*.rs` naming
- Benchmarks: Use `criterion` crate, place in `benches/` or `examples/bench.rs`
- Test data: Use `tempfile` crate for temporary files in tests

**Test Naming Conventions:**
- Test functions: `#[test] fn test_<functionality>()`
- Benchmark functions: `fn bench_<operation>(c: &mut Criterion)`
- Example tests: `examples/test_<feature>.rs`

**Assertions & Validation:**
- Use `assert!`, `assert_eq!`, `assert_ne!` for basic checks
- Use `approx` crate for floating-point comparisons (if needed)
- Test both success and failure cases
- Test boundary conditions (empty meshes, single vertex, etc.)

**Performance Testing:**
- Use `criterion` for benchmarking (not `#[bench]`)
- Benchmark critical paths: mesh operations, feature matching, rendering
- Set `harness = false` in Cargo.toml for criterion benches
- Profile with `opt-level = 0` for tests, `opt-level = 3` for benchmarks

**Test Data & Fixtures:**
- Use simple geometric primitives for unit tests (triangle, quad, cube)
- Load test meshes from `examples/data/` if needed
- Use `tempfile::tempdir()` for file I/O tests
- Clean up resources in test teardown

**Coverage Expectations:**
- Core data structures: High coverage (>80%)
- Algorithms: Test correctness with known inputs/outputs
- Edge cases: Boundary meshes, degenerate cases, empty inputs
- Error paths: Test error handling and recovery

### Code Quality & Style Rules

**File & Directory Organization:**
- Directory names: PascalCase (Core/, Tools/, Utils/, fusion/)
- File names: snake_case (handles.rs, soa_kernel.rs, diff_renderer.rs)
- Module structure:
  - Core/: Fundamental data structures
  - Tools/: Algorithms and operations
  - Utils/: Helper utilities
  - features/: Feature extraction (RustSLAM)
  - fusion/: 3D Gaussian Splatting (RustSLAM)

**Naming Conventions:**
- Types: PascalCase (VertexHandle, BaseHandle, Frame)
- Functions: snake_case (new, from_usize, is_valid)
- Constants: SCREAMING_SNAKE_CASE (if used)
- Modules: snake_case (core, tools, utils)
- Traits: PascalCase with descriptive names

**Code Formatting:**
- Use default `rustfmt` settings (no custom config)
- Run `cargo fmt` before committing
- Maximum line length: 100 characters (rustfmt default)
- Use trailing commas in multi-line expressions

**Documentation Requirements:**
- Every public module MUST have `//!` documentation
- Every public type/function MUST have `///` documentation
- Document non-obvious behavior and edge cases
- Include examples for complex APIs
- Document coordinate systems and conventions explicitly

**Import Organization:**
- Group imports: std → external crates → internal modules
- Use explicit imports, avoid glob imports (`use foo::*`)
- Prefer `use crate::` for internal imports

**Code Complexity:**
- Keep functions focused and single-purpose
- Extract complex logic into helper functions
- Avoid deep nesting (max 3-4 levels)
- Use early returns to reduce nesting

**Safety & Correctness:**
- Minimize `unsafe` code - justify when necessary
- Use `Option` and `Result` instead of panicking
- Validate inputs at API boundaries
- Use type system to enforce invariants (newtype pattern)

### Development Workflow Rules

**Git & Repository:**
- Main branch: `main` (use for PRs)
- Branch naming: Use descriptive names (feature/*, fix/*, refactor/*)
- Commit messages: Use conventional commits style
  - feat: New feature
  - fix: Bug fix
  - refactor: Code refactoring
  - docs: Documentation changes
  - test: Test additions/changes
  - perf: Performance improvements

**Build & Development:**
- Build RustMesh: `cd RustMesh && cargo build`
- Build RustSLAM: `cd RustSLAM && cargo build --release`
- Run tests: `cargo test` in respective directories
- Run benchmarks: `cargo bench` in respective directories
- Format code: `cargo fmt` before committing
- Check lints: `cargo clippy` to catch common issues

**Pull Request Requirements:**
- Code must compile without warnings
- All tests must pass
- Run `cargo fmt` and `cargo clippy`
- Update documentation if API changes
- Add tests for new functionality

**Release Process:**
- Use release profile for production builds
- Profile settings: LTO=true, codegen-units=1, opt-level=3, strip=true
- Test thoroughly before releasing
- Update version in Cargo.toml

**Dependencies:**
- Prefer stable, well-maintained crates
- Document why optional features are needed
- Keep dependencies up to date
- Avoid unnecessary dependencies

### Critical Don't-Miss Rules

**Anti-Patterns to AVOID:**

❌ **NEVER use i32 for handle indices** - Always use u32
- Reason: Rust's usize conversion is cleaner with u32, and we use u32::MAX as sentinel

❌ **NEVER directly modify connectivity arrays** - Use provided methods
- Reason: Half-edge invariants are complex and easy to break
- Breaking invariants leads to crashes or infinite loops in circulators

❌ **NEVER mix glam and nalgebra types without explicit conversion**
- Reason: They have different memory layouts and SIMD optimizations
- Use glam for 3D graphics, nalgebra for linear algebra

❌ **NEVER use std::sync::mpsc for inter-thread communication**
- Reason: crossbeam-channel is faster and more feature-rich
- Project standard is crossbeam

❌ **NEVER panic in library code** - Return Result or Option
- Reason: Libraries should let callers decide how to handle errors
- Only panic for truly unrecoverable programmer errors

❌ **NEVER use Euclidean distance for binary descriptor matching**
- Reason: Binary descriptors (BRIEF, ORB) require Hamming distance
- Use `HammingMatcher` for binary descriptors, `KnnMatcher` for float descriptors

❌ **NEVER forget division guards in TSDF operations**
- Reason: Division by near-zero causes NaN propagation
- Always use `.max(1e-8)` or similar guards

**Edge Cases to Handle:**

⚠️ **Empty meshes and degenerate cases:**
- Handle meshes with 0 vertices, 0 faces
- Handle single vertex, single edge cases
- Check for degenerate triangles (zero area)

⚠️ **Boundary conditions in half-edge mesh:**
- Boundary halfedges have invalid face handles
- Check `is_valid()` before dereferencing handles
- Circulators may terminate early on boundaries

⚠️ **Floating-point precision:**
- Use epsilon comparisons for float equality
- Be aware of numerical instability in matrix operations
- Normalize vectors after multiple operations

⚠️ **GPU memory limits:**
- Check tensor sizes before GPU allocation
- Batch operations to fit in GPU memory
- Free GPU tensors when no longer needed

**Performance Gotchas:**

🐌 **Avoid frequent CPU↔GPU transfers:**
- Keep data on GPU as long as possible
- Batch transfers when necessary
- Use GPU-native operations

🐌 **Avoid allocations in hot loops:**
- Pre-allocate vectors with capacity
- Reuse buffers when possible
- Use iterators instead of collecting

🐌 **Avoid unnecessary clones:**
- Use references when possible
- Implement Copy for small types
- Use Cow for conditional ownership

**Security Considerations:**

🔒 **Validate all external inputs:**
- Check mesh file formats for malformed data
- Validate image dimensions and formats
- Sanitize file paths

🔒 **Avoid buffer overflows:**
- Use safe indexing with bounds checks
- Prefer iterators over manual indexing
- Use `get()` instead of `[]` when unsure

**Coordinate System Conventions:**

📐 **Camera poses use "world from camera" convention:**
- T_wc transforms points from camera to world
- Camera center in world: C_w = -R^T * t
- Be consistent across all pose operations

📐 **Right-handed coordinate system:**
- X: right, Y: down, Z: forward (camera convention)
- Follow OpenCV conventions for compatibility

---

## Usage Guidelines

**For AI Agents:**

- Read this file before implementing any code
- Follow ALL rules exactly as documented
- When in doubt, prefer the more restrictive option
- Update this file if new patterns emerge

**For Humans:**

- Keep this file lean and focused on agent needs
- Update when technology stack changes
- Review quarterly for outdated rules
- Remove rules that become obvious over time

## Preferred Development Agent
- Default: Codex CLI
- Fallback: Claude Code

Last Updated: 2026-03-01
