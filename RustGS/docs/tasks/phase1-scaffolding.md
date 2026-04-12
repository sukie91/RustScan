# Phase 1: Create wgpu scaffolding for RustGS migration

## Context
We are migrating RustGS from candle-metal to burn+wgpu architecture. This is Phase 1 of a 7-phase migration plan documented in `RustGS/docs/plans/2026-04-12-burn-wgpu-migration-plan.md`.

Working directory: `/Users/tfjiang/Projects/RustScan/RustGS`

## Tasks

### 1. Update Cargo.toml
Add new burn/wgpu dependencies under `[dependencies]` section (all optional):

```toml
# GPU dependencies (new - burn+wgpu)
burn = { git = "https://github.com/tracel-ai/burn", features = ["autodiff", "wgpu"], optional = true }
burn-wgpu = { git = "https://github.com/tracel-ai/burn", features = ["exclusive-memory-only"], optional = true }
burn-cubecl = { git = "https://github.com/tracel-ai/burn", optional = true }
burn-fusion = { git = "https://github.com/tracel-ai/burn", optional = true }
burn-ir = { git = "https://github.com/tracel-ai/burn", optional = true }
wgpu = { version = "29", default-features = false, features = ["naga-ir"], optional = true }
naga_oil = { version = "0.22", optional = true }
bytemuck = { version = "1.20", features = ["derive"], optional = true }
tokio = { version = "1", features = ["rt"], optional = true }
```

Keep existing candle dependencies for now.

Add new feature flag in `[features]` section:
```toml
gpu-wgpu = [
    "dep:burn",
    "dep:burn-wgpu",
    "dep:burn-cubecl",
    "dep:burn-fusion",
    "dep:burn-ir",
    "dep:wgpu",
    "dep:naga_oil",
    "dep:bytemuck",
    "dep:tokio",
]
```

### 2. Create directory structure
```
src/training/wgpu/
├── mod.rs
├── backend.rs
├── gpu_primitives/
│   └── mod.rs
├── render/
│   └── mod.rs
├── render_bwd/
│   └── mod.rs
└── shaders/
    └── .gitkeep
```

### 3. Implement src/training/wgpu/backend.rs
```rust
//! Backend type aliases for burn+wgpu

use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::WgpuRuntime;
use burn::backend::Autodiff;

/// Base wgpu compute backend
pub type GsBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Fusion-optimized backend (merges consecutive GPU dispatches)
pub type GsBackend = Fusion<GsBackendBase>;

/// Differentiable backend (for training)
pub type GsDiffBackend = Autodiff<GsBackend>;

/// Device type
pub type GsDevice = <GsBackend as burn::prelude::Backend>::Device;
```

### 4. Implement src/training/wgpu/mod.rs
```rust
//! Burn+wgpu training backend

pub mod backend;
pub mod gpu_primitives;
pub mod render;
pub mod render_bwd;

pub use backend::{GsBackend, GsBackendBase, GsDiffBackend, GsDevice};
```

### 5. Create placeholder modules
- `src/training/wgpu/gpu_primitives/mod.rs` - empty comment: `//! GPU primitives (radix sort, prefix sum)`
- `src/training/wgpu/render/mod.rs` - empty comment: `//! Forward rendering pipeline`
- `src/training/wgpu/render_bwd/mod.rs` - empty comment: `//! Backward rendering pipeline`

### 6. Update src/training/mod.rs
Add at the end of the file:
```rust
#[cfg(feature = "gpu-wgpu")]
pub mod wgpu;
```

## Verification
After implementation, run:
```bash
cd /Users/tfjiang/Projects/RustScan/RustGS && cargo check --features gpu-wgpu
```

## Reference
- Migration plan: `RustGS/docs/plans/2026-04-12-burn-wgpu-migration-plan.md`
- Brush reference: `/Users/tfjiang/Projects/brush` (for architecture patterns only, do not add dependencies)
