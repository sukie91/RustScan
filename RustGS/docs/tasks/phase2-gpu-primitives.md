# Phase 2: Implement GPU Primitives (Radix Sort & Prefix Sum)

## Context
This is Phase 2 of the burn+wgpu migration. Phase 1 (scaffolding) is complete. We now need to implement GPU primitives that will be used by the rendering pipeline.

Working directory: `/Users/tfjiang/Projects/RustScan/RustGS`

Reference implementation: `/Users/tfjiang/Projects/brush/crates/brush-sort/` and `/Users/tfjiang/Projects/brush/crates/brush-prefix-sum/`

## Tasks

### 1. Implement GPU Radix Sort

**Files to create:**
- `src/training/wgpu/gpu_primitives/radix_sort.rs`
- `src/training/wgpu/shaders/radix_sort.wgsl`

**Requirements:**
- Implement GPU radix sort for 32-bit keys (depth values)
- Return sorted indices (argsort functionality)
- Use WGSL compute shaders
- Support variable input sizes
- Reference: brush-sort implementation at `~/Projects/brush/crates/brush-sort/`

**Key features:**
- 4-bit radix (8 passes for 32-bit keys)
- Local histogram computation
- Global prefix sum for bucket offsets
- Scatter phase to reorder elements

**Rust API:**
```rust
/// GPU radix sort for 32-bit keys, returns sorted indices
pub async fn radix_sort_u32<B: Backend>(
    keys: Tensor<B, 1>,
    device: &B::Device,
) -> Result<Tensor<B, 1>, String>;
```

### 2. Implement GPU Prefix Sum

**Files to create:**
- `src/training/wgpu/gpu_primitives/prefix_sum.rs`
- `src/training/wgpu/shaders/prefix_sum.wgsl`

**Requirements:**
- Implement inclusive prefix sum (scan) on GPU
- Support u32 arrays
- Use work-efficient algorithm (Blelloch scan or similar)
- Reference: brush-prefix-sum at `~/Projects/brush/crates/brush-prefix-sum/`

**Rust API:**
```rust
/// GPU inclusive prefix sum for u32 arrays
pub async fn prefix_sum_u32<B: Backend>(
    input: Tensor<B, 1>,
    device: &B::Device,
) -> Result<Tensor<B, 1>, String>;
```

### 3. Update Module Exports

Update `src/training/wgpu/gpu_primitives/mod.rs`:
```rust
//! GPU primitives (radix sort, prefix sum)

pub mod radix_sort;
pub mod prefix_sum;

pub use radix_sort::radix_sort_u32;
pub use prefix_sum::prefix_sum_u32;
```

### 4. Write Unit Tests

Add tests in each module to verify:
- Radix sort correctness (compare with CPU sort)
- Prefix sum correctness (compare with CPU scan)
- Edge cases: empty arrays, single element, power-of-2 sizes, non-power-of-2 sizes

Example test structure:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;
    
    #[tokio::test]
    async fn test_radix_sort_small() {
        // Test with small array
    }
    
    #[tokio::test]
    async fn test_radix_sort_large() {
        // Test with larger array
    }
}
```

## Implementation Notes

### WGSL Shader Structure

Both shaders should follow this pattern:
```wgsl
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // ... implementation
}
```

### Burn Integration

Use `burn_cubecl` to dispatch WGSL compute shaders:
- Create compute pipeline with `wgpu::ComputePipeline`
- Use `naga_oil` for shader compilation
- Integrate with burn's execution graph via `CubeTask`

### Performance Considerations

- Use workgroup size of 256 (standard for GPU compute)
- Minimize global memory access
- Use shared memory for local reductions
- Ensure coalesced memory access patterns

## Verification

After implementation, run:
```bash
cd /Users/tfjiang/Projects/RustScan/RustGS
cargo test --features gpu-wgpu gpu_primitives
```

## Reference Files

Study these brush implementations:
- `~/Projects/brush/crates/brush-sort/src/lib.rs`
- `~/Projects/brush/crates/brush-sort/src/shaders/`
- `~/Projects/brush/crates/brush-prefix-sum/src/lib.rs`
- `~/Projects/brush/crates/brush-prefix-sum/src/shaders/`

Do NOT copy brush code directly - understand the algorithm and implement it for our burn+wgpu backend.
