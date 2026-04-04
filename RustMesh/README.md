# RustMesh

A high-performance mesh processing library in pure Rust, inspired by OpenMesh with SIMD optimizations.

## Features

### Core Data Structures
- **Half-Edge Mesh**: Industry-standard connectivity representation
- **SoA Layout**: Structure-of-Arrays memory layout for SIMD performance
- **Smart Handles**: Type-safe vertex/edge/face/halfedge handles

### IO Support
- **OBJ**: Complete read/write support with normals and texcoords ✅
- **OFF**: Read/write support for polygon meshes ✅
- **PLY**: Export (ASCII/Binary) ✅, import (planned)
- **Conversion API**: `from_triangle_mesh()` for easy integration ✅

### Mesh Algorithms
- Decimation (Quadric-based)
- Subdivision (Loop, Catmull-Clark, Sqrt3)
- Smoothing (Laplace, Tangential)
- Hole Filling
- Mesh Repair
- Dualization

### Performance
- SIMD-optimized operations via `glam`
- Separate x/y/z coordinate storage for vectorization
- Zero-cost abstractions

## Quick Start

```rust
use rustmesh::{RustMesh, Vec3};

// Create mesh
let mut mesh = RustMesh::new();

// Add vertices
let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));

// Add face
mesh.add_face(&[v0, v1, v2]);

// Export
rustmesh::io::write_obj(&mesh, "output.obj")?;
```

## Integration with RustSLAM

RustMesh provides seamless integration with RustSLAM's 3DGS mesh extraction:

```rust
use rustslam::fusion::MeshExtractor;
use rustmesh::RustMesh;

// Extract mesh from 3D Gaussians
let slam_mesh = extractor.extract_with_postprocessing();

// Convert to RustMesh
let vertices: Vec<Vec3> = slam_mesh.vertices.iter()
    .map(|v| v.position).collect();
let triangles: Vec<[usize; 3]> = slam_mesh.triangles.iter()
    .map(|t| t.indices).collect();
let normals: Vec<Vec3> = slam_mesh.vertices.iter()
    .map(|v| v.normal).collect();
let colors: Vec<[f32; 3]> = slam_mesh.vertices.iter()
    .map(|v| v.color).collect();

let mesh = RustMesh::from_triangle_mesh(
    &vertices,
    &triangles,
    Some(&normals),
    Some(&colors),
);

// Export
rustmesh::io::write_obj(&mesh, "scan.obj")?;
```

## Examples

```bash
# End-to-end export example
cargo run --example e2e_export

# Smart handles demo
cargo run --example smart_handles_demo

# OpenMesh benchmark parity
cargo run --release --example openmesh_compare_benchmark

# OpenMesh tutorial parity
cargo run --release --example openmesh_compare_examples

# OpenMesh smoothing comparison
cargo run --release --example openmesh_compare_smoothing

# OpenMesh decimation comparison
cargo run --release --example openmesh_compare_decimation

# OpenMesh VectorT benchmark parity
cargo run --release --example openmesh_compare_vector_benchmark
```

## Testing

```bash
cargo test --lib          # Run all tests (129 passing)
cargo test core::io::     # IO tests only
cargo bench               # Benchmarks
```

**Test Status**: 129/129 tests passing ✅

## Architecture

```
src/
├── Core/
│   ├── handles.rs          # Type-safe handles
│   ├── connectivity.rs     # RustMesh main struct
│   ├── soa_kernel.rs       # SIMD-optimized storage
│   ├── geometry.rs         # Geometric primitives
│   └── io/                 # File I/O
│       ├── obj.rs          # OBJ format ✅
│       └── ply.rs          # PLY format ✅
├── Tools/
│   ├── decimation.rs       # Mesh simplification
│   ├── subdivision.rs      # Refinement
│   ├── smoother.rs         # Smoothing
│   ├── hole_filling.rs     # Hole repair
│   └── mesh_repair.rs      # Mesh fixing
└── Utils/
    ├── circulators.rs      # Mesh traversal
    ├── quadric.rs          # Error metrics
    └── smart_ranges.rs     # Range iterators
```

## Comparison with OpenMesh

| Feature | OpenMesh | RustMesh |
|---------|----------|----------|
| Language | C++ | Rust |
| Memory Safety | Manual | Automatic |
| SIMD | Partial | Built-in (SoA) |
| Half-Edge | ✅ | ✅ |
| Smart Handles | ❌ | ✅ |
| Iterator Ranges | Limited | ✅ |
| OBJ I/O | ✅ | ✅ |
| OFF I/O | ✅ | ✅ |
| PLY I/O | ✅ | ⚠️ Export only |

## File Format Support

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| OBJ    | ✅   | ✅    | Normals, texcoords, colors |
| PLY    | ⏳   | ✅    | ASCII/Binary export |
| STL    | ⏳   | ⏳    | Placeholder created |
| OFF    | ✅   | ✅    | Polygon mesh roundtrip |

## Status

**Current Progress: ~85%**

✅ **Complete**:
- Half-edge data structure with SoA kernel
- OBJ read/write (normals, texcoords, colors)
- OFF read/write
- PLY export (ASCII/Binary)
- Conversion API (`from_triangle_mesh`)
- Mesh algorithms (decimation, subdivision, smoothing, hole filling, repair)
- Smart handles with type-safe navigation
- OpenMesh comparison examples and benchmark parity harnesses
- All core tests passing (129/129) ✅

⏳ **In Progress**:
- PLY import
- STL format support
- Performance optimizations
- Advanced attribute system

## Building

```bash
cargo build --release
cargo test
cargo bench
```

## License

[Add your license here]

## References

- [OpenMesh](https://www.openmesh.org/) - Original inspiration
- [glam](https://github.com/bitshifter/glam-rs) - SIMD math library
- [RustScan](../README.md) - Parent project
