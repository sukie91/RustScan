# RustScan Scan Summary

**Updated:** 2026-04-06
**Type:** Workspace snapshot summary

This file is a lightweight structural summary, not a primary status source.

## Workspace Snapshot

- Total Rust files in the repository snapshot: `195`
- RustMesh source files: `28`
- RustMesh examples: `36`
- RustSLAM source files: `70`
- RustSLAM examples: `5`
- RustGS source files: `26`
- RustViewer source files: `15`
- `rustscan-types` source files: `4`

## Verification Snapshot

- RustMesh library tests: `234 passed; 0 failed`
- RustMesh remeshing tests: `8 passed; 0 failed`
- RustMesh VDPM tests: `9 passed; 0 failed`
- RustMesh decimation trace example: matches OpenMesh for the first 10 traced steps under the default baseline
- RustMesh normals benchmark: current release-mode harness shows RustMesh ahead of OpenMesh; remaining gap is semantic rather than raw-speed driven

## Canonical Documents

Use these docs for current status:

- [../README.md](../README.md)
- [index.md](index.md)
- [../RustMesh/README.md](../RustMesh/README.md)
- [RustMesh-OpenMesh-Progress-2026-04-05.md](RustMesh-OpenMesh-Progress-2026-04-05.md)
- [../ROADMAP.md](../ROADMAP.md)

## Compatibility Notes

- `docs/README.md`, `docs/RustMesh-README.md`, and `docs/ROADMAP.md` now exist only as redirect-style compatibility files.
- Older planning detail that duplicated completed parity-debugging steps was folded into the progress and roadmap docs.
