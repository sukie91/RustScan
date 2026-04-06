# RustScan Documentation Index

**Updated:** 2026-04-06
**Scope:** Current workspace and `rm-opt` worktree
**Status Policy:** Use the documents in the "Canonical Docs" section as the only maintained status sources.

## Canonical Docs

| Document | Purpose |
|----------|---------|
| [Repository README](../README.md) | Short workspace overview and verified snapshot |
| [Project Overview](project-overview.md) | Stable project summary and crate roles |
| [RustMesh README](../RustMesh/README.md) | RustMesh capability matrix and user-facing crate docs |
| [RustMesh OpenMesh Progress](RustMesh-OpenMesh-Progress-2026-04-05.md) | Current `rm-opt` branch facts and verified test results |
| [RustMesh OpenMesh Roadmap](RustMesh-OpenMesh-Parity-Roadmap.md) | Remaining RustMesh parity backlog organized as epics and stories |
| [Roadmap](../ROADMAP.md) | Workspace-level forward plan |
| [Development Guide](DEVELOPMENT.md) | Build, test, and contribution workflow |
| [Architecture](ARCHITECTURE.md) | System architecture and integration notes |
| [API Reference](API.md) | Public API and usage examples |

## Compatibility Entry Points

The following files are kept only to preserve older links and reduce breakage:

- [README](README.md)
- [RustMesh README Redirect](RustMesh-README.md)
- [Roadmap Redirect](ROADMAP.md)

They should not carry independent status tables.

## Verified Snapshot

The following checks were re-verified against the current worktree on 2026-04-06:

| Area | Result |
|------|--------|
| RustMesh library tests | `250 passed; 0 failed` |
| RustMesh decimation tests | `12 passed; 0 failed` |
| RustMesh remeshing tests | `8 passed; 0 failed` |
| RustMesh VDPM tests | `16 passed; 0 failed` |
| RustMesh decimation parity example | default `OpenMeshParity` baseline matches OpenMesh for the first 10 traced steps |
| RustMesh normals benchmark | current release-mode harness shows RustMesh ahead of OpenMesh; remaining gap is semantics and refresh policy |

## Reading Order

### If you are new to the repo

1. Read [Repository README](../README.md)
2. Read [Project Overview](project-overview.md)
3. Use [Development Guide](DEVELOPMENT.md) to build and test

### If you are working on RustMesh

1. Read [RustMesh README](../RustMesh/README.md)
2. Read [RustMesh OpenMesh Progress](RustMesh-OpenMesh-Progress-2026-04-05.md)
3. Use [RustMesh OpenMesh Roadmap](RustMesh-OpenMesh-Parity-Roadmap.md) for the authoritative epic/story backlog

### If you are working on broader workspace issues

1. Read [Project Overview](project-overview.md)
2. Read [Architecture](ARCHITECTURE.md)
3. Read [Roadmap](../ROADMAP.md)

## Generated Snapshots

The following docs are structural snapshots, not status sources:

- [PROJECT-SCAN-SUMMARY](PROJECT-SCAN-SUMMARY.md)
- [Source Tree Analysis](source-tree-analysis.md)
- [project-scan-report.json](project-scan-report.json)

They are maintained to stay directionally accurate, but authoritative status belongs to the canonical docs above.
