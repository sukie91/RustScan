# LiteGS Canonical Parity Fixture

This directory is reserved for the checked-in LiteGS convergence fixture used by
`DEFAULT_CONVERGENCE_FIXTURE_ID`.

Expected contents:

- the minimal repo-safe COLMAP fixture payload for the canonical Apple Silicon parity run
- `parity-reference.json`: a real LiteGS-derived parity report in RustGS
  `ParityHarnessReport` JSON shape

Current state:

- RustGS can already auto-discover `parity-reference.json` here
- parity reports now emit a structured gate summary and will report
  `missing_reference` until the real LiteGS reference file is checked in

Do not replace this with synthetic or RustGS-generated reference data. The goal
of this fixture is to anchor side-by-side LiteGS parity, not internal
self-comparison.
