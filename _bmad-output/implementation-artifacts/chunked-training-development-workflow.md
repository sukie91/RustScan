# Chunked Training Development Workflow

This document defines the required long-running development workflow for the RustGS chunked-training epics.

## Status Convention

Every chunked-training story in [_bmad-output/planning-artifacts/epics.md](/Users/tfjiang/Projects/RustScanner/_bmad-output/planning-artifacts/epics.md) must contain exactly one `Status:` line directly under the story title.

Allowed values:

- `todo`: story has been planned but active implementation has not started
- `in_progress`: implementation or investigation is actively underway
- `done`: code, tests, and required status updates have been completed

Rules:

- New stories must start at `todo`
- Change `todo -> in_progress` before making the first implementation edit for that story
- Change `in_progress -> done` only after code changes and relevant tests complete
- Keep [_bmad-output/implementation-artifacts/sprint-status.yaml](/Users/tfjiang/Projects/RustScanner/_bmad-output/implementation-artifacts/sprint-status.yaml) synchronized in the same change set
- If all stories inside an epic are `done`, mark the corresponding `epic-*` entry in `sprint-status.yaml` as `done`

## Handoff Rule

Whenever work stops while a story is still `in_progress` or becomes blocked, write or update a handoff note using the template in [chunked-training-handoff-template.md](/Users/tfjiang/Projects/RustScanner/_bmad-output/implementation-artifacts/chunked-training-handoff-template.md).

Minimum required handoff fields:

- current story id
- current status
- files changed
- last verified command or test
- exact next step
- blocker or known risk

## Development Sequence

For chunked-training work, use this order on every session:

1. Read [epics.md](/Users/tfjiang/Projects/RustScanner/_bmad-output/planning-artifacts/epics.md) and [sprint-status.yaml](/Users/tfjiang/Projects/RustScanner/_bmad-output/implementation-artifacts/sprint-status.yaml).
2. Confirm the next story to implement and set it to `in_progress`.
3. Make code changes.
4. Run targeted tests for the affected story.
5. Update story and epic statuses in the same change set.
6. If stopping before completion, write a handoff note.

## Completion Rule

A chunked-training story is only complete when all of the following are true:

- implementation is present in the repo
- relevant tests or verification commands were run
- `epics.md` status matches reality
- `sprint-status.yaml` status matches reality
- if work was interrupted earlier, the stale handoff note has been removed or superseded
