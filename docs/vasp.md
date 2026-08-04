# VASP utilities

InterfaceForge separates three concerns: generating new stages, auditing
existing runs, and mutating a failed/finished run for recovery.

## Preparation

`iface prepare` creates `train`, `refit` and `stability` directories for every
configured system. It copies only explicit inputs, updates MLFF tags while
preserving unrelated INCAR content, and writes scheduler jobs without
submitting them.

Standalone `iface vasp incar` presets cover static, relaxation, MD and DOS
control tags. They intentionally leave scientific convergence choices to the
campaign.

Geometry tools use ASE and write VASP 5 structures with Selective Dynamics
where applicable. `convert --cell-from` is intended for XYZ files whose
periodic cell was lost.

## Recovery

`iface vasp recover` supports:

- `continue`: continue on-the-fly training from `ML_ABN` or `ML_AB`.
- `expand`: continue after a recognized `ML_MB`/`ML_MCONF` capacity stop.
- `refit`: produce a fast potential from the accumulated database.
- `stability`: promote a verified `ML_FFN` to `ML_FF` and run prediction MD.

Every operation snapshots recoverable run files under
`.interfaceforge/archive/` before mutation. A stability run rejects an
`ML_FFN` that does not advertise `ML_LFAST=true`.

## Audit

`iface audit` interprets train, refit and run modes differently. It parses
ML_LOGFILE error/spilling records, MD progress and temperature, refit outputs,
normal termination, OOM evidence and common VASP warnings. Results are emitted
as JSON, CSV and Markdown. With the `report` or `all` installation extra, the
audit also writes `audit.xlsx`: its first `At a glance` sheet contains the key
progress, error, health and next-action columns, while `Full audit` preserves
every parsed field. The compact view is always available separately as
`audit_summary.csv`, even when Excel support is not installed.

Run directories with `archive` anywhere in their relative path are excluded by
default, so recovery snapshots do not distort the current campaign summary.
Use `iface audit ROOT --include-archives` (or the same option with
`iface status`) when historical archived runs should be inspected explicitly.
InterfaceForge's internal `.interfaceforge` state directory is always excluded.

The audit gives triage guidance, not a scientific acceptance certificate.
Held-out DFT energies/forces and interface observables remain required.
