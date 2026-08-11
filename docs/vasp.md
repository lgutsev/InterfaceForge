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
- `discard`: continue after a local-reference capacity stop while keeping
  memory bounded with `ML_LBASIS_DISCARD=.TRUE.`.
- `expand`: continue after a recognized `ML_MB`/`ML_MCONF` capacity stop.
- `refit`: produce a fast potential from the accumulated database.
- `stability`: promote a verified `ML_FFN` to `ML_FF` and run prediction MD.

Every operation snapshots recoverable run files under
`.interfaceforge/archive/` before mutation. A stability run rejects an
`ML_FFN` that does not advertise `ML_LFAST=true`.

When VASP stops with `Not enough storage reserved for local reference
configurations`, the safer bounded-memory recovery is to keep the existing
`ML_MB`, copy `ML_ABN` to `ML_AB`, continue from `CONTCAR`, and enable
`ML_LBASIS_DISCARD=.TRUE.`. Although this is the current VASP default, the
capacity stop indicates that discarding was inactive for the failed run (for
example, explicitly set to `.FALSE.`). InterfaceForge uses `.TRUE.` for the
default recovery:

```bash
iface vasp submit run/ --recover-capacity
```

VASP's second bounded-storage suggestion is to increase `ML_EPS_LOW` tenfold.
For a training run without an explicit value, VASP's default is `1E-9`, so this
sets `1E-8`. It increases CUR sparsification and may reduce accuracy, especially
in multicomponent systems, so InterfaceForge requires an explicit opt-in:

```bash
iface vasp submit run/ --recover-capacity --increase-eps-low
```

The command reads an explicit existing `ML_EPS_LOW` when present, multiplies it
by ten, and refuses the recovery unless the result remains strictly below
`1E-7`, matching VASP's limit.

To retain every local reference instead, explicitly opt into the
higher-memory path by supplying a larger allocation:

```bash
iface vasp submit run/ --recover-capacity --ml-mb 12000
```

The standalone submission command prefers `runvasp.sh` when both supported
launchers are present, and falls back to campaign-generated `run.slurm`. Use
`--launcher NAME` to override the selection. The run is archived before the
capacity-recovery mutation, and the command submits only after preparation
succeeds. The recovery is refused unless OUTCAR contains a recognized
`ML_MB`/`ML_MCONF` capacity stop. An optional `--ml-mconf N` can enlarge the
configuration allocation together with `--ml-mb`.


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
InterfaceForge's internal state remains excluded, except that recovery snapshots
below `.interfaceforge/archive` are included when `--include-archives` is set.

The full audit records the active `ML_LBASIS_DISCARD`, `ML_EPS_LOW`,
`ML_IALGO_LINREG`, `ML_SION1`, and `ML_MRB2` values so that recovery and
accuracy-profile choices remain visible.

## Bayesian-error plots

`iface vasp beef-plot ROOT` restores the campaign-level diagnostic formerly
provided by `ML_BayesianErrorPlot`. It reads every active `ML_LOGFILE`, uses
each run's `POTIM` to convert steps to femtoseconds, and writes:

- `ML_BayesianErrorPlot_campaign.png`, with one panel per run;
- `ML_BayesianErrorPlot_campaign.csv`, containing the complete plotted series.

Each panel shows the maximum Bayesian force-error estimate, contemporaneous
`ML_CTIFOR`, and learning/critical event markers. Archive directories are
excluded by default; use `--include-archives` to include them. Add
`--individual` for separate per-run PNGs. Plotting requires the `report` or
`all` installation extra.

Recovery commands move the preceding `ML_LOGFILE` below
`.interfaceforge/archive` before starting a clean segment. Consequently, use
`--include-archives` to plot the completed historical segments while a newly
prepared continuation has not yet written BEEF records.

```bash
iface vasp beef-plot runs/vasp
iface vasp beef-plot runs/vasp --individual
iface vasp beef-plot runs/vasp --include-archives -o historical_beef.png
```

The audit gives triage guidance, not a scientific acceptance certificate.
Held-out DFT energies/forces and interface observables remain required.
