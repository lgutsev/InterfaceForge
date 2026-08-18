# VASP utilities

> **Verification note:** the VASP-MLFF preparation, audit, plotting, and
> restart/recovery path has been used on real runs. That experience should not
> be generalized to every static, relaxation, DOS, band, or geometry helper.

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

### POTCAR generation at submission

`iface vasp submit` verifies that a nonempty POTCAR exists before calling
`sbatch`. If it is missing, InterfaceForge generates it atomically from POSCAR
using the built-in `POTCAR_DEFS` dictionary supplied for this project. Standard
VASP 5+ species lines are preferred; the legacy convention used by
`POTCAR_gen_v2`, with element symbols in the first POSCAR line, is also
supported without modifying POSCAR.

The licensed pseudopotential files remain local. The root is resolved in this
order: `--potcar-root`, `IFACE_POTCAR_ROOT`, `VASP_PP_PATH` (including its
`potpaw_PBE` child), then `~/pot/potpaw_PBE`. Submission stops before `sbatch`
if any mapped source is missing; partial POTCAR files are never written.

```bash
export IFACE_POTCAR_ROOT=/home/$USER/pot/potpaw_PBE
iface vasp submit run/
```

The standalone generator uses the same dictionary and discovery rules:

```bash
iface vasp potcar POSCAR --root /path/to/potpaw_PBE
```

Use `--potcar-map custom.yaml` during submission, or `--map custom.yaml` with
the standalone command, only when intentionally overriding the supplied
dictionary. Existing nonempty POTCAR files are preserved.

### One-command MLFF continuation

An interrupted on-the-fly training job can be prepared and submitted in one
transactional command:

```bash
iface vasp submit run/ --ml-continue --temperature 450 --nsw 3000
```

Before `sbatch`, InterfaceForge archives the old state, copies `CONTCAR` to
`POSCAR`, promotes `ML_ABN` (falling back to `ML_AB`) to `ML_AB`, restores
`ML_MODE=train`, preserves the existing `POTIM`, applies optional
temperature/NSW overrides, and clears stale runtime outputs. Submission occurs
only if every preparation step succeeds. `--ml-continue` and
`--ml-capacity-recovery` are mutually exclusive. The older
`--recover-continue` and `--recover-capacity` spellings remain compatibility
aliases.

## Recovery

`iface vasp ml-recover` supports (with `recover` retained as an alias):

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
iface vasp submit run/ --ml-capacity-recovery
```

VASP's second bounded-storage suggestion is to increase `ML_EPS_LOW` tenfold.
For a training run without an explicit value, VASP's default is `1E-9`, so this
sets `1E-8`. It increases CUR sparsification and may reduce accuracy, especially
in multicomponent systems, so InterfaceForge requires an explicit opt-in:

```bash
iface vasp submit run/ --ml-capacity-recovery --increase-eps-low
```

The command reads an explicit existing `ML_EPS_LOW` when present, multiplies it
by ten, and refuses the recovery unless the result remains strictly below
`1E-7`, matching VASP's limit.

To retain every local reference instead, explicitly opt into the
higher-memory path by supplying a larger allocation:

```bash
iface vasp submit run/ --ml-capacity-recovery --ml-mb 12000
```

The standalone submission command prefers `runvasp.sh` when both supported
launchers are present, and falls back to campaign-generated `run.slurm`. Use
`--launcher NAME` to override the selection. The run is archived before the
capacity-recovery mutation, and the command submits only after preparation
succeeds. The recovery is refused unless OUTCAR contains a recognized
`ML_MB`/`ML_MCONF` capacity stop. An optional `--ml-mconf N` can enlarge the
configuration allocation together with `--ml-mb`.


## Work function (surface optimizations)

Two ways exist to plot the planar-averaged LOCPOT potential and estimate a
work function; both require `LVHAR = .TRUE.` in the INCAR that produced
LOCPOT, since VASP only writes the electrostatic potential (rather than just
the charge density) into LOCPOT with that tag set:

```bash
iface vasp incar static INCAR --workfunction   # or: relax ... --workfunction
```

- Interactively, from a controller/analysis environment with InterfaceForge,
  ASE, and matplotlib installed: `iface vasp workfunction LOCPOT OUTCAR
  --plot-output workfunction.png`. This path is unit tested and also writes
  a JSON summary.
- Unattended, at the end of a LONI Slurm job with no InterfaceForge install
  on the compute node: the standalone
  [`examples/vasp/workfunction/plot_workfunc.py`](../examples/vasp/workfunction/plot_workfunc.py),
  deployed to `/home/$USER/bin` and invoked directly by `runvasp.sh`/
  `runvasp_bigmem.sh` after VASP exits. The invocation is guarded on `LVHAR`
  so it is a no-op for any job that did not request a work function. See
  [`examples/vasp/workfunction/README.md`](../examples/vasp/workfunction/README.md)
  for deployment and the exact guard.

## Audit

`iface audit` interprets train, refit and run modes differently. It parses
ML_LOGFILE error/spilling records, MD progress and temperature, refit outputs,
normal termination, OOM evidence and common VASP warnings. Results are emitted
as JSON, CSV and Markdown. With the `report` or `all` installation extra, the
audit also writes `audit.xlsx`: its first `At a glance` sheet contains the key
progress, error, health and next-action columns, while `Full audit` preserves
every parsed field. The compact view is always available separately as
`audit_summary.csv`, even when Excel support is not installed.

### Standard (non-MLFF) relaxation and MD runs

A run directory without `ML_MODE`/`ML_LMLFF` is not an MLFF stage, so `iface
audit` classifies it from `IBRION`/`NSW` instead and reports it as `run_kind`
`opt`, `md`, or `static` alongside the existing `train`/`refit`/`run` modes,
in the same JSON/CSV/Markdown/xlsx outputs:

- **`opt`** (a standard ionic relaxation, `IBRION` in `1,2,3,5,6,7,8` with
  `NSW > 0`): tracks every ionic step's `energy(sigma->0)` (the same quantity
  a `grep "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" OUTCAR | grep "without
  entropy" | tail -1 | awk '{print $7}'` pipeline collects, kept as a full
  series here rather than only the final value) and the last ionic step's
  maximum atomic force from `OUTCAR`. Health is one of `converged` (VASP's
  "reached required accuracy - stopping structural energy minimisation"
  found), `not converged: reached NSW`, `energy increased on the last ionic
  step`, `incomplete or running`, or `finished; convergence marker not
  found`.
- **`md`** (`IBRION=0` with `NSW > 0`, no MLFF tags): reuses the same
  temperature series already parsed from `OSZICAR` and reports the average
  behavior of the trajectory — `temperature_mean_k`/`temperature_std_k`
  against the `TEBEG`/`TEEND` target — rather than a per-frame judgment.
  Health is `completed; average behavior reported`, `temperature drift from
  target` (mean temperature off the target by more than 15%), `incomplete or
  running`, or `no MD steps parsed`.
- **`static`** (`IBRION=-1` or `NSW` unset/zero): reports only whether the
  calculation finished normally.

As with the MLFF modes, this is triage, not a correctness proof: a
`converged` relaxation still warrants checking that `EDIFFG`/`ISIF` matched
the intended degrees of freedom, and a `completed` MD run still warrants
checking `XDATCAR` for structural drift the temperature average would not
show.

Run directories with `archive` anywhere in their relative path are excluded by
default, so recovery snapshots do not distort the current campaign summary.
Use `iface audit ROOT --include-archives` (or the same option with
`iface status`) when historical archived runs should be inspected explicitly.
InterfaceForge's internal state remains excluded, except that recovery snapshots
below `.interfaceforge/archive` are included when `--include-archives` is set.

The full audit records the active `ML_LBASIS_DISCARD`, `ML_EPS_LOW`,
`ML_IALGO_LINREG`, `ML_SION1`, and `ML_MRB2` values so that recovery and
accuracy-profile choices remain visible.

The audit parses every VASP `ERR` line as training-set RMSE and reports the
latest energy RMSE (eV/atom), force RMSE (eV/A), and stress RMSE (kbar) in the
full outputs. The latest force RMSE is also included in `audit_summary.csv`,
the `At a glance` workbook sheet, and the Markdown run table. VASP recomputes
these `ERR` quantities against all accumulated training structures whenever
the force field is generated; they are therefore fitting diagnostics, not
independent test errors. Use paired DFT/ML calculations on decorrelated,
held-out prediction snapshots for the final force RMSE and property checks.

### Perovskite readiness profile

Use the named **`perovskite`** profile for fluxional halide-perovskite MLFF
sampling campaigns:

```bash
iface audit runs/vasp --readiness-profile perovskite
iface status runs/vasp --readiness-profile perovskite
```

This profile exists because adaptive `ML_CTIFOR` is a sampling threshold, not
a convergence target. With `ML_ICRITERIA=1`, the threshold follows the recent
Bayesian-error distribution, so a stable trajectory can retain a finite
learning-event rate indefinitely. The profile therefore evaluates the latest
250 `STATUS`/BEEF records rather than requiring the whole-run learning rate,
BEEF, or training RMSE to approach zero.

A perovskite sampling plateau requires all of the following:

- at least 200 usable recent BEEF records;
- recent BEEF 95th percentile no greater than 0.03 eV/A;
- absolute difference between the mean BEEF of the two recent half-windows no
  greater than 0.002 eV/A;
- no critical events in the recent window;
- recent learning-event rate no greater than 20%.

When these checks pass, an unfinished active run is reported as
`perovskite sampling plateau reached`. A clean `ML_MB`/`ML_MCONF` capacity
stop is reported as `perovskite sampling checkpoint reached`. Reaching the
planned `NSW` budget is also treated as the end of that sampling stage, but it
still requires validation. Capacity stops that occur before the recent-window
checks pass remain incomplete and retain recovery guidance.

The option changes audit classification only; it never cancels a running job.
After a checkpoint:

1. Preserve and review `ML_ABN`.
2. Run `ML_MODE=SELECT` with a larger local-reference allocation if reselection
   is needed.
3. Run the SVD-based `ML_MODE=REFIT`.
4. Run prediction MD with `ML_ESTBLOCK=20-100` to monitor spilling.
5. Label decorrelated prediction snapshots with the identical DFT settings and
   compare forces, energies, and relevant perovskite/interface observables.

### Long-term model archives

After accepting one or more trained models, preserve their model state and
compact run provenance in one verifiable ZIP:

```bash
cd successful_runs
iface vasp archive-models

# Or choose the source root and output name explicitly:
iface vasp archive-models stored_models/perovskite_mlff_v1.zip \
    --root successful_runs
```

The command discovers directories with a nonempty `ML_AB` and stores `ML_AB`,
available `ML_ABN`/`ML_FF`/`ML_FFN`, inputs, compact MD outputs, launchers, and
logs. It adds `interfaceforge-model-archive.json`, containing per-file sizes
and SHA-256 checksums, and reports the checksum of the completed ZIP. `POTCAR`
is excluded. Use `--include-large` when `OUTCAR`, `vasprun.xml`, `XDATCAR`, and
`LOCPOT` are also worth the storage cost.

The scanner does not descend into a directory whose name contains `backup`
(case-insensitive) or begins with `X`. This prevents backup runs and earlier
`X_OutPack...` results from being recursively stored again. `CHG`, `CHGCAR`,
and `WAVECAR` are always excluded. The JSON result reports the archive size,
total uncompressed size, excluded directories, and the ten largest retained
files so unexpectedly large archives can be diagnosed directly.

With no arguments, the command scans the current folder and writes a
timestamped `MLFF_Models_<folder>_<UTC timestamp>.zip` there, matching the
working-directory behavior of the earlier `PackageOutputsMD` script without
its unconditional deletion step.

The command does not infer scientific success from VASP termination or an
error threshold. Point `--root` at models you have already accepted after the
audit and independent validation steps above.

This follows VASP's recommended separation of bounded on-the-fly sampling,
optional reselection, SVD refitting, and independent testing. The numerical
plateau thresholds are conservative InterfaceForge defaults motivated by the
approximately 0.02 eV/A Bayesian-error scale commonly attainable around
300-500 K; they are triage criteria rather than a universal scientific
acceptance standard. See the
[VASP MLFF best-practices guide](https://vasp.at/wiki/Best_practices_for_machine-learned_force_fields)
and a recent
[halide-perovskite surface workflow](https://arxiv.org/html/2502.19772v2)
that accepted the final model using force-error and physical validation after
staged bulk/surface training.

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
