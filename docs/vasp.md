# VASP utilities

> **Verification note:** the VASP-MLFF preparation, audit, plotting, and
> restart/recovery path has been used on real runs. That experience should not
> be generalized to every static, relaxation, DOS, band, or geometry helper.

InterfaceForge separates three concerns: generating new stages, auditing
existing runs, and mutating a failed/finished run for recovery.

For campaign-level vacuum alignment of pristine and passivated slabs, including
side-specific LOCPOT plateau checks and VBM/CBM reference subtraction, see the
[slab-alignment example](../examples/vasp/slab-alignment/README.md).

## Audit and launch a generated OPT batch

`iface vasp opt-prepare` turns an existing tree of slab-relaxation inputs into
one immutable, launchable batch. It is designed for generators such as
`notebooks/nio_m110_hydroxylation`, where every selected leaf already contains
`POSCAR`, `INCAR`, and `KPOINTS` and LONI's local `POTCAR_gen` must run inside
that directory.

Use the notebook CSV as the explicit allowlist. This is preferable to recursive
discovery when some cases (for example OH-free pilot runs) are being submitted
separately:

```bash
# No writes and no POTCAR generation.
iface vasp opt-prepare notebooks/nio_m110_hydroxylation/generated \
  --manifest notebooks/nio_m110_hydroxylation/generated/manifest_batch.csv \
  --exclude-prefix OH0 \
  --launcher-template launch_scripts/runvasp.sh \
  --dry-run

# Copy the launcher, run POTCAR_gen only where POTCAR is missing, then audit.
iface vasp opt-prepare notebooks/nio_m110_hydroxylation/generated \
  --manifest notebooks/nio_m110_hydroxylation/generated/manifest_batch.csv \
  --exclude-prefix OH0 \
  --launcher-template launch_scripts/runvasp.sh
```

Every existing nonempty POTCAR is preserved byte-for-byte. For a missing or
empty POTCAR, the default command is exactly `POTCAR_gen`, invoked without a
shell and with the leaf as its working directory. Override the executable or
arguments with `--potcar-command`, or use `--audit-only` to prohibit both
POTCAR generation and launcher copying.

`--exclude-prefix OH0` removes the pristine branch even though it is present
in `manifest_batch.csv`. The selection is recorded in `opt_manifest.json`, so
a later `opt-prepare ROOT --audit-only` automatically reuses both the source
CSV and its exclusions.

The audit requires and records:

- nonempty `INCAR`, `POSCAR`, `KPOINTS`, and `POTCAR`;
- a fixed-cell ionic optimization (`IBRION` 1/2/3, `ISIF=2`, `NSW>0`);
- `LDAUL`, `LDAUU`, and `LDAUJ` lengths matching the POSCAR species;
- a full-length `MAGMOM` whenever `ISPIN=2`;
- both frozen and mobile Selective Dynamics atoms;
- coherent `LDIPOL` / `IDIPOL` / three-component `DIPOL` settings;
- POTCAR/POSCAR species order when standard `VRHFIN` records are readable;
- an executable launcher declaring the production `vasp6/6.5.1-cpu` module by default;
- absence of VASP runtime outputs that would indicate a started calculation.

It writes `opt_manifest.json` and `opt_audit.{json,tsv,md}` at the selected
root and never submits. Review `opt_audit.md`, then preflight the complete
batch again:

```bash
iface vasp opt-launch notebooks/nio_m110_hydroxylation/generated
```

The default is a dry run. Only the explicit execution form calls `sbatch`:

```bash
iface vasp opt-launch notebooks/nio_m110_hydroxylation/generated --execute
```

`opt-launch` re-hashes every audited input and launcher, rejects runtime
outputs, and preflights every selected leaf before submitting the first job.
It writes `opt_launch.{json,tsv}` with Slurm job IDs and blocks another launch
when submitted jobs are already recorded.

## Promote an OPT tree into a Step1 preheat tree

`iface vasp step1-prepare <OPT_root>` recursively discovers finished
geometry optimizations (a local `INCAR` and nonempty `CONTCAR`) and builds a
sibling `Step1/` tree of forced-thermalization MD runs:

```bash
iface vasp step1-prepare OPT --dry-run
iface vasp step1-prepare OPT --protocol training     # NSW=400 (~0.4 ps)
iface vasp step1-prepare OPT                         # academic, NSW=2000 (~2 ps)

# fresh, unoptimized POSCAR -> Step1 100 K preheat, no WAVECAR needed
iface vasp step1-prepare OPT --source-structure POSCAR --fresh-start \
    --temperature 100 --protocol training
```

Per OPT run:

- `CONTCAR` → `POSCAR` (or another `--source-structure`, e.g. `POSCAR`);
- **electronic restart adapts per run**: when the OPT left a nonempty
  `WAVECAR` it is **hard-linked** in (verified by device/inode; falls back to
  a copy on filesystems that refuse the link) and the preheat runs
  `ISTART=1` from the converged AFM state — no `MAGMOM` re-init. When there
  is no usable `WAVECAR`, Step1 falls back to a fresh electronic start
  (`ISTART=0`, no `WAVECAR`, moments from the inherited `MAGMOM`) and prints
  a warning. `--fresh-start` forces `ISTART=0` for every run even when a
  `WAVECAR` exists; `--require-wavecar` restores the old strict behaviour and
  refuses any run without one;
- `GGA`, `ISPIN`, `LASPH`, `LNONCOLLINEAR`, `MAGMOM`, and the full active
  `LDAU*` / `LMAXMIX` block are copied **byte-for-byte** from the OPT
  `INCAR` (`LDAUPRINT` is not — it is print verbosity, not a U parameter);
- every other tag — MD-tuned electronic settings (`ENCUT=400`,
  `PREC=Normal`, `EDIFF=1E-4` (the VASP default; plenty for MD forces),
  `ALGO=Fast`, `LREAL=Auto`, `MAXMIX=40`, `NELM=60`; each step reuses the
  previous wavefunction so the SCF loop is short), the preheat MD block
  (`IBRION=0`, `SMASS=-1`,
  `NBLOCK=4`, `NSW` from the protocol), and output — comes from the packaged
  `INCAR.step1_preheat` template (override with `--template`);
- `--temperature` (default 300) sets `SYSTEM` / `TEBEG` / `TEEND`;
- `KPOINTS` and the launcher (`runvasp.sh` / `run.slurm`) are copied from
  the nearest run-specific or shared ancestor; `POTCAR` is copied too when
  present, but is **optional** — if the OPT tree has none, Step1 is prepared
  without one (with a warning) for workflows that build `POTCAR` at launch.
  The launcher is also accepted from the **current working directory** — the
  folder `step1-prepare` is run from — even when it sits above `source`
  itself (e.g. `runvasp.sh` in `OH0/` while `source` is `OH0/OPT`).

`LDAU*` array lengths and `MAGMOM` length are checked against the species /
ion count in the `CONTCAR`. `Step1/` receives `step1_manifest.json` plus
`step1_audit.{json,tsv,md}`; the audit re-derives the render, checks the
inherited tags survived, confirms `IBRION=0`/`SMASS=-1` and the expected
`ISTART` (`1` with a `WAVECAR`, `0` without), runs the
[protocol preheat-length check](#step1-switch-and-audit-the-preheat), and
notes a thin [slab vacuum](#slab-vacuum-check). Existing `Step1/` is never
overwritten; re-audit with `--audit-only`.

Then feed `Step1/` to `step2-prepare` as usual.

## Promote Step1 into a Step2 temperature series

`iface vasp step2-prepare` recursively discovers finished Step1 runs (a local
`INCAR` plus nonempty `CONTCAR`) and creates sibling `Step2_300K`,
`Step2_450K`, and `Step2_600K` trees by default. Preview the complete mapping
and all inherited Hubbard values before writing anything:

```bash
iface vasp step2-prepare Step1 --dry-run
```

Then prepare the three trees:

```bash
iface vasp step2-prepare Step1 --temperatures 300 450 600
```

The precedence rule is intentionally narrow and deterministic:

1. The packaged `INCAR.step2_dft_md` (or `--template INCAR_FINAL`) supplies
   every ordinary INCAR tag.
2. Every active `LDAU*` assignment and `LMAXMIX` is copied verbatim from that
   individual Step1 run. Template Hubbard values are ignored.
3. Every active `ISPIN`, `MAGMOM`, `LASPH`, and `LNONCOLLINEAR` line is
   likewise copied verbatim from that Step1 run, so a fixed-temperature
   Step2 MD keeps Step1's magnetic ground state instead of silently running
   non-spin-polarised. `ISTART` is *not* set (Step2 does not inherit a
   WAVECAR, so the moments still initialise from the inherited `MAGMOM`).
4. The requested temperature overrides `SYSTEM`, `TEBEG`, and `TEEND`.
5. `NSW` and the frame policy come from `--protocol` (see
   [AIMD protocols](#aimd-protocols-academic-vs-training) below): the default
   `academic` runs `NSW=5000` (5 ps) and keeps the dense every-`NBLOCK`
   stride; `training` runs `NSW=1000` (1 ps) and thins by decorrelation at
   collection time. `NBLOCK=4` either way.

Before creating any output tree, the command verifies that `LDAUL`, `LDAUU`,
and `LDAUJ` each contain one value per species in that run's `CONTCAR`, and
that an inherited `MAGMOM` has one value per ion. This is what makes mixed
atom-type orders safe: InterfaceForge never reconstructs or reorders a
Hubbard or moment array. `CONTCAR` becomes the new `POSCAR`; the nearest
run-specific or shared ancestor `KPOINTS`, `POTCAR`, `runvasp.sh`, and
`run.slurm` files are copied. Runtime outputs such as `OUTCAR`, `WAVECAR`, and
`CHGCAR` are not inherited.

Each temperature root receives `step2_manifest.json` plus
`step2_audit.json`, `step2_audit.tsv`, and `step2_audit.md`. The audit reopens
every generated file, verifies exact INCAR/structure/input hashes, checks
temperature, Hubbard values, inherited spin tags (`FAIL` if Step1 was
`ISPIN=2` but Step2 is not, or if `MAGMOM` has the wrong length), the
protocol's `NSW`, `NBLOCK=4`, and the frame policy, rejects inherited
runtime outputs, and
clearly states that submission was not performed. Existing `Step2_<T>K` roots
are never overwritten. To use another parent or the attached template
explicitly:

```bash
iface vasp step2-prepare Step1 \
  --output-root /path/to/MD_Period \
  --template INCAR_FINAL \
  --temperatures 300 450 600
```

Re-audit the existing trees at any point without preparing or submitting:

```bash
iface vasp step2-prepare Step1 --temperatures 300 450 600 --audit-only
```

Preparation never submits. After reviewing the Markdown/TSV audits, preview a
full launch of every manifest-listed daughter run. One command may cover all
three temperature roots:

```bash
iface vasp step2-launch Step2_300K Step2_450K Step2_600K
```

Only after that preview is correct, submit them:

```bash
iface vasp step2-launch Step2_300K Step2_450K Step2_600K --execute
```

`step2-launch` requires a PASS audit, rechecks the current INCAR/POSCAR/input
hashes, rejects folders with existing runtime outputs, and preflights every
root before submitting the first job. It writes `step2_launch.json` and
`step2_launch.tsv` with the Slurm job IDs and refuses a duplicate launch when
submitted jobs are already recorded. The audited `runvasp.sh` is preferred
over `run.slurm`; use `--launcher NAME` only to select another already-audited
launcher.

## AIMD protocols: `academic` vs `training`

InterfaceForge historically used one AIMD recipe for two different jobs: a
long hand-curated **Step1 preheat** followed by a **long Step2 trajectory
with dense retention** (every `NBLOCK`-th step). That is right for
publication-oriented production but wrong for MLIP training data:

- Frames a few fs apart in one trajectory are strongly autocorrelated.
  Keeping hundreds of them just resamples one thermal basin and overweights
  it in the training set.
- A long discard-equilibration burn-in throws away exactly the
  transient/reactive configurations (an anchor first contacting the surface,
  a proton transfer) that are non-equilibrium by definition and matter most
  for training.
- Training-set diversity should come from **many independent short runs** —
  different pre-relaxed starting structures (e.g. a range of hydroxylation
  patterns) — not from one long trajectory of a single structure.

`--protocol` makes the two recipes explicit. `academic` is the default.

| | `academic` | `training` |
|---|---|---|
| Purpose | Publication / production AIMD (e.g. NAMD analysis) | MLIP training-data generation |
| Step1 preheat | ~2 ps (`NSW=2000`, full thermalization) | ~0.4 ps (`NSW=400`; geometry already pre-relaxed by classical + VASP+U opt) |
| Step2 trajectory | **5 ps** (`NSW=5000`) | **1 ps** (`NSW=1000`) — a short thermal burst per start structure |
| Step2 INCAR | template as-is | template + `NWRITE=1`, `LDAUPRINT=0`, `LORBIT`/dipole stripped (small OUTCARs) |
| Step2 spin tags | `ISPIN`/`MAGMOM`/`LASPH` inherited verbatim from Step1 (both protocols) | same |
| Step2 frame retention | every `NBLOCK`-th step | spaced at the measured total-energy decorrelation time, ~15–40/run |

`NBLOCK=4` only controls how often VASP writes `XDATCAR`; forces and
energies land in `OUTCAR` every step regardless, and InterfaceForge trains
from `OUTCAR`. So `NBLOCK` is **not** the training-frame knob — `iface vasp
step2-sample` is.

### Step1: switch and audit the preheat

Step1 INCARs are hand-curated; `iface vasp step1-protocol` only retargets
their `NSW` (preheat length) and audits the result against the profile — it
never touches `SMASS`, thermostat, `TEBEG`, DFT+U, or anything else:

```bash
# Audit an existing Step1 tree against the training profile (no writes):
iface vasp step1-protocol Step1 --protocol training --audit-only

# Shorten every Step1 preheat below the tree to the training default (~0.4 ps):
iface vasp step1-protocol Step1 --protocol training

# Or one run / one file, with an explicit length:
iface vasp step1-protocol Step1/NiO_m110_Big_U46 --protocol training --nsw 200
```

The audit is `PASS`/`WARN`/`FAIL`: `FAIL` only if the INCAR is not a
fixed-temperature MD preheat at all (`IBRION≠0`, `NSW≤0`); `WARN` for a
preheat whose length is outside the profile's expected window, whose restart
hygiene is off (`LWAVE=.FALSE.` under `academic`), or — under `training` —
that carries over-converged settings for training data (`LREAL=.FALSE.`,
`PREC=Accurate/High`, `EDIFF≤1E-6`, `ADDGRID`, `LORBIT≥10`, `LDAUPRINT≥1`,
`LDIPOL`). `archive/`, `backup/`, and `X*` paths are skipped.

### Step2: prepare with a protocol, then sample by decorrelation

```bash
iface vasp step2-prepare Step1 --temperatures 300 450 600 --protocol training
```

`academic` leaves the template untouched — only the retention policy is
recorded in `step2_manifest.json` / `step2_audit.*`, and the audit adds an
informational note (never a `FAIL`) when a discovered Step1 preheat length
does not match the protocol.

`training` additionally rewrites the generated Step2 INCAR to keep OUTCARs
small, since per-step projection and occupancy tables are useless for
force/energy labels and easily push a spin-polarised MD OUTCAR past 1 GB:

- forces `NWRITE = 1` and `LDAUPRINT = 0`;
- strips `LORBIT` and the slab dipole correction (`LDIPOL`, `IDIPOL`,
  `DIPOL`) — for training data the dipole error on forces is negligible;
  use a large vacuum (≥15 Å) instead;
- the audit adds `over-converged for training` **notes** (not `FAIL`) for
  `LREAL=.FALSE.`, `PREC=Accurate/High`, `EDIFF≤1E-6`, `ADDGRID=.TRUE.`
  that a training trajectory does not need. Keep the tight settings for a
  final publication run.

Add `gzip -f OUTCAR` to the end of your Step1 `runvasp.sh` (it propagates
verbatim into every Step2 run) — a gzipped OUTCAR is ~10–20× smaller and
`iface collect` / `iface leaf-collect` now discover `OUTCAR.gz` transparently.

To switch an **already-prepared** tree without deleting it, add
`--set-protocol`:

```bash
iface vasp step2-prepare Step1 --temperatures 300 --protocol training --set-protocol --dry-run
iface vasp step2-prepare Step1 --temperatures 300 --protocol training --set-protocol
```

This re-renders every run's INCAR at the new protocol — **re-inheriting
`LDAU*`/`LMAXMIX` and the spin tags verbatim from Step1**, so the DFT+U
parameters are untouched (only print/output tags change) — and refreshes
`step2_manifest.json` / `step2_audit.*`. `POSCAR`, `KPOINTS`, `POTCAR`, and
the launcher are left alone. It refuses once a run has produced any runtime
output or recorded a submitted job; at that point the INCAR change is moot,
so just `gzip` the finished OUTCAR.

After the Step2 jobs finish, select the frames:

```bash
iface vasp step2-sample Step2_300K Step2_450K Step2_600K
```

For each run this reads the full `OSZICAR` energy series, estimates the
integrated autocorrelation time τ of the total energy (Sokal automatic
windowing), drops a short burn-in (~0.05 ps — Step1 already thermalized and
Step2 continues from its `CONTCAR`), and keeps frames spaced ~τ apart —
nudged so the count lands in 15–40. It writes `step2_sample.json`
(per-run τ, stride, burn-in, and the selected frame indices) and
`step2_sample.tsv`. Runs with no MD steps yet are reported `PENDING`.
`--dry-run` prints the plan without writing. Under `academic` the same
command just reproduces the dense `NBLOCK` stride.

### Spend a step budget on many short trajectories

`training`'s `NSW=1000` is deliberately short: from a pre-relaxed, preheated
start you only need enough steps for the electronic transient to settle
(~50 fs) plus ~15–40 decorrelation times of sampling. Spend the rest of a
fixed AIMD budget on **more starting structures**, not longer runs — each
new hydroxylation pattern / adsorbate geometry adds real diversity, another
picosecond of the same trajectory adds almost none. InterfaceForge makes
the per-trajectory defaults correct; it does not orchestrate the fan-out
over structures itself.

### Slab vacuum check

For a slab periodic in the surface plane, the one quantity that matters is
`vacuum_a` — the gap between the top of the tallest adsorbate and the
**bottom of the slab's own periodic image**, measured through the vacuum on
*both* sides of the (centred) slab. It is frame-independent: where the slab
sits in the cell, and which face an adsorbate is on, do not change it. A
tall passivant eats into the headroom the bare-slab cell had to spare, so
`step2-prepare`'s audit notes (never a `FAIL`) when the promoted structure
drops below 12 Å, with the fix command.

```bash
# report vacuum_a + slab span along the auto-detected normal
iface vasp geom vacuum Step1/NiO_m110_Big_U46_DCZ-4P/CONTCAR

# audit whole trees (several at once)
iface vasp geom vacuum Step2_300K Step2_450K Step2_600K

# --- extend: dry by default, --execute to write ---

# one file -> what would change (writes nothing)
iface vasp geom vacuum <CONTCAR> --extend 18

# one file -> a named copy
iface vasp geom vacuum <CONTCAR> --extend 18 -o POSCAR

# whole tree -> plan every thin structure (writes nothing)
iface vasp geom vacuum Step1 --extend 18

# whole tree -> actually stretch each thin structure in place
iface vasp geom vacuum Step1 --extend 18 --execute
```

`--extend` only adds empty space along the normal — every atomic position
and bond is unchanged — then re-centres the slab in the enlarged cell
(`--no-recenter` to skip; it is cosmetic and does not change `vacuum_a`). A
Step1 CONTCAR can be stretched and reused directly; the MD re-settles the
first ~50 fs anyway. A cell that already clears the target is left alone.
`--axis a|b|c` overrides the auto-detected normal.

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
- `heat`: same as `stability`, plus `ML_LHEAT=.TRUE.` for Green-Kubo
  thermal-conductivity heat-flux production (writes `ML_HEAT`). See
  [docs/mlff-interfaces.md](mlff-interfaces.md#5-ml_lheat-production-try-it-out).

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

## Work of adhesion

> **Verification note:** ported from a script the maintainer already used
> successfully outside InterfaceForge. The CLI wiring and geometry/INCAR
> logic have automated coverage; no end-to-end adhesion campaign has yet been
> run and audited through this integration specifically.

`iface vasp adhesion prepare` builds a sibling calculation tree for computing
the work of adhesion of a two-fragment interface, from either a VASP-MLFF or
a plain DFT reference run:

```bash
iface vasp adhesion prepare path/to/interface_run --method mlff \
  --lower-name TiN --upper-name SiN --distances 0.5 1 2 3 4 6 8
```

It never launches VASP and never modifies the reference directory. It splits
the reference structure into two fragments at a Cartesian *z* plane (either
explicit `--z-plane`, or auto-detected as the most balanced internal vacuum
gap), then generates:

- **`reference`**: a relative symlink back to the untouched reference
  directory — the zero-separation point.
- **`slabs/<lower-name>` and `slabs/<upper-name>`**: isolated-slab inputs
  for each fragment, with a POTCAR subset to only the species present in
  that fragment. `--slab-mode relax` (the default, `IBRION=2, ISIF=2`) lets
  each slab relax; `--slab-mode static` (`IBRION=-1`, no ionic motion)
  evaluates it at the as-cut geometry instead. Prefer `static` when the
  driving model extrapolates poorly for an isolated, vacuum-exposed
  fragment — for example an MLIP trained mostly on the interface that lets
  a bare fragment collapse into an unphysical geometry once allowed to
  relax on its own. A collapsed slab makes the work of adhesion meaningless
  regardless of how well the interface itself is described, so inspect each
  slab's `CONTCAR` under `relax` before trusting the result, especially for
  an MLIP.
- **`rigid_curve/sep_XXX.XX_A/`**: one static single-point (`IBRION=-1,
  NSW=1`) per `--distances` value, with the upper fragment and the cell's
  *c* vector both translated along the interface normal by that separation —
  preserving outer vacuum while opening a rigid gap. Supply `--curve-incar`
  to use one fixed INCAR for every point instead of the generated static one.

In `--method mlff` (the default), every generated run gets a **verified hard
link** to the reference `ML_FF`: VASP sees a regular file, but all copies
share one inode, so committee-sized models are not duplicated per run. Link
identity (same device/inode as the source) is checked immediately after
creation and before any calculation input is written; a filesystem that
silently copies instead of linking (some network filesystems do) fails the
command rather than quietly consuming the storage many times over.
`--method dft` strips every `ML_` INCAR tag and creates no `ML_FF`.

By default, whichever launcher `iface vasp submit` would pick for the
reference directory — `runvasp.sh`, else `run.slurm` — is copied into every
generated slab and rigid-curve directory, so each is independently
submittable without manually copying a launcher in first. Use `--launcher
NAME` to propagate a specific script instead of auto-detecting, or
`--no-launcher` to skip propagation entirely. Neither the reference
directory nor its launcher is ever modified; each generated directory gets
its own copy.

`manifest.json` in the output directory records the split plane, detected
gap, interface area, every slab and curve-point directory, and the formula
to combine converged energies once VASP has run:

```text
W_ad = (E_lower_slab + E_upper_slab - E_reference) / interface_area_A2
```

`one_interface_J_m2_per_eV` in the manifest is the eV/Å² → J/m² conversion
factor for that specific interface area. `prepare` only prepares inputs and
never computes that division itself.

A guard distance (`--guard`, default `0.20` Å) refuses to cut through an atom
that sits too close to the split plane; `--min-side-fraction` bounds how
unbalanced an auto-detected split may be. The command refuses to write into
an existing output directory rather than overwrite prior calculations.

### Auditing finished calculations

Once the propagated launcher (or your own) has run the reference, both
slabs, and the rigid-curve points, read the results back with:

```bash
iface vasp adhesion audit path/to/interface_run_adhesion_dft
```

This audits the reference and every generated directory with the same
mode-aware OUTCAR/OSZICAR parsing `iface audit` uses (so it works whether
each run was MLFF or DFT), taking each run's converged energy as
`energy(sigma->0)` from its last completed ionic step. For whichever runs
have already finished, it assembles the small CSVs
`iface validate adhesion` and `iface validate separation` expect and calls
that existing, unit-tested math directly rather than duplicating it — so
those two commands remain available separately for CSVs assembled by hand
or from a non-`adhesion`-prepared campaign. Partial results are reported for
an in-progress campaign: the work of adhesion needs the reference and both
slabs to have finished; the separation curve is computed from whichever
rigid-curve points have finished, and `rigid_curve_points_ready`/
`rigid_curve_points_total` in the output say how many that is. This never
launches or resubmits VASP.

Output lands under `<output_dir>/audit/`:

```text
adhesion_audit.json       # full per-run status, work_of_adhesion, separation_curve
adhesion_audit.md         # human-readable summary table
adhesion_energies.csv     # assembled input to iface validate adhesion (when ready)
adhesion_results.csv      # iface validate adhesion's own output
separation_energies.csv   # assembled input to iface validate separation (when any point is ready)
separation_curve.csv      # iface validate separation's own output
```

`iface validate adhesion`/`iface validate separation` remain independently
usable for a CSV assembled by hand or from a campaign that did not go
through `adhesion prepare` — `adhesion audit` is a convenience on top, not a
replacement:

- **`iface validate adhesion energies.csv results.csv`** expects columns
  `area_a2`, `interface_energy_ev`, `slab_a_energy_ev`, `slab_b_energy_ev`,
  and optional `interface_sigma_ev`/`slab_a_sigma_ev`/`slab_b_sigma_ev`.
- **`iface validate separation energies.csv results.csv`** expects columns
  `model`, `distance_a`, `energy_ev`, and optional `area_a2`.

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

# Skip named immediate children:
iface vasp archive-models --exclude-folders old_300 test_run rejected_model

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

The default scan depth is the current folder plus its immediate child folders,
matching the original shell script's `*/` loop. It does not inspect daughter
folders below those children. Add exact names with
`--exclude-folders NAME [NAME ...]`; the option may be repeated. Matching is
case-sensitive and applies at every scanned depth. `--recursive` opts back into
deeper discovery when it is genuinely needed.

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
