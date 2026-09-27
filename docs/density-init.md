# Neural density initialization (optional, opt-in)

`iface vasp initialize-density` gives a *fresh* VASP calculation a physically
informed starting `CHGCAR`. It uses the Complete Neural Electronic Initializer of
Ærtebjerg, Elsborg & Bhowmik, *Complete Neural Electronic Initialization
Accelerates Materials DFT* (arXiv:2609.21759; code:
[aerte/neural_paw_dft](https://github.com/aerte/neural_paw_dft)). The goal is
fewer SCF iterations, and therefore less wall time, when generating DFT
training labels.

> **Status: automated-test only.** The adapter, file-safety logic, magnetism
> guards and benchmark tooling are covered by unit tests with a mocked
> `neural_paw_dft`. Real neural inference has **not** been run on a
> VASP-compatible case through InterfaceForge, and no speed-up is claimed.
> See [Verification and maturity](verification.md) and
> [Validation](#validation-methodology) below.

## What it does, and what it does not do

| It does | It does **not** |
|---|---|
| Predict the smooth valence density ρ̃⁺ (ELECTRAFI) and the PAW augmentation occupancies (AugNet) for the run's own POSCAR, writing a VASP-readable `CHGCAR` on the run's own FFT grid | Change the DFT method: INCAR, POTCAR, KPOINTS, functional, +U, ENCUT, smearing and so on are untouched |
| Set `ICHARG = 1` (its only INCAR edit) so VASP reads that density | Replace the SCF: VASP still converges the density self-consistently, and energies, forces and stresses are VASP's |
| Keep the INCAR `MAGMOM` authoritative by default, including signed AFM-II patterns | Choose the magnetic ordering for you. CHGNet moments are used only on explicit request, and never for mixed-sign INCARs |
| Record full provenance in `density_init.json` | Become part of the MLIP: CHGNet and the initializer are **not** part of the final MACE/DeePMD/NequIP/Allegro model |

The feature speeds up the **generation of DFT training labels**. A
geometry-only MLIP trained on those labels still learns the DFT magnetic
branch only implicitly, through the energies and forces of whatever state
VASP converged to. That is why the magnetic branch has to be controlled at
the DFT level (see [Magnetism](#magnetism-and-the-nio-afm-ii-caveat)).

The default everywhere is the standard VASP start; neural initialization only
runs when requested.

## Installation

`neural_paw_dft` is **not** an InterfaceForge dependency, and there is
deliberately no `interfaceforge[neural-paw]` extra:

- it pins `mace-torch==0.3.16`, `e3nn==0.4.4`, `torch>=2.4.1` and `pykeops`,
  which conflicts with `interfaceforge[mace-roi]` (`mace-torch>=0.3.17`);
- it is not published on PyPI, and a git direct reference cannot be put in a
  PyPI-published extra;
- it is licensed **CC-BY-NC-4.0** (non-commercial). Check that this fits your
  use.

Install it instead in its **own** environment. InterfaceForge talks to it
through a small stdlib-only worker script run with that interpreter
(`--ndi-python` or `$IFACE_NDI_PYTHON`). A crash, OOM or import problem in the
ML stack therefore never reaches the orchestration process or the run's
inputs.

```bash
examples/density-init/create_ndi_env.sh ~/envs/ndi cu124     # or: cpu
export IFACE_NDI_PYTHON=~/envs/ndi/bin/python

# On a node WITH network (weights come from huggingface.co/faerte/neural_paw_dft):
iface vasp density-init-probe --prefetch --weights-dir /shared/ndi_weights
```

HPC portability notes. These are documented rather than worked around:

- **Pinned upstream commit.** The adapter targets the packaged
  `neural_paw_dft.pipeline` API on `main` (commit `bde513b`). The paper tag
  `v1-paper` predates that API and will not work. `density-init-probe` reports
  an incompatible version as unavailable.
- **Weights download on first use.** Compute nodes are often offline, so
  prefetch on a login node and pass `--weights-dir` (or set `NDI_WEIGHTS_DIR`).
- **pykeops JIT-compiles** on first use and needs a C++ compiler on the node
  that runs inference (for example `module load gcc`). Run one inference there
  before a campaign.
- **CUDA.** Choose the torch wheel that matches the cluster's driver. On CPU
  nodes use `--device cpu`; inference is slower there, and that time is counted
  as overhead.

## Quick start

```bash
# Plan only: validates inputs, resolves grid/NELECT/LMAXMIX/magnetism, probes backend; writes nothing
iface vasp initialize-density RUN_DIR --backend neural-paw --magmom-source incar \
    --grid-from /path/to/same-settings/OUTCAR --dry-run

# Generate, stage, promote, set ICHARG = 1, write density_init.json
iface vasp initialize-density RUN_DIR --backend neural-paw --magmom-source incar \
    --grid 96 96 240 --device cuda

# After VASP finished: convergence + magnetic-branch audit
iface vasp density-init-audit RUN_DIR
```

The run directory must contain `INCAR`, `POSCAR`, `POTCAR` and `KPOINTS`.
`KPOINTS` may be omitted when the INCAR sets `KSPACING`. The command refuses
to run in a directory that already contains VASP output (`OUTCAR`, `OSZICAR`,
`vasprun.xml`). It also refuses a nonempty `WAVECAR` with `ISTART ≠ 0`,
because converged orbitals are a better start, and non-SCF `ICHARG ≥ 10`, as
well as non-collinear/SOC INCARs.

### FFT grid

The CHGCAR must be on VASP's fine grid (`NGXF NGYF NGZF`). The grid is taken
from, in order of priority:

1. `--grid NGXF NGYF NGZF`;
2. `NGXF/NGYF/NGZF` in the INCAR;
3. `--grid-from` an OUTCAR or CHGCAR of a run with identical
   `ENCUT/PREC/ENAUG/NG*` and an identical lattice;
4. `--grid-dry-run-command 'srun vasp_std'`, which runs one throw-away
   `NELM = 1` step on copies and reads VASP's own grid. Its time counts as
   initialization overhead.

InterfaceForge does not reimplement VASP's grid selection.

## Magnetism and the NiO AFM-II caveat

The upstream spin model (spin-ELECTRAFI) constrains the predicted
spin-difference density with a **net** moment: in the paper's convention, the
sum of unsigned CHGNet site moments. An antiferromagnet such as NiO AFM-II has
a net moment of ≈ 0, so that constraint cannot represent it. If the CHGCAR
does contain a magnetization density, VASP uses `MAGMOM` only for symmetry
and takes the initial moments from the CHGCAR
([VASP wiki: MAGMOM](https://vasp.at/wiki/MAGMOM)). If the CHGCAR has **no**
magnetization density, VASP initializes the on-site moments from `MAGMOM`.
That is exactly the `ICHARG = 1`, `MAGMOM = m -m` restart the wiki recommends
for AFM states.

InterfaceForge therefore applies this policy:

| `--magmom-source` | INCAR `MAGMOM` | `--spin-channel auto` result | Initial moments in VASP come from |
|---|---|---|---|
| `incar` (default) | mixed-sign (NiO AFM-II, ferri) | **charge-only** CHGCAR; `--spin-channel model` is refused | the signed INCAR `MAGMOM`, exactly |
| `incar` | sign-uniform (FM) | model spin channel constrained to the INCAR moments and their net moment | CHGCAR magnetization (model spatial distribution) |
| `incar` | absent, all-zero, or `ISPIN = 1` | charge-only (with a warning when `ISPIN = 2` and `MAGMOM` is absent) | `MAGMOM` / VASP default |
| `initializer` | mixed-sign | **refused** | — |
| `initializer` | absent or sign-uniform | CHGNet moments + model spin channel (explicit opt-in, warned) | CHGCAR magnetization |
| `none` | `ISPIN = 1` only | charge-only | — |

With `--magmom-source incar`:

- the INCAR is never rewritten beyond `ICHARG`, and `MAGMOM` signs and site
  ordering are byte-identical afterwards;
- CHGNet is not run at all. The report records
  `automatic_magnetic_initialization: "overridden: INCAR MAGMOM is authoritative…"`;
- where a spin channel is allowed, the INCAR moments are passed to the upstream
  `Pipeline.predict(site_moments=…)`, which is the only place its API accepts
  user moments.

The price of the charge-only seed is that the paper's ablations show valence
density *without* spin initialization can erase or reverse the speed-up for
magnetic systems. The NiO rows of the pilot benchmark exist to measure whether
a charge-only seed plus signed `MAGMOM` still helps. Do not assume it does.

After VASP, always audit the branch:

```bash
iface vasp density-init-audit RUN_DIR      # PASS / WARN / FAIL / INCOMPLETE, writes density_init_audit.json
```

The audit uses sites with `|MAGMOM| ≥ 0.5` to define the pattern. A converged
site with `|m| < 0.5 μB` counts as quenched. A global sign flip of every site
is the same collinear state and is reported as `PRESERVED_GLOBAL_FLIP`. The
INCAR needs `LORBIT ≥ 10` for local moments to appear in the OUTCAR.

## POTCAR compatibility

AugNet predicts augmentation occupancies in the projector layout of the
**Materials Project POTCAR set** it was trained on (for example `Ni_pv`,
`Ti_pv`, `Fe_pv`, `Sr_sv`, `O`). The worker parses the projector l-channels
from each `Non local Part` of the run's POTCAR and:

- **always refuses** a projector-set mismatch. For example `Ni` (d, s) versus
  `Ni_pv` (p, d, s) changes the size of the augmentation block VASP reads, so
  the result would be silently wrong;
- refuses a different dataset name with identical projectors unless
  `--allow-potcar-variant` is given (out of distribution, and warned).

InterfaceForge's own default map (`templates/potcar_pbe_54.yaml`) differs from
the MP set for several elements used in this repository: `Ni`, `Ti_sv`, `Fe`,
`Co`, `Zr_sv`, `Nb_sv`. **NiO runs prepared with the default `Ni` POTCAR will
be refused.** Using the neural initializer on them would mean changing the
POTCAR, which changes the DFT method, and this feature must not do that. Use
it only for campaigns whose POTCARs already match, or decide on the POTCAR on
scientific grounds first.

A refused POTCAR is recorded as `failure_code: UNSUPPORTED_POTCAR_SCHEMA`
(projector mismatch) or `POTCAR_VARIANT_NOT_ALLOWED` (name differs, projectors
agree) in `density_init.json`, so a status audit can report it without logs.

### POTCAR selection and provenance

POTCAR selection belongs to the VASP workflow, never to the initializer:

```text
workflow/profile -> choose POTCAR mapping -> POTCAR_gen -> actual POTCAR
                 -> density-init compatibility check -> neural seed if compatible
```

The initializer inspects the POTCAR it is given and either seeds it or refuses
it; it never regenerates or substitutes one. For a controlled test with the
Materials Project-compatible mapping, generate the POTCAR explicitly, e.g.
`POTCAR_gen --defs /path/to/POTCAR_DEFS_MP.txt` (plain `POTCAR_gen` keeps the
production mapping), and keep such runs out of production training datasets:
`Ni → Ni_pv` changes the reference electronic structure.

Every run's POTCAR provenance is recorded: `density_init.json` → `potcar` and
each `step1_manifest.json` run → `potcar`, with `potcar_sha256` and the
dataset per element (`potcar_variants`) read from the POTCAR itself, which is
authoritative. Declare how it was generated with `--potcar-definitions FILE`
(and optionally `--potcar-generator PATH`) on `initialize-density`,
`step1-prepare` or `density-init-bench prepare` (or `potcar_definitions:` per
pilot case). The declared file is recorded with its SHA-256 and checked
against the POTCAR: a declaration that disagrees with the actual datasets is
refused before anything is written. `$POTCAR_DEFS` is deliberately not read,
since a job environment may export it for an unrelated POTCAR.

```json
"potcar": {
  "potcar_generator": "/home/user/bin/POTCAR_gen",
  "potcar_definitions": "/home/user/bin/POTCAR_DEFS_MP.txt",
  "potcar_definitions_sha256": "…",
  "potcar_sha256": "…",
  "potcar_variants": {"Ni": "Ni_pv", "O": "O"},
  "declared_variants": {"Ni": "Ni_pv", "O": "O"},
  "consistent_with_definitions": true
}
```

## File safety

| Situation | Behavior |
|---|---|
| Generated density | Always written first to `CHGCAR.neural_init` (staged), then promoted |
| No `CHGCAR` | Promoted to `CHGCAR`; INCAR gets `ICHARG = 1`; original saved as `INCAR.pre_density_init` |
| Foreign `CHGCAR` (not promoted by InterfaceForge) | **Kept untouched**; status `STAGED`, INCAR untouched |
| `--overwrite` | Foreign `CHGCAR` renamed `CHGCAR.pre_density_init.<UTC time>` (never deleted), then promotion |
| `--stage-only` | Never promotes and never edits the INCAR |
| `--no-set-icharg` | Promotes but leaves the INCAR alone, with a warning that VASP ignores the CHGCAR unless `ICHARG = 1` |
| Re-run, inputs unchanged | `ALREADY_INITIALIZED`: nothing regenerated or written |
| Re-run after POSCAR/INCAR/POTCAR/KPOINTS changed | Previously promoted CHGCAR demoted to `CHGCAR.neural_init.stale-<time>` and `ICHARG` restored, then regeneration |
| Any failure after validation | Promotion and INCAR edits rolled back; `density_init.json` records `status: FAILED` and the error |
| Concurrent invocation | Refused through the `.density_init.lock` file |

The initializer only ever receives *copies* of the inputs, in a private
scratch directory. The originals are re-hashed afterwards and restored from a
pristine copy if anything changed. The INCAR edit is verified to change
exactly one active tag.

## Provenance (`density_init.json`)

Format `interfaceforge-density-init`, schema version 1. The key fields are:

- `status` (`PROMOTED` / `STAGED` / `STANDARD_START` / `FAILED`), `active`,
  and `mode`;
- `backend_info`: package, version, git commit and source URL, Python, torch,
  device, and each model's registry name, resolved path and SHA-256;
- `interfaceforge`: version, commit, and whether the tree is dirty;
- `structure`: POSCAR SHA-256, geometry-only SHA-256, and composition;
- `inputs.sha256_before/after`, `incar_changes`, and `grid`, `nelect` and
  `lmaxmix`, each with its source;
- `magnetism`: `magmom_source`, the INCAR moments (expanded and signed),
  order, `spin_channel_written`, `initializer_moments`,
  `automatic_magnetic_initialization`, and `initial_moments_in_vasp_from`;
- `files`: staged, promoted, backups, demotions, log, and SHA-256s;
- `timing`: `model_load_s`, `inference_s`, `write_s`, `grid_discovery_s`,
  `subprocess_s`, and `total_s`/`overhead_s`, which is what the benchmark
  charges to the neural arm;
- `potcar`: POTCAR provenance (see [POTCAR selection and provenance](#potcar-selection-and-provenance));
- `warnings`, plus `error` and `failure_code` (`UNSUPPORTED_POTCAR_SCHEMA`,
  `POTCAR_VARIANT_NOT_ALLOWED`, `BACKEND_UNAVAILABLE`, or null) when the status
  is `FAILED`.

Example (mocked backend):
[`examples/density-init/example_density_init_nio_afm2.json`](../examples/density-init/example_density_init_nio_afm2.json).

## Using it from `step1-prepare` and launch workflows

```bash
iface vasp step1-prepare OPT --fresh-start --density-init neural-paw \
    --density-init-ndi-python ~/envs/ndi/bin/python --density-init-weights-dir /shared/ndi_weights
iface vasp step1-prepare OPT --profile nio --density-init neural-paw        # seeds the preconditioner SCF
iface vasp step1-launch Step1 --execute
```

### HPC integration strategy (technical note)

Inference is **never** run during preparation. Preparation nodes may lack the
GPU, CUDA toolkit, compiler, network or Python environment the initializer
needs, and a prediction made there would not be tied to the job that uses it.
Instead:

1. `step1-prepare --density-init neural-paw` validates the magnetic policy
   when it prepares each run, so for example `--density-init-magmom-source
   initializer` on an AFM INCAR fails immediately. It then records the request
   in `step1_manifest.json`, both at the top level and per run, and wraps the
   launcher. The Step1 audit checks the wrapped launcher byte for byte.
2. Inside the job, directly before the fresh-start SCF, the launcher runs
   `… -m interfaceforge vasp initialize-density . …`:
   - it runs before the VASP line for `ISTART = 0` runs;
   - under `--precondition` (the NiO profile) it runs inside the `precondition/`
     subshell, so it seeds the NSW = 0 static SCF, and the MD then restarts from
     its WAVECAR;
   - `ISTART = 1` WAVECAR restarts are left alone and recorded as skipped.
3. The grid is reused from the OPT OUTCAR when the rendered INCAR's
   grid-deciding tags and the lattice match. Otherwise the job does a
   `NELM = 1` dry run with the launcher's own VASP command. Step1 renders
   `ENCUT = 400/PREC = Normal`, so the dry run is the common case there.
4. On failure (`--density-init-on-failure standard`, the default), the job
   prints a message and VASP proceeds with its **standard start**. The inputs
   are guaranteed untouched, so a broken ML stack costs seconds rather than an
   allocation. `abort` stops the job with exit code 3 instead, also under
   `--precondition` (the job stops before the preconditioner's VASP call and
   before the MD). Either way the launcher writes `density_init_fallback.json`
   (`action`, the hook's `exit_code`, UTC time) next to the seeded inputs; a
   successful hook removes a stale one. The hook is written
   `hook || density_init_rc=$?`, so launchers running under `set -e` still
   reach the fallback branch.

### Requested vs executed (`iface vasp step1-status`)

`step1-status` (and `--json`) reports, per run, whether initialization was
requested, compatible and actually executed, so campaign audits do not depend
on job logs:

| Field | Meaning |
|---|---|
| `density_init_requested` | `neural-paw` or `standard` (from `step1_manifest.json`) |
| `density_init_compatible` | `true` once the worker accepted the POTCAR, `false` for a refused POTCAR, `null` while unknown |
| `density_init_executed` | `true` only when a generated density was promoted and VASP reads it (`ICHARG = 1`), with no fallback record |
| `density_init_status` | `PROMOTED`, `STAGED`, `PENDING` (job not run yet), `SKIPPED` (ISTART = 1 restart), `UNSUPPORTED_POTCAR_SCHEMA`, `POTCAR_VARIANT_NOT_ALLOWED`, `BACKEND_UNAVAILABLE`, `FAILED`, or `NOT_REQUESTED` |
| `density_init_fallback_occurred` / `_action` | whether the launcher fell back (`standard`) or aborted (`abort`) |

```json
{"density_init_requested": "neural-paw", "density_init_compatible": false,
 "density_init_executed": false, "density_init_status": "UNSUPPORTED_POTCAR_SCHEMA",
 "density_init_fallback_occurred": true, "density_init_fallback_action": "standard"}
```

The payload also carries a `density_init_tally`. For `--precondition` runs the
report and fallback record live in `precondition/`, where they are looked up
too. The detailed `density_init.json` is kept unchanged.

A separate GPU pre-step is also supported. Run `iface vasp initialize-density`
on each prepared run from a GPU job, then `iface vasp step1-launch`. The launch
preflight accepts the INCAR hash change only when `density_init.json` proves
the prepared-INCAR → `ICHARG = 1` transition (kind `prepared+density-init`).
Any other INCAR drift is still refused.

## Validation methodology

A CHGCAR being produced is **not** success. Success is measured on a controlled
paired benchmark: the same structure, the same INCAR/POTCAR/KPOINTS, the same
hardware, and only the start differing.

```bash
iface vasp density-init-bench prepare examples/density-init/pilot.yaml BENCH --dry-run
iface vasp density-init-bench prepare examples/density-init/pilot.yaml BENCH
# submit every BENCH/<case>/{standard,neural} on the same partition/node type
iface vasp density-init-bench compare BENCH                 # JSON + Markdown + TSV
iface vasp density-init-bench compare BENCH --markdown      # per-case tables to stdout
```

- `prepare` copies **completed** calculations (the final `CONTCAR`) into small
  directories. Nothing expensive is rerun in place. `mode: static` gives both
  arms `NSW = 0, IBRION = -1, ISTART = 0`, turns LWAVE/LCHARG off, and sets
  `ISIF = 2` if the source had `ISIF = 0`, so that forces *and* stress exist.
  The standard arm gets `ICHARG = 2` and a `STANDARD_START` provenance record.
  The neural arm's launcher runs the initializer before VASP with
  `on_failure = abort`, so an initializer failure counts as a neural failure
  rather than silently becoming a standard run.
- Both arms get the *same* POSCAR/POTCAR/KPOINTS (hash-checked at `prepare`;
  their SHA-256s and the POTCAR provenance go into the manifest). `compare`
  re-hashes them: if either arm's input was replaced afterwards (for example
  a regenerated POTCAR), the case is `INPUT_MISMATCH` and no acceptance
  criterion is evaluated. A `density_init_fallback.json` in the neural arm
  makes it a neural failure even if a report says `PROMOTED`.
- `compare` reports, per case, the table below, plus a verdict:
  `SAME_SOLUTION`, `DIFFERENT_SOLUTION`, `NEURAL_FAILED`, `STANDARD_FAILED`,
  `BOTH_FAILED`, `INPUT_MISMATCH` or `INCOMPLETE`.

| Metric | Standard start | Neural start | Difference |
| --- | ---: | ---: | ---: |
| SCF iterations | | | |
| VASP wall time | | | |
| Inference time | 0 | | |
| Total wall time | | | |
| Final energy | | | |
| Max force difference | | | |
| Stress difference | | | |
| Magnetic state | | | |
| Electronic convergence | | | |

"SCF iterations" counts the electronic steps of the seeded SCF, taken from the
OSZICAR. It measures **SCF acceleration**. "Total wall time" is VASP's
`Elapsed time` plus *all* initializer overhead: model loading, inference,
CHGCAR writing and any grid dry run. It measures **end-to-end acceleration**.
The two are reported as separate speed-ups, because a large SCF reduction can
still lose end to end once inference is paid for.

Default tolerances are 1e-4 eV/atom (energy), 5e-3 eV/Å (maximum per-atom
force-vector difference), 0.5 kB (maximum stress component) and 0.05 μB
(local moments). Override them with `--energy-tol`, `--force-tol`,
`--stress-tol` and `--moment-tol`. Choose them relative to your `EDIFF`.

The aggregate report evaluates the six acceptance criteria:

1. the same electronic solution;
2. unchanged forces and stress;
3. no increase in failure rate;
4. AFM-II preserved;
5. fewer SCF iterations in at least some cases;
6. end-to-end wall time saved.

It evaluates them **only once every case has finished**; before that they read
`not evaluated (benchmark incomplete)`. Wall time on shared clusters is noisy,
so repeat pairs (prepare several bench roots) before quoting a number.

The pilot ([`examples/density-init/pilot.yaml`](../examples/density-init/pilot.yaml))
has five slots:

- two nonmagnetic perovskite/interface structures;
- one metallic or narrow-gap interface;
- two NiO structures with explicit signed AFM-II `MAGMOM`, declared
  `expect_magnetic_order: afm-ii`. `prepare` refuses those two unless the
  INCAR moments really are mixed-sign.

An example report generated from synthetic fixtures is in
[`examples/density-init/example_bench_report.md`](../examples/density-init/example_bench_report.md).

## Limitations

- Collinear only; non-collinear and SOC runs are refused.
- Only POTCARs whose projector sets match the MP training set; see
  [POTCAR compatibility](#potcar-compatibility).
- The model's coverage of elements, chemistries, and large interface or slab
  cells (vacuum, charged defects, adsorbates) is whatever the upstream training
  data covers. Treat interfaces as out-of-distribution until the benchmark
  says otherwise.
- `ICHARG = 1` in an MD INCAR affects only the first ionic step.
- The initializer does not reduce the number of ionic steps and does not
  change any converged observable. If a benchmark case shows a different
  converged state, treat it as a multiple-minima problem (magnetic or +U
  occupation branch) and investigate it before using either label.
