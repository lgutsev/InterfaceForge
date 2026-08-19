# InterfaceForge

InterfaceForge is an early-stage research codebase for building reproducible
heterogeneous-interface potential workflows. Its intended scope is to turn VASP
and VASP-MLFF trajectories into traceable datasets, generate MACE and DeePMD
training jobs, select new DFT labels, and compare interface properties rather
than stopping at a global force RMSE. Most of that end-to-end workflow is still
an implementation target, not an experimentally demonstrated capability.

The optional AI2-Kit integration now supports the example-driven TESLA path for
an existing MACE committee: MACE/OpenMM exploration → committee force
deviation → VASP labeling. oh-my-batch generates the exploration/labeling job
matrix and provides Slurm retry/recovery. The older config-driven
DeepMD/LAMMPS/VASP adapter remains available as `workflow: cll_deepmd`. Both
paths are dry-run by default and keep imported labels outside the canonical
dataset until review. Neither has completed a human-tested real campaign. See
[the AI2-Kit guide](docs/ai2kit.md).

An optional InterMat adapter is implemented to generate commensurate crystalline
film/substrate registries while leaving all calculators and campaign mutation
under explicit InterfaceForge control. It has not been human-tested with a real
InterMat workflow. See [the InterMat guide](docs/intermat.md).

The project distills the reusable ideas in the original campaign templates into
a portable Python CLI. It does **not** contain structures, trajectories,
checkpoints, containers, or licensed VASP pseudopotentials. Personal,
cluster-specific launcher backups are isolated under `launch_scripts/`; they are
not used as package-generated templates.

## Verification status

The distinction below is important: **automated tests are not evidence that an
external scientific engine or a complete HPC workflow has run successfully.**
As of August 2026, the only parts exercised by a human on real scientific data
are:

- the VASP-MLFF preparation, audit, restart/recovery, and related plotting path;
- the VASP trajectory data generators for MACE extxyz and DeePMD NPY datasets.

Everything else is currently code-only: it may have unit tests, mocked command
tests, schema checks, or shell-generation checks, but it has not been verified
by a human in an end-to-end run with the named external engines. In particular,
the repository does **not** yet establish that generated MACE, DeePMD, Allegro,
LAMMPS, AI2-Kit, InterMat, or MACE-ROI workflows train, deploy, restart, or
produce scientifically valid results.

See [Verification and maturity](docs/verification.md) for the evidence standard,
known gaps, and the minimum checks needed to promote a feature's status.

## Implemented scope

| Layer | Implemented functionality | Verification note |
|---|---|---|
| VASP preparation | static, relaxation, DFT-MD, DOS and line-band setup; geometry conversion, supercells, slabs, duplicate checks and Selective Dynamics | **Partially human-tested.** The VASP-MLFF workflow is used on real runs; do not infer that every generic VASP preset or geometry subcommand has been exercised. |
| VASP-MLFF | train → refit → stability scaffolding, continuation/capacity recovery, mode-aware audits | **Human-tested on real campaigns.** This is the most mature part of the repository, but each new VASP version, cluster profile, and recovery condition still requires inspection. |
| Dataset | streaming OUTCAR collection, leakage-resistant splits, synchronized extxyz and DeePMD NPY layouts | **Human-tested for data generation.** Real VASP data have been exported for MACE and DeePMD. This does not validate downstream training or model quality. |
| MACE | two-stage energy/force job generation; optional interface-local force weighting and thermodynamic-cycle loss | **Code-only beyond dataset generation.** Generated training, restart, committee, evaluation, and MACE-ROI paths have not been verified end to end. |
| MLIP archiving | immutable collection of completed MACE committees into a checksummed directory and ZIP; final models only by default, with optional training data written to a separate ZIP | **Automated-test verified.** Collection, duplicate/missing-member rejection, archive safety, checksums, and verification are tested; long-term storage and restore have not yet been human-tested on a real committee archive. |
| DeePMD | DPA-1, DPA-2, DPA-3 and experimental DPA-4 job generation; TensorFlow/PyTorch backends; preflight → smoke → full → evaluation | **Code-only beyond dataset generation.** No generated training/deployment chain is currently claimed as human-tested. |
| Allegro | training/job-generation and LAMMPS-oriented adapter code | **Code-only.** No human-tested training or LAMMPS deployment. |
| Active learning | thermodynamic exploration matrix and uncertainty-plus-diversity labeling queue; AI2-Kit TESLA MACE/OpenMM/VASP with oh-my-batch; legacy config-driven DeepMD/LAMMPS/VASP | **Code-only.** No completed external-engine loop. |
| Validation | parity metrics, work of adhesion with uncertainty propagation, rigid-separation curves | **Code-only.** Numerical routines may be unit-tested, but no scientific validation campaign has established model accuracy. |
| Crystalline interface generation | optional InterMat surface matching, separation/registry scans, deduplicated POSCAR export | **Code-only.** Generated structures have not been human-reviewed in a real InterMat campaign. |
| Provenance | manifests, hashes, append-only events, JSON/CSV/Markdown audits and a self-contained HTML report | **Mostly code-only.** VASP-MLFF audit outputs are the exception; the broader campaign provenance chain has not been exercised end to end. |

## Install

```bash
git clone https://github.com/lgutsev/InterfaceForge.git
cd InterfaceForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

For configuration, audits and validation only, `pip install -e .` is enough.
ASE-backed geometry and OUTCAR collection require `.[vasp]`.
Experimental MACE-ROI training requires `pip install -e ".[mace-roi]"`.
InterMat geometry generation requires `pip install -e ".[intermat]"`.
AI2-Kit/oh-my-batch controller support requires `pip install -e ".[ai2kit]"`;
the GPU environment used for exploration must additionally provide MACE,
OpenMM, and OpenMM-ML.

## Ten-minute start

```bash
iface init my-interface
cd my-interface
# Edit campaign.yaml and profiles/loni.yaml; add structures and VASP inputs.
iface plan
iface prepare
iface submit                     # dry-run: lists jobs
iface submit --system interface_ab_300k --stage train --execute
```

After reference trajectories exist:

```bash
iface audit runs/vasp
iface collect
iface train mace
iface train deepmd
```

Collect a completed four-member MACE committee into a compact, immutable bundle
before deployment or active learning:

```bash
iface committee collect mace_committee stored_models/tin_sin_mace_v1.zip \
  --expected-members 4
iface committee verify stored_models/tin_sin_mace_v1
```

The collector creates both an extracted bundle and ZIP archive, copies only
final models, rejects missing or duplicate members, and writes checksums plus
source provenance. See
[the MACE committee guide](docs/mace-committee.md).

For interface-local and thermodynamic-cycle-aware MACE training, configure
`models.mace.roi`, then prepare the immutable derived data before generating
jobs:

```bash
iface mace-roi prepare
iface train mace
```

The generated DeePMD jobs have an intentional order:

```bash
sbatch models/deepmd/run_preflight.slurm
sbatch models/deepmd/run_smoke.slurm
# Inspect every smoke task before continuing.
sbatch models/deepmd/run_ensemble.slurm
sbatch models/deepmd/run_evaluate.slurm
```

DPA-4 stays labeled experimental. A trained checkpoint is not considered
deployable until freeze/export and the target LAMMPS path are independently
verified.

## Scientific invariants

- Reference forces are read with constraints unapplied. Frozen atoms remain
  represented in `REF_forces`; mobility is stored separately as `move_mask`.
- MACE-ROI metadata is written to a hashed derived dataset. Canonical labels
  remain unchanged, and thermodynamic cycles cannot cross data splits.
- The default split assigns whole trajectories, so nearby MD frames do not leak
  across train/validation/test. A guarded contiguous-block strategy is also
  available.
- Generated INCAR presets do not guess ENCUT, k-point density, spin, dispersion,
  convergence thresholds or chemistry-specific settings.
- New VASP-MLFF campaign templates opt into the documented `accurate` profile
  (`ML_IALGO_LINREG=1`, `ML_SION1=0.3`, `ML_MRB2=12` during training, followed
  by SVD refitting); existing campaigns remain unchanged.
- POTCAR is assembled only from an explicit licensed local tree and is excluded
  from portable archives.
- Submission is a dry-run unless `--execute` is present. Mutating recovery
  operations archive the current run first.

## Useful VASP commands

```bash
iface vasp incar md INCAR --temperature 600 --nsw 5000 --potim 1
iface vasp geom convert structure.xyz POSCAR --cell-from trusted.vasp
iface vasp geom slab bulk.vasp slab.vasp --miller 1 1 1 --layers 8
iface vasp geom freeze slab.vasp POSCAR --axis z --upper 5 --region inside
iface vasp restart run/ --clean-electronic
iface vasp ml-recover continue run/ --temperature 450 --nsw 3000
iface vasp submit run/ --ml-continue --temperature 450 --nsw 3000
iface vasp ml-recover expand run/ --ml-mb 12000
iface vasp submit run/  # generates a missing POTCAR from the supplied dictionary
iface vasp submit run/ --ml-capacity-recovery  # bounded-memory recovery via runvasp.sh
iface vasp submit run/ --ml-capacity-recovery --increase-eps-low  # optional 10x sparsification
iface audit runs/vasp --readiness-profile perovskite
iface vasp beef-plot runs/vasp --individual
iface vasp band scf/ bands/ --kpoints KPOINTS.line
iface vasp workfunction LOCPOT OUTCAR --plot-output workfunction.png
iface vasp adhesion prepare interface_run --method mlff --distances 0.5 1 2 3 4 6 8
iface vasp pack campaign_outputs.zip --root runs/vasp
iface vasp archive-models  # current folder; automatic timestamped ZIP name
iface vasp archive-models stored_models.zip --root successful_runs
iface vasp archive-models --exclude-folders old_300 test_run rejected_model
```

`iface vasp archive-models` packages only run directories containing a
nonempty `ML_AB`. It preserves `ML_AB` together with available MLFF model
state, inputs, compact trajectory outputs, launchers, and logs, and writes a
SHA-256 manifest inside the ZIP for long-term integrity checks. `POTCAR` is
always excluded. Add `--include-large` to retain `OUTCAR`, `vasprun.xml`,
`XDATCAR`, and `LOCPOT`. Model presence is a discovery rule, not a scientific
validation claim: point `--root` at runs you have already accepted.
When the output argument is omitted, the archive is written to the current
directory as `MLFF_Models_<folder>_<UTC timestamp>.zip`.
Directory trees are pruned when any folder name contains `backup`
(case-insensitive) or starts with `X`, preventing old backups and
`X_OutPack...` outputs from being archived again. `CHG`, `CHGCAR`, and
`WAVECAR` are never retained. The command reports compressed and uncompressed
sizes plus the ten largest stored files.
By default, only the current directory and its immediate child folders are
checked; daughter folders below those children are not scanned. Supply exact
names with `--exclude-folders NAME [NAME ...]` to prune additional folders at
the first scanned level. Use `--recursive` only when nested run discovery is
intentional.

`recover expand` requires a recognized capacity failure in OUTCAR unless
`--force-expand` is explicitly supplied.

### Perovskite MLFF readiness profile

Fluxional halide perovskites can retain a finite learning-event rate because
VASP's adaptive `ML_CTIFOR` follows the recent Bayesian-error distribution.
Requiring the learning rate or BEEF to approach zero can therefore prolong
on-the-fly sampling without improving the true force error. Make the intended
alternative explicit with the named **`perovskite`** option:

```bash
iface audit . --readiness-profile perovskite
iface status . --readiness-profile perovskite
```

The profile examines the most recent 250 records. It reports a perovskite
sampling plateau after at least 200 usable BEEF records when the recent BEEF
95th percentile is at most 0.03 eV/A, the means of the two half-windows differ
by at most 0.002 eV/A, the recent window has no critical events, and its
learning-event rate is at most 20%. A clean local-reference capacity stop is
then reported as a sampling checkpoint instead of an automatic instruction to
continue to the original `NSW` target.

The audit also reports the latest VASP `ERR` force RMSE prominently in
`audit_summary.csv`, `audit.xlsx`, and `audit.md`; energy, force, and stress
training RMSE values are retained in the full JSON/CSV/XLSX audit. These are
fit errors over the accumulated training structures, not held-out test RMSE,
so they inform refitting but do not by themselves certify the potential.

This option changes audit guidance only; it does not cancel a running Slurm
job or certify the final potential. At a reported checkpoint, preserve
`ML_ABN`, perform `ML_MODE=SELECT` when reselection is needed, perform the SVD
`ML_MODE=REFIT`, and validate prediction MD against held-out DFT forces and
application-specific structural observables. The default `general` profile is
unchanged.

### VASP-MLFF Bayesian-error campaign plots

Run the plotting command from the campaign root: the directory containing the
individual VASP run folders. InterfaceForge searches recursively for runs with
both `INCAR` and `ML_LOGFILE`:

```bash
cd /path/to/campaign
iface vasp beef-plot .
```

By default this writes `ML_BayesianErrorPlot_campaign.png` (one panel per
run) and `ML_BayesianErrorPlot_campaign.csv` to the campaign root. Add
`--individual` to also create separate per-run PNG files:

```bash
iface vasp beef-plot . --individual
```

MLFF recovery archives the preceding `ML_LOGFILE` under
`.interfaceforge/archive/`. If a newly prepared continuation has not yet
written new BEEF records, include those archived trajectory segments:

```bash
iface vasp beef-plot . --include-archives --individual
```

Use `-o`, `--data-output`, and `--dpi` to customize the outputs:

```bash
iface vasp beef-plot . \
    --include-archives \
    -o ML_Bayesian_campaign.png \
    --data-output ML_Bayesian_campaign.csv \
    --dpi 200
```

The plot reads BEEF/BEFF, ML_CTIFOR, and STATUS records from each
`ML_LOGFILE`, and uses `POTIM` from the corresponding `INCAR` to convert
steps to femtoseconds. Plotting requires Matplotlib, installed with
`pip install -e ".[report]"` or `pip install -e ".[all]"`. If the command
reports that no usable BEEF records were found, verify that the selected root
contains completed MLFF runs or retry with `--include-archives`.

## Modern DeePMD configuration

```yaml
models:
  deepmd:
    enabled: true
    profile: deepmd_gpu
    backend: pytorch          # tensorflow, pytorch, or pt_expt
    architectures: [dpa2, dpa3]
    committee: 4
    seeds: [11, 23, 37, 53]
    numb_steps: 500000
    batch_atoms: 1024
    max_concurrent: 2
    container_image: /path/to/deepmd-kit.sif
```

Use `backend: tensorflow` for established DPA-1/`se_atten` workflows. DPA-2,
DPA-3 and DPA-4 require a PyTorch backend. `pt_expt` records that the
experimental PyTorch implementation is intended, while the generated DeePMD CLI
commands correctly use the public `--pt` switch.

See [the campaign format](docs/campaign.md),
[the VASP guide](docs/vasp.md), and
[the DeePMD guide](docs/deepmd.md). The experimental method is documented in
[the MACE-ROI guide](docs/mace-roi.md), and deployable committee storage in
[the MACE committee guide](docs/mace-committee.md). A complete editable configuration is in
[examples/interface-campaign/campaign.yaml](examples/interface-campaign/campaign.yaml).

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

Passing the unit tests means that the Python-level contracts still behave as
encoded; it does not promote a feature to human-tested status. InterfaceForge
is an early research tool. Inspect generated inputs and complete a small,
independently checked smoke run for every engine/version/cluster combination
before spending a large allocation.
