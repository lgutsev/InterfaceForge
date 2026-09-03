# InterfaceForge

InterfaceForge is a research workflow toolkit for reproducible atomistic
interface calculations. It now covers the path from audited VASP slab inputs
and staged AIMD, through synchronized MACE/DeePMD datasets and committee
training jobs, to matched-frame model comparison and interface-specific
validation. It also provides declarative builders for reactive magnetic
surfaces and optional adapters for active learning and crystalline interface
generation.

The package is deliberately review-first. Preparation and submission are
separate operations, submission is dry-run by default, generated scientific
inputs are audited and hashed, and questionable outputs are flagged rather
than silently repaired or relaunched. Implementation breadth has grown quickly;
the [verification status](#verification-status) still distinguishes real
human-tested paths from automated-test-only and code-only features.

The project distills the reusable ideas in the original campaign templates into
a portable Python CLI. It does **not** contain structures, trajectories,
checkpoints, containers, or licensed VASP pseudopotentials. Personal,
cluster-specific launcher backups are isolated under `launch_scripts/`; they are
not used as package-generated templates.

## What changed in the latest development cycle

- **Audited VASP slab pipeline:** notebook-generated OPT batches can be
  prepared, preflighted and launched, then promoted through `OPT -> Step1 ->
  Step2_<T>K`. Step1 supports adaptive WAVECAR reuse or an explicit fresh
  electronic start; Step1 and Step2 expose separate `academic` and `training`
  AIMD protocols.
- **Vacuum and band-alignment audits:** slab-to-image vacuum can be checked over
  complete directory trees and safely extended only with `--execute`.
  `iface vasp slab-align` performs side-specific LOCPOT plateau fitting,
  vacuum-aligned VBM/CBM comparison, automatic flatness triage, marker files,
  and reviewable `INCAR.dipole_fix` proposals without submitting VASP.
- **Modern DeePMD campaigns:** DPA-1/2/3 and experimental DPA-4 are joined by
  `dpa2_ft`, which fine-tunes a DPA-2 foundation checkpoint while preserving a
  separate from-scratch DPA-2 baseline. Generated jobs follow a guarded
  preflight -> smoke -> ensemble -> evaluation sequence.
- **Matched committee benchmarking:** `iface mlip-compare` proves that MACE and
  DeePMD evaluate the same canonical frames, then reports micro, macro,
  chemistry-, temperature- and oxidation-resolved errors, uncertainty
  calibration, publication figures, and per-system committee heatmaps. DPA-2,
  DPA-2 fine-tune, DPA-3 and DPA-4 comparison trees are supported.
- **Live MLIP progress:** `iface mlip-progress` gives one read-only status table
  for running MACE and DeePMD committees, completed/frozen models, per-system
  evaluations and comparison jobs by parsing the files they produce in place.
- **Reactive NiO surface campaigns:** `iface surface` now handles exposed-site
  analysis, AFM-compatible cell selection, water-balanced hydroxylation and
  proton-transfer states, direct/H-bond phosphonate docking, VASP export, and
  post-relaxation chemistry/spin audits. The bundled NiO(110) benchmark uses
  the tractable 200-atom `[[4,0],[0,5]]` compromise cell.
- **HPC guardrails:** LONI profiles include the current DeePMD 3.2 GPU module,
  explicit legacy VASP 6.5.1 options, runtime preflights, restartable arrays,
  and independent DeePMD/LAMMPS compatibility checks.

## Verification status

The distinction below is important: **automated tests are not evidence that an
external scientific engine or a complete HPC workflow has run successfully.**
As of September 2026, the parts recorded as exercised by a human on real
scientific data are:

- the VASP-MLFF preparation, audit, restart/recovery, and related plotting path;
- the VASP trajectory data generators for MACE extxyz and DeePMD NPY datasets;
- standard four-member MACE committee training and held-out evaluation on the
  periodic SiN/TiN/TiO interface campaign;
- DeePMD DPA-2 committee training, checkpoint continuation, freezing and
  `dp test` evaluation on the same real campaign;
- live MACE/DeePMD monitoring with `iface mlip-progress` while those committee
  jobs were running.

Those successful runs establish practical usability for the tested chemistry,
architecture, software version and LONI environment; they are not universal
scientific validation. DPA-3, DPA-4, DPA-2 fine-tuning, foundation-model MACE
fine-tuning, MACE-ROI, Allegro, MLIP deployment through LAMMPS, AI2-Kit and
InterMat retain their narrower automated-test-only or code-only status.

See [Verification and maturity](docs/verification.md) for the evidence standard,
known gaps, and the minimum checks needed to promote a feature's status.

## Implemented scope

| Layer | Implemented functionality | Verification note |
|---|---|---|
| VASP preparation | static, relaxation, DFT-MD, DOS and line-band setup; audited OPT batches; `OPT -> Step1 -> Step2` AIMD staging; geometry conversion, supercells, slabs, slab-vacuum checks and Selective Dynamics | **Partially human-tested.** The VASP-MLFF workflow is used on real runs; newer generic VASP, staged AIMD and geometry commands retain their narrower status in `docs/verification.md`. |
| Slab electrostatics | per-face vacuum-gap audits and guarded cell extension; side-specific LOCPOT plateau fitting; vacuum-aligned VBM/CBM shifts; automatic flatness triage and proposed dipole-correction INCARs | **Automated-test verified.** The analyzer never overwrites INCAR or relaunches VASP. Scientific interpretation still requires visual inspection of the vacuum profile, electronic convergence and projected states. |
| VASP-MLFF | train → refit → stability scaffolding, continuation/capacity recovery, mode-aware audits | **Human-tested on real campaigns.** This is the most mature part of the repository, but each new VASP version, cluster profile, and recovery condition still requires inspection. |
| Dataset | streaming OUTCAR collection, leakage-resistant splits, synchronized extxyz and DeePMD NPY layouts | **Human-tested for data generation.** Real VASP data have been exported for MACE and DeePMD. This does not validate downstream training or model quality. |
| MACE | two-stage energy/force job generation; committee training/evaluation; optional interface-local force weighting and thermodynamic-cycle loss | **Human-tested for standard committee training and evaluation.** A real four-seed committee completed on the periodic SiN/TiN/TiO campaign and produced held-out energy/force metrics. Foundation-model fine-tuning, MACE-ROI, deployment and transferability remain unverified separately. |
| MLIP archiving | immutable collection of completed MACE committees into a checksummed directory and ZIP; final models only by default, with optional training data written to a separate ZIP | **Automated-test verified.** Collection, duplicate/missing-member rejection, archive safety, checksums, and verification are tested; long-term storage and restore have not yet been human-tested on a real committee archive. |
| DeePMD | DPA-1, DPA-2, DPA-3, experimental DPA-4 and DPA-2 foundation-checkpoint fine-tuning; TensorFlow/PyTorch backends; preflight → smoke → full → evaluation | **Human-tested for PyTorch DPA-2 committee training and evaluation.** Real LONI runs exercised checkpoint continuation, model freezing and per-system `dp test` reporting. Other architectures, fine-tuning and LAMMPS deployment remain separate unverified gates. |
| Allegro | training/job-generation and LAMMPS-oriented adapter code | **Code-only.** No human-tested training or LAMMPS deployment. |
| Active learning | thermodynamic exploration matrix and uncertainty-plus-diversity labeling queue; AI2-Kit TESLA MACE/OpenMM/VASP with oh-my-batch; legacy config-driven DeepMD/LAMMPS/VASP | **Code-only.** No completed external-engine loop. |
| Validation | parity metrics, work of adhesion with uncertainty propagation, rigid-separation curves | **Code-only.** Numerical routines may be unit-tested, but no scientific validation campaign has established model accuracy. |
| MLIP comparison | matched-frame MACE vs DPA-2/DPA-2-FT/DPA-3/DPA-4 audits on identical canonical test configurations; committee micro/macro/grouped metrics, spread calibration, publication summaries and heatmaps | **Automated-test verified.** A repository-owned synthetic `prepare → status → finalize` workflow and failure guards run in CI; no real cross-backend committee has yet completed the full audit. The present legacy MACE launcher and comparison evaluator both use float32. |
| MLIP progress | read-only rollup of live MACE epochs/RMSE, DeePMD steps/RMSE/checkpoint/freeze state, per-system evaluation completion and comparison status | **Human-tested on running committees.** It reads generated filesystem artifacts safely while jobs run; it does not query or replace Slurm accounting. |
| Crystalline interface generation | optional InterMat surface matching, separation/registry scans, deduplicated POSCAR export | **Code-only.** Generated structures have not been human-reviewed in a real InterMat campaign. |
| Reactive magnetic surfaces | exposed-site analysis, AFM-compatible cell optimization, stoichiometric hydroxylation/proton-transfer states, direct/H-bond phosphonate docking, runnable VASP export, and post-relaxation chemistry/spin audits | **Automated-test verified on the bundled 200-atom NiO(110) system.** The generator recovers the `[[4,0],[0,5]]` compromise cell and builds the full reference campaign; no generated VASP relaxation has yet been scientifically validated. |
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
Slab-alignment plots and SUMO-oriented analysis require `.[slab-align]`.
Repository development and the complete test suite require `.[dev]`.
Experimental MACE-ROI training requires `pip install -e ".[mace-roi]"`.
InterMat geometry generation requires `pip install -e ".[intermat]"`.
AI2-Kit/oh-my-batch controller support requires `pip install -e ".[ai2kit]"`;
the GPU environment used for exploration must additionally provide MACE,
OpenMM, and OpenMM-ML.

## Core campaign quick start

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
iface validate stratified predictions.csv results.csv  # per-class errors, see docs/stratified-validation.md
iface validate separation-energy audit/sep "LEAF=sharifi_111_Nterm" --mace-model M1 --deepmd-model D1 -c campaign.yaml  # gamma_sep DFT vs MLIP, see docs/separation-energy.md
```

## Task-oriented entry points

### Slab optimization to AIMD training data

The current VASP path keeps generation, audit and submission as distinct
decisions:

```bash
# Audit a generated optimization batch, then submit only after review.
iface vasp opt-prepare OPT --manifest OPT/manifest_batch.csv --dry-run
iface vasp opt-prepare OPT --manifest OPT/manifest_batch.csv
iface vasp opt-launch OPT
iface vasp opt-launch OPT --execute

# Promote completed optimizations into short MLIP-training trajectories.
iface vasp step1-prepare OPT --protocol training --dry-run
iface vasp step1-prepare OPT --protocol training
# Proton-rich surfaces (OH50): same 400-step budget, 0.5 fs, tighter SCF,
# Langevin friction, and a gentle warm-up ramp.
iface vasp step1-prepare OPT_OH50 --protocol training --fresh-start --conservative \
    --langevin --ramp-from 200
iface vasp step1-status Step1
# Dry-run first: identify numerical runaways and the clean rewind frame.
iface vasp step1-repair Step1
# Archive failed state, rewind, tighten the SCF, and prepare the remaining steps.
iface vasp step1-repair Step1 --execute --langevin
iface vasp step2-prepare Step1 --temperatures 300 450 600 --protocol training
iface vasp step2-launch Step2_300K Step2_450K Step2_600K
iface vasp step2-launch Step2_300K Step2_450K Step2_600K --execute
```

Use `--fresh-start` with `step1-prepare` when old or incompatible WAVECARs
must not be reused. Without it, each run adapts independently: a usable
WAVECAR is hard-linked or copied and restarted, while a missing WAVECAR falls
back to `ISTART=0` with inherited magnetic moments. POTCAR may be generated by
the launcher and is therefore optional at this stage. See the
[VASP guide](docs/vasp.md).

Use `--conservative` for proton-rich or reactive starting structures. It
preserves the selected protocol's `NSW` budget and AFM initialization while
hardening the run against the failure mode seen on dissociated-hydroxyl
surfaces — forces read off a sloshing SCF, not just an over-large timestep:

- `POTIM=0.5` fs and `ALGO=Normal` (override with `--algo All`/`Conjugate`,
  which VASP recommends for magnetic + DFT+U);
- a tighter electronic loop: `EDIFF=1E-5`, `NELM=120`, `NELMIN=6`;
- `--langevin` swaps `SMASS=-1` velocity rescaling for a Langevin thermostat
  (`MDALGO=3`, `LANGEVIN_GAMMA` per species, default 10 ps⁻¹) whose friction
  also damps an incipient runaway;
- `--ramp-from T0` starts `TEBEG` at `T0` and ramps to the target, softening
  the cold start.

The sampled physical duration is half that of a 1 fs run with the same `NSW`.
The promoted `POSCAR` has its trailing MD velocity block stripped (pass
`--keep-velocities` to keep it) so VASP draws fresh Maxwell-Boltzmann
velocities at `TEBEG`; `step2-prepare` does the same on the Step1 → Step2
hand-off, which matters when a Step2 temperature differs from the preheat.

`step1-status` also diagnoses energy/temperature runaways and repeated hits at
the electronic `NELM` ceiling. `step1-repair` never trusts a crash-time
`CONTCAR` or `WAVECAR`: it selects an earlier saved XDATCAR frame, keeps an
eight-step safety margin, and prepares only the remaining step budget with
`ISTART=0`, `ALGO=Normal` (or `--algo`), `POTIM=0.5` (or `--potim`), the same
tightened SCF, and optional `--langevin` / `--ramp-from`. The failed state is
archived, the accepted prefix is recorded in `step1_repair.json`, and runs
updated within the last six hours are protected from mutation. Repair
preparation never submits jobs.

### Slab vacuum and band-edge alignment

```bash
# Audit a complete tree; preview cell extension first, write only with --execute.
iface vasp geom vacuum slab_runs --min-vacuum 12
iface vasp geom vacuum slab_runs --extend 18
iface vasp geom vacuum slab_runs --extend 18 --execute

# Analyze existing LOCPOT/OUTCAR/vasprun.xml families on a compute node.
sbatch launch_scripts/run_slab_alignment_single.sbatch
```

The alignment audit writes per-run vacuum profiles, a flatness table, marker
files, and `relaunch_review_queue.txt`. Flagged folders receive a proposed
`INCAR.dipole_fix`; InterfaceForge does not apply it or resubmit the job. See
the [slab-alignment example](examples/vasp/slab-alignment/README.md).

### Compare trained MACE and DeePMD committees

```bash
# Safe to repeat while the training/evaluation jobs are running.
iface mlip-progress .

iface mlip-compare prepare --deepmd-arch dpa2 --force
sbatch audit/mlip_compare/run_mace_evaluate.slurm
iface mlip-compare status --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
iface mlip-compare finalize --deepmd-eval-root models/deepmd/evaluation/dpa2/job_<jobid>
```

Repeat with `--deepmd-arch dpa2_ft`, `dpa3`, or `dpa4` and a distinct
`--output-root` to compare architectures against the same MACE committee. The
workflow refuses mismatched frame identities before inference and rechecks the
DeePMD reference columns before finalization. See the
[MLIP-comparison guide](docs/mlip-comparison.md).

When reference trajectories live under several unrelated roots, use a mapped leaf
campaign to create synchronized MACE and DeePMD datasets plus a visual cross-audit:

```bash
iface-mapped-collect examples/mapped-leaf-campaign/template.yaml          # dry-run
iface-mapped-collect examples/mapped-leaf-campaign/template.yaml --execute --collect
```

The periodic SiN/TiN/TiO example is available as
`launch_scripts/prepare_periodic_nitride_mlips.sh`. See the
[mapped leaf-campaign guide](examples/mapped-leaf-campaign/README.md).

For reactive magnetic oxide surfaces, start from the packaged declarative
campaign instead of copying notebook cells:

```bash
iface surface init -o surface_campaign.yaml
iface surface plan surface_campaign.yaml
iface surface build surface_campaign.yaml
iface surface audit generated_surface_campaign -o surface_audit.csv
```

`iface surface cell-optimize` chooses an in-plane transformation under a real
adsorbate-image clearance, atom budget, aspect-ratio limit, and optional AFM
translation parity. The bundled NiO(110)/Me4PACz benchmark independently
recovers the 200-atom `[[4,0],[0,5]]` compromise. See the
[reactive-surface guide](docs/reactive-surfaces.md).

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

PyTorch evaluation uses the restartable `model.ckpt.pt` artifacts and writes
component-weighted `rmse_by_system.csv`, `rmse_overall.csv`, and
`rmse_audit.json` reports. Freeze/export and LAMMPS compatibility remain
separate deployment gates; see [the DeePMD guide](docs/deepmd.md).

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
  when `stages.vasp_mlff.enabled: true`; VASP-MLFF generation defaults to false
  so existing DFT-labelled campaigns cannot create MLFF jobs implicitly.
- Mapped VASP datasets require and cross-audit explicit `ENCUT`, `IVDW`, and
  `POTIM`, compare complete INCARs, fingerprint source inputs/OUTCARs, retain
  VASP/POTCAR/k-point identity, and balance sampled frames per leaf by default.
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
iface vasp geom vacuum slab_runs --min-vacuum 12
iface vasp geom vacuum slab_runs --extend 18 --execute
iface vasp restart run/ --clean-electronic
iface vasp opt-prepare generated --manifest generated/manifest_batch.csv --exclude-prefix OH0 --dry-run
iface vasp opt-prepare generated --manifest generated/manifest_batch.csv --exclude-prefix OH0 --launcher-template runvasp.sh
iface vasp opt-launch generated                 # full preflight; no submission
iface vasp opt-launch generated --execute       # submit the unchanged PASS-audited batch
iface vasp step1-prepare OPT --protocol training --fresh-start --conservative --langevin --ramp-from 200
iface vasp step1-status Step1
iface vasp step1-repair Step1                       # inspect only
iface vasp step1-repair Step1 --execute --langevin  # archive + prepare; never submits
iface vasp step2-prepare Step1 --protocol training --temperatures 300 450 600
iface vasp step2-launch Step2_300K Step2_450K Step2_600K --execute
iface vasp ml-recover continue run/ --temperature 450 --nsw 3000
iface vasp submit run/ --ml-continue --temperature 450 --nsw 3000
iface vasp ml-recover expand run/ --ml-mb 12000
iface vasp submit run/  # generates a missing POTCAR from the supplied dictionary
iface vasp submit run/ --ml-capacity-recovery  # bounded-memory recovery via runvasp.sh
iface vasp submit run/ --ml-capacity-recovery --increase-eps-low  # optional 10x sparsification
iface audit runs/vasp --readiness-profile perovskite
iface vasp beef-plot runs/vasp --individual
iface vasp band scf/ bands/ --kpoints KPOINTS.line
iface vasp slab-align slab_campaign --config slab_alignment.json --run-sumo
iface vasp slab-publish slab_campaign --config slab_publication.json --run-sumo
iface vasp workfunction LOCPOT OUTCAR --plot-output workfunction.png
iface vasp adhesion prepare interface_run --method mlff --distances 0.5 1 2 3 4 6 8
iface vasp adhesion audit interface_run_adhesion_mlff  # after VASP finishes
iface vasp adhesion audit interface_run_adhesion_mlff -c campaign.yaml --interface "interface/450K/Real/Ti_Term/*"  # + literature comparison
iface vasp adhesion summary "LEAF=run1_adhesion_mlff" "LEAF=run2_adhesion_mlff" -c campaign.yaml -o audit/adhesion_summary  # W_ad table + publication figure
iface reference show sharifi2026  # bundled Si3N4/TiN work-of-adhesion reference profile
iface reference activate sharifi2026 -c campaign.yaml --write  # splice it into validation.reference_profiles
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
    architectures: [dpa2, dpa2_ft, dpa3]
    committee: 4
    seeds: [11, 23, 37, 53]
    numb_steps: 500000
    batch_atoms: 1024
    max_concurrent: 2
    container_image: /path/to/deepmd-kit.sif
    finetune:
      pretrained: /path/to/dpa2_foundation_checkpoint.pt
      model_branch: RANDOM
```

Use `backend: tensorflow` for established DPA-1/`se_atten` workflows. DPA-2,
DPA-3 and DPA-4 require a PyTorch backend. `pt_expt` records that the
experimental PyTorch implementation is intended, while the generated DeePMD CLI
commands correctly use the public `--pt` switch. `dpa2_ft` requires the
`finetune` block and generates the checkpoint-driven `--finetune ...
--use-pretrain-script` path described in the DeePMD guide.

See [the campaign format](docs/campaign.md),
[the VASP guide](docs/vasp.md), and
[the DeePMD guide](docs/deepmd.md). The experimental method is documented in
[the MACE-ROI guide](docs/mace-roi.md), and deployable committee storage in
[the MACE committee guide](docs/mace-committee.md). Active-learning adapters
are covered by [the AI2-Kit guide](docs/ai2kit.md), crystalline matching by
[the InterMat guide](docs/intermat.md), and reactive magnetic slabs by
[the surface-campaign guide](docs/reactive-surfaces.md). A complete editable
configuration is in
[examples/interface-campaign/campaign.yaml](examples/interface-campaign/campaign.yaml).

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

Passing the unit tests means that the Python-level contracts still behave as
encoded; it does not promote a feature to human-tested status. InterfaceForge
is an early research tool. Inspect generated inputs and complete a small,
independently checked smoke run for every engine/version/cluster combination
before spending a large allocation.
