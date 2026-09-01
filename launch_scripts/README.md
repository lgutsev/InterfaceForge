# Personal HPC launch-script backups

These launchers are preserved as working LONI examples for Lavrenty's campaigns.
They contain cluster-, account-, environment-, and project-specific settings and are
not portable InterfaceForge templates. Review the partition, allocation, module,
environment paths, wall time, job name, executable, and resource counts before reuse.

## Files

- `runvasp.sh`: two-node, 128-rank VASP Gamma launcher for `workq`.
- `runvasp_bigmem.sh`: two-node, 128-rank VASP Gamma launcher for `bigmem`.
- `run_mace_gpu2_nomask_v2.sh`: original two-GPU TiN/SiN MACE training launcher.
- `mace_train_committee.sh`: isolated, fixed-split launcher that writes each
  committee member to a seed-specific directory.
  It uses conservative batches (`8` training, `4` validation), expandable CUDA
  segments, and records five-second GPU telemetry in each seed directory as
  `gpu_usage_<jobid>.log`. Canonical InterfaceForge labels default to
  `REF_energy` and `REF_forces`; legacy datasets can override them through
  `MACE_ENERGY_KEY` and `MACE_FORCES_KEY`.
- `deepmd_32_gpu_preflight.sbatch`: non-mutating `gpu2` check for LONI's
  `deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu` module, DeePMD 3.2 PyTorch commands,
  CUDA visibility, and optional LAMMPS DeePMD support. It does not request
  memory manually.
- `deepmd_lammps_30_gpu_audit.sbatch`: audits LONI's existing
  `lammps/29Aug2024-r8.0-deepmd3.0.0-gpu` module without being fooled by an
  unrelated user-level `lmp`. With a model-list file and one canonical DeePMD
  system, it performs a one-step inference test and verifies committee model
  deviation.
- `run_slab_alignment_single.sbatch`: one-core `single`-partition launcher for
  `iface vasp slab-align`; it auto-audits LOCPOT flatness, writes review flags
  and non-destructive `INCAR.dipole_fix` proposals only for non-flat cases,
  creates a root relaunch-review queue, and runs `sumo-dosplot` in each matched
  folder. It never edits `INCAR` or submits VASP.
- `restart_daughter_jobs.sh`: one-level campaign helper for immediate VASP daughter
  directories. It can run either `Restart <daughter>` or `TotalRestart <daughter>`,
  copies root `INCAR`, `KPOINTS`, and `runvasp.sh`, verifies the expected restart
  postcondition, preserves `CONTCAR` independently before `TotalRestart`, and writes
  a timestamped TSV launch audit.
- `restart_leaf_jobs.sh`: campaign helper that copies the root `INCAR` and
  `runvasp.sh` into every deepest/leaf calculation directory, requires a non-empty
  `CONTCAR`, runs `Restart <leaf-name>` from the leaf's parent directory, captures
  the Slurm job ID, checks `squeue`/`sacct` when available, and writes a timestamped
  TSV launch audit.
- `collect_leaf_mace.py`: collects VASP `OUTCAR` trajectories only from deepest
  directories into MACE `train.extxyz`, `valid.extxyz`, and `test.extxyz` files.
- `collect_leaf_deepmd.py`: collects the same leaf trajectories into native DeePMD
  systems while physically retaining the source directory hierarchy.

Submit four independent committee members from the directory containing
`train.extxyz`, `valid.extxyz`, and `test.extxyz`:

```bash
for seed in 11 23 37 53; do
    sbatch --export=ALL,MACE_SEED="$seed" mace_train_committee.sh
done
```

Runs are stored under `mace_committee/seed_<seed>/`. The default model prefix is
`SiN_TiN_TiO_periodic_mace`; set `MACE_MODEL_PREFIX` to use another descriptive
prefix. For an older extxyz dataset whose labels are named `energy` and `forces`,
submit with:

```bash
sbatch \
    --export=ALL,MACE_SEED=11,MACE_ENERGY_KEY=energy,MACE_FORCES_KEY=forces \
    mace_train_committee.sh
```

The preflight reads reference labels directly from `atoms.info` and
`atoms.arrays`. It does not require an ASE calculator to be attached to frames.

### Fine-tune a MACE committee from a foundation model

`mace_finetune_committee.sh` is the fine-tuning counterpart of
`mace_train_committee.sh`: same fixed `train/valid/test.extxyz` split, same
seeds, but every member starts from a foundation model instead of random
initialization. Output goes to `mace_finetune_committee/seed_<seed>/` so it
never collides with the from-scratch committee.

First download a foundation model to a compute-node-readable path (a bare
`small|medium|large` name only works with outbound network access):

```bash
# MACE-MPA-0 medium (MPtrj + sAlex, PBE(+U)) — recommended, needs mace >= 0.3.10
wget -P /project/lgutsev/foundational_models/mace/ \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model
# older mace: MACE-MP-0 medium instead
#   .../mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
```

Check `python -c "import mace; print(mace.__version__)"` in `mace_env` first.

Then submit four seeds **from the same directory that holds the committee's
`train.extxyz` / `valid.extxyz` / `test.extxyz`** (the parent of the existing
`mace_committee/`):

```bash
FM=/project/lgutsev/foundational_models/mace/mace-mpa-0-medium.model
for seed in 11 23 37 53; do
    sbatch --export=ALL,MACE_SEED="$seed",MACE_FOUNDATION_MODEL="$FM" \
        mace_finetune_committee.sh
done
```

Defaults: naive fine-tuning (`MACE_MULTIHEADS=False` — specialise to this
dataset, no replay head), `MACE_E0S=foundation` (reuse the foundation's atomic
energies; set `MACE_E0S=average` if the reference DFT is not MP-compatible),
`MACE_DEFAULT_DTYPE=float64`. The architecture (`r_max`, channels, `max_L`,
`correlation`, interactions) and float precision are inherited from / matched to
the foundation model — the MP/MPA/OMAT checkpoints are float64, and fine-tuning
at float32 fails with `both inputs should have same dtype`. Other knobs:
`MACE_MULTIHEADS=True` with `MACE_PT_TRAIN_FILE=<replay data>` for replay
fine-tuning, `MACE_MAX_EPOCHS`, `MACE_START_STAGE_TWO`, `MACE_MODEL_PREFIX`.

**Comparison confounds to state in the writeup:** the fine-tuned committee
differs from the from-scratch `mace_committee/` in (a) architecture — MACE-MPA-0
medium is `L=1`, `r_max=6`, Agnesi radial, vs the from-scratch `L=2`, `r_max=5`,
Bessel radial — and (b) precision — float64 here vs float32 there. Both are
inherent to fine-tuning a foundation model; neither the accuracy gain nor any
gap can be attributed purely to the fine-tuning itself.

To fold the fine-tuned committee into the matched comparison, point
`iface mlip-compare` at it:

```bash
iface mlip-compare prepare \
  --mace-models-root "<committee dir>/mace_finetune_committee" \
  --deepmd-arch dpa2 \
  --output-root "$CAMP/audit/mlip_compare_mace_ft" --force
```

`_discover_models` globs `seed_<seed>/mace_model/*_stagetwo.model`, which
matches the `*_ft_seed<seed>_stagetwo.model` files this script produces. The
MACE rows in that run are the fine-tuned committee's numbers; compare them
against the MACE rows in the from-scratch `audit/mlip_compare/`.

## Restart immediate daughter calculations

Use `restart_daughter_jobs.sh` when the campaign root directly contains the VASP
calculation directories you want to start. It only selects immediate daughter
folders containing `POSCAR` or `CONTCAR`; directories beginning with `X` and names
containing `backup` are ignored.

Preview a normal restart campaign first:

```bash
/path/to/InterfaceForge/launch_scripts/restart_daughter_jobs.sh restart --dry-run
```

Then launch it:

```bash
/path/to/InterfaceForge/launch_scripts/restart_daughter_jobs.sh restart
```

For each selected daughter, the helper copies the root `INCAR`, `KPOINTS`, and
`runvasp.sh`, requires a non-empty `CONTCAR`, executes `Restart <daughter-name>`
from the campaign root, and verifies afterward that the daughter `POSCAR` matches
the pre-launch `CONTCAR`.

For a clean restart that intentionally discards VASP electronic/continuation state:

```bash
/path/to/InterfaceForge/launch_scripts/restart_daughter_jobs.sh total-restart --dry-run
/path/to/InterfaceForge/launch_scripts/restart_daughter_jobs.sh total-restart
```

Before invoking `TotalRestart <daughter-name>`, InterfaceForge independently saves
any non-empty `CONTCAR` as
`CONTCAR.pre_TotalRestart_<timestamp>_<pid>`. After the helper returns, it checks
that `WAVECAR`, `CHGCAR`, and `CONTCAR` are gone. Thus the wrapper does not merely
assume the personal `TotalRestart` helper behaved as intended.

Both modes capture a Slurm `Submitted batch job <id>` when available, query
`squeue`/`sacct`, and write `daughter_restart_audit_YYYYmmdd_HHMMSS.tsv`. Use
`--no-copy` if the daughter inputs should be left untouched. `Restart` and
`TotalRestart` may be executables, shell functions, or aliases loaded by
`~/.bashrc`.

## Restart all leaf calculations

Copy `restart_leaf_jobs.sh` (or call it by full path) and run it from the root of a
calculation tree containing the replacement `INCAR` and `runvasp.sh`:

```bash
/path/to/InterfaceForge/launch_scripts/restart_leaf_jobs.sh
```

For each leaf directory, the helper skips cases without a usable `CONTCAR`, copies
`INCAR` and `runvasp.sh`, makes the launcher executable, steps to the leaf's parent,
and executes:

```bash
Restart <leaf-directory-name>
```

It records `Restart`'s exit status, any `Submitted batch job <id>` returned by
Slurm, and the immediately visible scheduler state in
`restart_audit_YYYYmmdd_HHMMSS.tsv`. If `Restart` is defined only as a personal
shell alias/function, the helper falls back to an interactive Bash so `~/.bashrc`
can provide it.

To inspect which directories would be touched without copying files or launching
jobs:

```bash
/path/to/InterfaceForge/launch_scripts/restart_leaf_jobs.sh --dry-run
```

## Collect leaf calculations for MACE / DeePMD

The leaf collectors deliberately do **not** treat every terminal directory as an
independent statistical sample. The complete source-root-relative parent lineage is
the grouping key. All sibling leaf runs from the same parent branch are therefore
assigned together to train, validation, or test and cannot leak across splits.

`--heritage-depth` controls how many immediate ancestor names are repeated as
human-readable context metadata. The default is two. The full parent lineage is
always retained as the actual split-group identity, so repeated folder names in
unrelated branches do not accidentally merge.

Preview the discovered leaves and their split assignment before writing anything:

```bash
/path/to/InterfaceForge/launch_scripts/collect_leaf_mace.py \
  --root . --output mace_leaf_dataset --dry-run

/path/to/InterfaceForge/launch_scripts/collect_leaf_deepmd.py \
  --root . --output deepmd_leaf_dataset --dry-run
```

Create the datasets:

```bash
/path/to/InterfaceForge/launch_scripts/collect_leaf_mace.py \
  --root . --output mace_leaf_dataset

/path/to/InterfaceForge/launch_scripts/collect_leaf_deepmd.py \
  --root . --output deepmd_leaf_dataset
```

Useful options shared by both collectors:

```text
--heritage-depth 2
--ratios 0.8 0.1 0.1
--seed 20260730
--stride 1
--include-virial
--force
```

For MACE, every emitted frame records `IF_leaf`, `IF_heritage`,
`IF_heritage_parent`, and `IF_heritage_context` in the extxyz metadata.

For DeePMD, context is additionally preserved in the filesystem itself. A source
leaf such as:

```text
material/termination/450K/replica_03/OUTCAR
```

becomes, for example:

```text
deepmd_leaf_dataset/train/material/termination/450K/replica_03/
├── type.raw
├── type_map.raw
├── heritage.json
├── frame_map.csv
└── set.000/
    ├── coord.npy
    ├── box.npy
    ├── energy.npy
    └── force.npy
```

The collectors also write `leaf_manifest.csv` and `leaf_manifest.json`, recording
source leaf, ancestry/group identity, assigned split, frame count, output path, and
any failed leaf conversions. Backup-containing paths and directories beginning with
`X` are ignored.

## Combine several source trees into synchronized datasets

Use `iface-mapped-collect` when the VASP leaves are distributed across separate
temperature, termination, interface, and bulk roots. A YAML file maps each physical
source tree into one clean logical staging hierarchy. Only selected direct VASP files
are hard-linked into the staged leaves, so large `OUTCAR` files are not duplicated and
restart/archive daughter directories cannot make a valid calculation non-terminal.

```bash
iface-mapped-collect examples/mapped-leaf-campaign/template.yaml
iface-mapped-collect examples/mapped-leaf-campaign/template.yaml --execute --collect
```

The first command is a non-mutating dry run. `--execute --collect` writes both the
MACE extxyz and DeePMD NPY representations using the same heritage grouping, ratios,
seed, stride, and virial policy. It then cross-checks their leaf membership, split
assignment, frame counts, and conversion status, writing JSON, CSV, Markdown, and a
self-contained SVG distribution dashboard.

The checked-in periodic SiN/TiN/TiO custom job supplies the exact LA Tech mapping:

```bash
launch_scripts/prepare_periodic_nitride_mlips.sh
launch_scripts/prepare_periodic_nitride_mlips.sh --execute --collect
```

See [`examples/mapped-leaf-campaign/README.md`](../examples/mapped-leaf-campaign/README.md)
for output locations, path overrides, and instructions for copying the generic template.
