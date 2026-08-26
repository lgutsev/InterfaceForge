# Personal HPC launch-script backups

These launchers are preserved as working LONI examples for Lavrenty's campaigns.
They contain cluster-, account-, environment-, and project-specific settings and are
not portable InterfaceForge templates. Review the partition, allocation, module,
environment paths, wall time, job name, executable, and resource counts before reuse.

## Files

- `runvasp.sh`: two-node, 128-rank VASP Gamma launcher for `workq`.
- `runvasp_bigmem.sh`: two-node, 128-rank VASP Gamma launcher for `bigmem`.
- `run_mace_gpu2_nomask_v2.sh`: original two-GPU TiN/SiN MACE training launcher.
- `mace_train_committee.sh`: isolated additional-seed launcher that preserves the
  original model and writes each new committee member to a seed-specific directory.
  It uses conservative batches (`8` training, `4` validation), expandable CUDA
  segments, and records five-second GPU telemetry in each seed directory as
  `gpu_usage_<jobid>.log`.
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

Submit the three additional committee members from the directory containing
`train.extxyz`, `valid.extxyz`, and `test.extxyz`:

```bash
for seed in 211 307 419; do
    sbatch --export=ALL,MACE_SEED="$seed" mace_train_committee.sh
done
```

The original `mace_model/TiN_SiN_mace_stagetwo.model` is retained as the first
committee member. New runs are stored under `mace_committee/seed_<seed>/`.

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
