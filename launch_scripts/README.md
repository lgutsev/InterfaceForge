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
- `restart_leaf_jobs.sh`: campaign helper that copies the root `INCAR` and
  `runvasp.sh` into every deepest/leaf calculation directory, requires a non-empty
  `CONTCAR`, runs `Restart <leaf-name>` from the leaf's parent directory, captures
  the Slurm job ID, checks `squeue`/`sacct` when available, and writes a timestamped
  TSV launch audit.

Submit the three additional committee members from the directory containing
`train.extxyz`, `valid.extxyz`, and `test.extxyz`:

```bash
for seed in 211 307 419; do
    sbatch --export=ALL,MACE_SEED="$seed" mace_train_committee.sh
done
```

The original `mace_model/TiN_SiN_mace_stagetwo.model` is retained as the first
committee member. New runs are stored under `mace_committee/seed_<seed>/`.

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
