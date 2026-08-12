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

Submit the three additional committee members from the directory containing
`train.extxyz`, `valid.extxyz`, and `test.extxyz`:

```bash
for seed in 211 307 419; do
    sbatch --export=ALL,MACE_SEED="$seed" mace_train_committee.sh
done
```

The original `mace_model/TiN_SiN_mace_stagetwo.model` is retained as the first
committee member. New runs are stored under `mace_committee/seed_<seed>/`.
