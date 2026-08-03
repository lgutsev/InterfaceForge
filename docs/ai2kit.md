# Optional AI2-Kit adapter

InterfaceForge can generate and supervise an AI2-Kit 1.0.9 config-driven CLL workflow:

```text
DeepMD committee → LAMMPS exploration → model deviation → VASP labels
```

AI2-Kit remains optional and process-isolated. InterfaceForge continues to own the campaign,
canonical datasets, raw-force policy, type ordering, hashes, lineage, standalone MACE and
DeePMD workflows, VASP inputs, and scientific validation. MACE is not supported by this
adapter. The [official AI2-Kit CLL manual](https://github.com/chenggroup/ai2-kit/blob/main/doc/manual/cll-workflow.md)
now describes config-driven CLL as deprecated but maintained for backward compatibility;
the adapter is therefore pinned and deliberately narrow.

## Installation and configuration

Install only when executing the controller:

```bash
python -m pip install -e '.[ai2kit]'
```

Set `active_learning.enabled: true` and fill the explicit thresholds and artifact paths shown
in `examples/ai2kit/campaign.yaml`. Machine-specific SSH, Slurm, executable, job-template,
remote work-directory, and licensed POTCAR paths belong in the scheduler profile; see
`examples/ai2kit/profile_loni.yaml`. Passwords, tokens, and private keys are rejected.

The reviewed MVP is `se_e2_a` with the TensorFlow backend. Other single-architecture modes
require `experimental_compatibility: true`; this is an opt-in boundary, not evidence that
DPA-1/2/3/4 works through AI2-Kit 1.0.9.

## External-controller architecture

On LONI, run the AI2-Kit controller from a workstation/WSL session or a permitted login or
service environment. Do **not** submit the controller as a compute job: LONI compute jobs may
not call `sbatch` for child jobs, and `run --execute` refuses whenever `SLURM_JOB_ID` is set.
AI2-Kit itself connects to the configured login host and submits the DeepMD, LAMMPS, and VASP
jobs represented by the existing InterfaceForge profile templates.

## Supervised operation

```bash
iface active-learning ai2kit export -c campaign.yaml
iface active-learning ai2kit preflight -c campaign.yaml --remote
iface active-learning ai2kit run -c campaign.yaml
iface active-learning ai2kit run -c campaign.yaml --execute
iface active-learning ai2kit status -c campaign.yaml
iface active-learning ai2kit import -c campaign.yaml --round 0 --result-root /path/to/results
iface active-learning ai2kit approve -c campaign.yaml --round 0
```

The first `run` is a dry-run. More than one configured iteration additionally requires
`--allow-multiple-iterations`. Execution requires a successful current remote preflight whose
campaign, profile, and generated-file hashes still match. Existing checkpoints require
`--resume`. Checkpoints are opaque; InterfaceForge never deserializes them with pickle.

Export writes only known files beneath `runs/active_learning/ai2kit/generated`. `--force`
replaces those files but does not delete checkpoints, logs, imports, POTCARs, or results.
POTCAR content is never copied into the adapter output.

Import is staging, not promotion. It writes `accepted.extxyz`, `rejected.csv`, `lineage.csv`,
`import_manifest.json`, and `approval.json` beneath `imports/round_NNN`. It checks species,
cell, coordinates, energies, raw force shapes and finiteness, source hashes, VASP completion
evidence, and duplicate structure identities. The canonical dataset is not modified.

## Recovery and limitations

- If export inputs change, export again with `--force` and repeat remote preflight.
- If the controller is interrupted, preserve `checkpoints/cll` and run with `--resume`.
- Failed execution preserves checkpoints and points to bounded stderr logs.
- YAML parsing, shell syntax checks, and mocked CI are not engine integration tests.
- Autonomous production use is not approved. A real one-round smoke workflow must first
  demonstrate correct remote execution, restart, label convergence, lineage, and independent
  scientific validation.
