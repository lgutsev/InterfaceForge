# DeePMD campaigns

InterfaceForge validates the canonical `train`, `valid` and `test` NPY systems
before generating any model directory. Every split must be non-empty and every
system must use the same `type_map.raw`.

## Architecture and backend matrix

| Architecture | Generated descriptor | Backend |
|---|---|---|
| DPA-1 | `se_atten` (`dpa1` for experimental implementation) | TensorFlow, PyTorch, `pt_expt` |
| DPA-2 | `dpa2` | PyTorch or `pt_expt` |
| DPA-3 | `dpa3` | PyTorch or `pt_expt` |
| DPA-4 | `dpa4` + `dpa4_ener` | PyTorch or `pt_expt`; experimental deployment |
| classic | `se_e2_a` | TensorFlow, PyTorch |

The input shapes mirror the supplied modern campaign primer and remain fully
editable after generation.

## Generated jobs

1. `run_preflight.slurm` verifies the container/runtime, a visible GPU, the
   DeePMD version and backend help.
2. `run_smoke.slurm` runs one isolated short model per architecture, freezes it,
   and tests up to 100 frames from each test system.
3. `run_ensemble.slurm` trains every architecture/seed pair as a bounded Slurm
   array, resumes existing checkpoints, freezes models and evaluates test data.
4. `run_evaluate.slurm` runs per-model tests and committee `model-devi` on every
   test system.

Smoke output is written under a job-specific directory and never overwrites a
full model.

## Containers

Set `container_image` in the campaign or omit it when DeePMD is activated
directly by the scheduler profile. Generated container jobs discover Apptainer
or Singularity and bind `/ddnB` and `/project` only when those roots exist.

The image path can be overridden at submission time:

```bash
DEEPMD_IMAGE=/new/path/deepmd.sif sbatch models/deepmd/run_preflight.slurm
```

Do not infer production readiness from training success alone. Verify the
frozen model with `dp test`, committee deviation, and the exact downstream MD
engine.
