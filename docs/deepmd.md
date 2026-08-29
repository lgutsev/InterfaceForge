# DeePMD campaigns

> **Verification note:** only generation of DeePMD NPY data from real VASP
> trajectories has been human-tested. The generated training, evaluation,
> freeze/export, and LAMMPS paths are code-only and remain unverified.

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

## LONI DeePMD-kit 3.2 module

The `deepmd_gpu_320` profile uses the native LONI module
`deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu`. It is separate from `deepmd_gpu`, so an
older campaign using an activated environment or container is not silently
moved to a different DeePMD runtime. Neither profile requests memory manually.

Use PyTorch and begin with one DPA-2 architecture for a four-model committee:

```yaml
models:
  deepmd:
    enabled: true
    profile: deepmd_gpu_320
    backend: pytorch
    architectures: [dpa2]
    committee: 4
    seeds: [11, 23, 37, 53]
    numb_steps: 500000
    batch_atoms: 1024
    max_concurrent: 2
```

Before generating or submitting production training, run the independent site
module preflight:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/deepmd_32_gpu_preflight.sbatch
```

Then generate jobs from the campaign root and execute them in order:

```bash
iface train deepmd
sbatch models/deepmd/run_preflight.slurm
sbatch models/deepmd/run_smoke.slurm
sbatch models/deepmd/run_ensemble.slurm
sbatch models/deepmd/run_evaluate.slurm
```

Wait for each preceding stage to succeed. In particular, do not submit the full
ensemble until the smoke job has trained, frozen, and tested its short DPA-2
model. The ensemble manifest records the exact scheduler profile and module.
