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
| DPA-2 fine-tune | `dpa2_ft` (dpa2 structure, fine-tuned) | PyTorch or `pt_expt` |
| DPA-3 | `dpa3` | PyTorch or `pt_expt` |
| DPA-4 | `dpa4` + `dpa4_ener` | PyTorch or `pt_expt`; experimental deployment |
| classic | `se_e2_a` | TensorFlow, PyTorch |

The input shapes mirror the supplied modern campaign primer and remain fully
editable after generation.

### Fine-tuning DPA-2 from a foundation checkpoint

`dpa2_ft` is a fine-tuning run of the `dpa2` architecture. It coexists with a
from-scratch `dpa2` entry in the same committee (own `models/deepmd/dpa2_ft/`
tree and evaluation), so the two can be compared directly. It requires a
`finetune` block:

```yaml
models:
  deepmd:
    backend: pt_expt
    architectures: [dpa2, dpa2_ft, dpa3]
    committee: 4
    seeds: [11, 23, 37, 53]
    finetune:
      pretrained: /project/lgutsev/models/dpa2_openlam.pt   # a DPA-2 checkpoint
      model_branch: RANDOM     # a named multi-task head, or RANDOM to reinit fitting
```

The generated `run_ensemble.slurm` runs, only for `$ARCH == dpa2_ft` and only on
the first pass (no local checkpoint yet):

```
dp --pt train input.json --finetune <pretrained> --model-branch <model_branch>
```

`--restart` takes precedence once a `model.ckpt.pt` exists, so continuation runs
behave like any other architecture. `RANDOM` keeps the pretrained descriptor and
reinitializes the fitting net — the most robust default across checkpoint
versions; a named branch requires the input-side descriptor to match that
branch. Verify `dp --pt train --help` in your DeePMD-kit build lists `--finetune`
and `--model-branch` before submitting, and run `run_smoke.slurm` first.

## Generated jobs

1. `run_preflight.slurm` verifies the container/runtime, a visible GPU, the
   DeePMD version and backend help.
2. `run_smoke.slurm` runs one isolated short model per architecture, attempts an
   export, and tests up to 100 frames from each test system.
3. `run_ensemble.slurm` trains every architecture/seed pair as a bounded Slurm
   array, resumes existing checkpoints, attempts exports and evaluates test data.
4. `run_evaluate.slurm` runs per-model tests and committee `model-devi` on every
   test system, then writes exact component-weighted RMSE reports.

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

For PyTorch, audit jobs use `model.ckpt.pt` directly. This avoids making model
quality assessment depend on TorchScript export compatibility. Export remains
a separate deployment gate: a failed or unreadable `frozen_model.pth` does not
invalidate a working training checkpoint, but that checkpoint is not yet
approved for LAMMPS. TensorFlow audits continue to use the frozen model.

The evaluation directory contains:

- `rmse_by_system.csv`: energy, force and optional virial errors for every
  model/system pair;
- `rmse_overall.csv`: committee-member metrics accumulated from the underlying
  squared errors over all test observations;
- `rmse_audit.json`: the same results with full structure and provenance.

Energy RMSE is reported per atom, force RMSE over Cartesian force components,
and virial RMSE per atom. InterfaceForge never averages already-computed system
RMSE values.

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
ensemble until the smoke job has trained and tested its short DPA-2 checkpoint.
The ensemble manifest records the exact scheduler profile, module, evaluation
artifact and expected reports.

The LONI DeePMD module already wraps its executable in a container/MPI runtime.
Generated jobs therefore call `dp` directly inside the batch allocation rather
than nesting that wrapper inside a second `srun` job step.

## Auditing the older LAMMPS/DeePMD module

LONI also supplies `lammps/29Aug2024-r8.0-deepmd3.0.0-gpu`, which has worked
with DPA-1 models. Audit its compiled pair style independently:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/deepmd_lammps_30_gpu_audit.sbatch
```

This confirms that the module exposes `pair_style deepmd`, but it cannot prove
that a model frozen by DeePMD 3.2 is readable by the older 3.0 runtime. After
the four-model ensemble is frozen, place one absolute model path per line in
`committee-models.txt`, select a canonical test system containing `set.000`,
and submit:

```bash
sbatch --export=ALL,DEEPMD_MODELS_FILE=/absolute/committee-models.txt,DEEPMD_SYSTEM=/absolute/test/system \
  /path/to/InterfaceForge/launch_scripts/deepmd_lammps_30_gpu_audit.sbatch
```

The audit converts the first canonical frame into a temporary LAMMPS data
file, loads every committee model through `pair_style deepmd`, runs one step,
and requires model-deviation output. A failure here is a runtime compatibility
failure even when `dp test` under the 3.2 training module succeeds.
