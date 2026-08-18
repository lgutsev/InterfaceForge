# Reproducible LONI environments

InterfaceForge keeps AI2-Kit orchestration separate from the GPU MACE/OpenMM
runtime.  This avoids forcing AI2-Kit's older NumPy constraints into the MACE
training environment and preserves the known-good training prefix.

| Purpose | Default prefix | Builder |
|---|---|---|
| AI2-Kit controller | `/project/lgutsev/env/iface_ai2kit_controller` | `examples/ai2kit/setup_iface_ai2kit_controller.sh` |
| MACE/OpenMM runtime | `/project/lgutsev/env/iface_mace_runtime` | `examples/ai2kit/setup_iface_mace_runtime.sh` |
| Source MACE environment | `/project/lgutsev/env/mace_env` | never modified by the runtime builder |

Submit the builders to a CPU partition because cloning a large Conda prefix can
take tens of minutes on the shared filesystem.  An apparently quiet clone is
not necessarily stuck.  Each builder validates an existing prefix first and
moves an invalid target to a timestamped `.incomplete-*` directory rather than
deleting it.  After the replacement is validated, old incomplete directories
may be removed manually.

The available `pytorch/*` modules are useful for software designed around those
modules, but should not be mixed automatically into a cloned Conda MACE runtime:
they can shadow the runtime's tested PyTorch/CUDA pair.  GPU jobs request a GPU
from Slurm and use the Conda PyTorch build by default.  Set `IFACE_GPU_MODULE`
only when a site-specific module combination has been tested as a unit.

## Rebuild records

Create small, text-only environment records (not package archives):

```bash
bash examples/ai2kit/export_environment_manifests.sh
```

The timestamped output contains portable YAML, exact Conda URLs, pip freezes,
and original prefixes for the controller, runtime, development, and source MACE
environments.  A portable first attempt is:

```bash
conda env create --prefix /new/prefix --file iface_mace_runtime.yml
```

Use the explicit list when reproducing on the same platform and the pip freeze
to diagnose packages installed outside Conda.

## GPU smoke test

The smoke test accepts either a model file or a seed directory.  For a seed
directory it searches `mace_model/` and prefers a filename containing
`stagetwo`, `stage_two`, or `stage2`.  It then compares energy and forces from
native MACE and OpenMM-ML on one ASE-readable structure:

```bash
sbatch examples/ai2kit/run_mace_openmm_smoke.sh \
  /path/to/mace_committee/seed_0 \
  /path/to/representative.extxyz
```

For the current LONI campaign, the seed example is
`/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface/MD/MACE/mace_dataset_all/mace_committee/seed_0`,
whose stage-two model is below `mace_model/`.  The structure remains an explicit
argument so the repository does not encode one user's dataset location.
