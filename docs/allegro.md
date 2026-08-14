# Allegro backend

> **Verification note:** code-only. No InterfaceForge-generated Allegro model has
> yet been trained, compiled, or run through LAMMPS under human inspection.

InterfaceForge treats Allegro as an optional MLIP backend for workloads where a strictly local equivariant model is attractive, especially large LAMMPS interface simulations. The adapter does not assume that the cluster-provided LAMMPS is usable. Instead it generates training, model-compilation, LAMMPS-build, runtime-preflight, and MD launcher assets under `models/allegro/`.

## Install

Install the InterfaceForge adapter and the current Allegro extension in a dedicated environment:

```bash
pip install -e ".[allegro]"
```

For LONI, set `ALLEGRO_ACTIVATE_SCRIPT` in the environment used by the generated Slurm jobs. Keep this environment consistent with the PyTorch/libtorch toolchain used to build pair_allegro.

## Campaign configuration

Enable the backend explicitly and provide an atom-type order. InterfaceForge refuses to infer the LAMMPS type mapping.

```yaml
models:
  allegro:
    enabled: true
    profile: allegro_gpu
    lammps_profile: allegro_lammps
    train_file: datasets/canonical/train.extxyz
    valid_file: datasets/canonical/valid.extxyz
    test_file: datasets/canonical/test.extxyz
    type_names: [Ni, O, C, H, N, P]
    r_max: 5.0
    batch_size: 4
    max_epochs: 200
    learning_rate: 0.001
    l_max: 1
    num_layers: 2
    num_scalar_features: 64
    num_tensor_features: 32
    compile_device: cuda
```

The canonical extxyz labels are read from `REF_energy` and `REF_forces`, matching the InterfaceForge dataset convention.

## Generate training and deployment assets

```bash
iface-allegro generate -c campaign.yaml
```

This writes:

- `models/allegro/config.yaml` — NequIP/Allegro training configuration.
- `models/allegro/run_train.slurm` — restart-capable Lightning/NequIP training job.
- `models/allegro/run_compile.slurm` — AOTInductor compilation job targeting `pair_allegro`.
- `models/allegro/lammps/build_pair_allegro.sh` — reproducible LAMMPS + Kokkos + pair_allegro build helper.
- `models/allegro/lammps/run_lammps.slurm` — guarded production launcher.
- `models/allegro/training_manifest.json` — provenance and deployment requirements.

Training is launched with `nequip-train`. If a training job is interrupted, resume it using the same generated configuration and an explicit Lightning checkpoint; do not modify the model configuration between restarts.

## Compile the trained model

The generated compilation launcher requires an explicit checkpoint:

```bash
export ALLEGRO_CHECKPOINT=/path/to/last.ckpt
sbatch models/allegro/run_compile.slurm
```

The default output is:

```text
models/allegro/compiled/model.nequip.pt2
```

AOTInductor compilation is hardware-specific. Run `nequip-compile` on the same GPU type used for production inference. The generated path targets `pair_allegro` and requires PyTorch 2.6 or newer.

## Build pair_allegro LAMMPS

The generated build helper follows the modern pair_nequip_allegro requirements:

- LAMMPS 10 Sep 2025 or newer.
- Kokkos compiled in its default double-double precision mode.
- CUDA Kokkos for the generated GPU launcher.
- PyTorch/libtorch ABI compatibility.
- `NEQUIP_AOT_COMPILE=ON` for `.nequip.pt2` models.

Run it inside the Allegro environment on the target cluster:

```bash
cd models/allegro/lammps
./build_pair_allegro.sh
```

Useful overrides include `LAMMPS_REF`, `PAIR_ALLEGRO_REF`, `CUDA_TOOLKIT_ROOT_DIR`, `CMAKE_CXX_COMPILER`, `KOKKOS_ARCH`, and `ALLEGRO_LAMMPS_BUILD_ROOT`. The defaults intentionally pin a known minimum LAMMPS release and a recorded pair_nequip_allegro commit instead of following moving branches silently.

## Preflight before MD

Never start a long MD job without checking the runtime first:

```bash
iface-allegro lammps-preflight \
  --lammps /path/to/lmp \
  --model models/allegro/compiled/model.nequip.pt2
```

The preflight verifies that the executable exists, parses a sufficiently new LAMMPS release, checks that Kokkos and `pair_style allegro` appear in `lmp -h`, and verifies the compiled model path and extension. Kokkos precision cannot be proven from `lmp -h`, so the build configuration still must be inspected once.

## Production launch

Set the explicit runtime inputs and submit the generated job:

```bash
export ALLEGRO_LAMMPS=/path/to/pair_allegro-enabled/lmp
export ALLEGRO_MODEL=$PWD/models/allegro/compiled/model.nequip.pt2
export LAMMPS_INPUT=$PWD/in.allegro
sbatch models/allegro/lammps/run_lammps.slurm
```

The GPU launcher uses Kokkos with `newton on` and half neighbor lists and runs `iface-allegro lammps-preflight` before starting LAMMPS. This is intentionally conservative: a missing pair style, stale LAMMPS, or wrong compiled-model path should fail before consuming a long allocation.

In the LAMMPS input, the type mapping must match the training `type_names` order exactly, for example:

```text
pair_style allegro
pair_coeff * * /path/to/model.nequip.pt2 Ni O C H N P
```

A wrong type mapping can produce physically nonsensical or immediately unstable trajectories even when the model and LAMMPS binary load successfully.
