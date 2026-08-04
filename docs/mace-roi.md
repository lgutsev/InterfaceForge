# MACE-ROI: interface-local and cycle-consistent training

MACE-ROI is InterfaceForge's experimental MACE training mode for heterogeneous
interfaces. It addresses two mismatches between ordinary atomistic fitting and
the properties an interface model is ultimately asked to predict:

1. A global force loss is dominated by bulk-like atoms when the chemically
   difficult interface region is small.
2. Adhesion, separation and reaction energies are differences among several
   total energies, but ordinary training treats those configurations as
   unrelated samples.

The method keeps MACE's architecture and trainer. InterfaceForge adds a derived,
auditable dataset, a custom loss and a complete-cycle batch sampler. Canonical
DFT energies and forces are never rewritten.

## Loss

For a batch with atoms $i$ and thermodynamic cycles $c$, MACE-ROI minimizes

$$
\mathcal{L} =
\lambda_E \mathcal{L}_E +
\lambda_F \frac{1}{3N}\sum_i w_i\lVert\mathbf{F}_i-\hat{\mathbf{F}}_i\rVert^2 +
\lambda_C \frac{1}{|C|}\sum_{c\in C}
\left[
\frac{\sum_{k\in c}\nu_{ck}(\hat E_k-E_k)}{s_c}
\right]^2.
$$

`w_i` is larger on interface atoms and is normalized to mean one in every
configuration. This changes where the force loss is spent without changing its
overall scale. A cycle coefficient $\nu_{ck}$ can be positive or negative;
$s_c$ is a user-chosen positive energy scale in eV. The cycle term fits the DFT
energy *difference* rather than forcing the difference itself to zero.

Interface atoms are found without assuming that the interface normal is the
Cartesian z axis. An atom is initially selected when its neighbor list contains
an atom from another immutable component. `shell_depth` can add adjacent
neighbor layers.

## Install and configure

Install the pinned MACE adapter separately from the lightweight core:

```bash
pip install -e ".[mace-roi]"
```

Add an `roi` block below `models.mace`:

```yaml
models:
  mace:
    enabled: true
    profile: mace_gpu
    batch_size: 16
    max_num_epochs: 200
    stage2_max_num_epochs: 100
    roi:
      enabled: true
      source_dir: datasets/canonical
      output_dir: datasets/mace_roi
      cutoff: 3.5
      interface_multiplier: 4.0
      shell_depth: 1
      component_key: IF_component
      component_ranges:
        "*interface_ab*": [[0, 72], [72, 144]]
      cycle_manifest: cycles.csv
      stage1_cycle_weight: 0.25
      stage2_cycle_weight: 1.0
```

`component_ranges` uses half-open atom-index intervals. It can be one range list
for a homogeneous dataset or a mapping from `source_run` glob patterns to range
lists. Unmatched sources, such as bulk and isolated slabs, receive unit weights.
Overlapping patterns are rejected. As an alternative, every extxyz frame may
carry an integer per-atom `IF_component` array; that array takes precedence.
Atom ordering and component membership must remain stable over a trajectory.
Copy the exact `source_run` values from `datasets/canonical/frames.csv` when
choosing patterns.

For cycle-only training, set `interface_multiplier: 1.0`; no interface ranges
are then required. With a multiplier greater than one, preparation fails if it
finds no cross-component neighbors, catching bad source patterns and cutoffs.

## Cycle manifest

Use `datasets/canonical/frames.csv` to identify exact frames. The CSV columns
are `split`, `source_run`, `source_frame`, `cycle_id`, `coefficient`, and optional
`scale_ev`:

```csv
split,source_run,source_frame,cycle_id,coefficient,scale_ev
train,interface_ab_300k,120,adhesion_120,1,1.0
train,slab_a_300k,80,adhesion_120,-1,1.0
train,slab_b_300k,95,adhesion_120,-1,1.0
```

Each frame can belong to at most one cycle in this first implementation. During
preparation InterfaceForge requires at least two members, a consistent scale,
composition conservation $\sum_k\nu_{ck}n_{kZ}=0$ for every element, and one
train/validation/test split per cycle. It never moves frames between canonical
splits. A leaking or incomplete cycle fails loudly.

## Run

```bash
iface collect
iface mace-roi prepare
# Inspect datasets/mace_roi/manifest.json and sample IF_roi_mask values.
iface train mace
sbatch models/mace/stage1/run.slurm
sbatch models/mace/stage2/run.slurm
```

`iface mace-roi prepare --cycles another.csv` overrides the configured cycle
CSV. `--source` and `--output` override dataset paths. Rebuilding a non-empty
derived directory requires `--force`; canonical data cannot be selected as or
nested inside the output.

The preparation manifest records source and output hashes, ROI fractions and
cycle counts. Generated training manifests point to it. Stage two correctly
uses `stage1_epochs + stage2_epochs` as MACE's absolute stopping epoch.

Evaluate a trained model with MACE's standard `MACE_` output prefix, then let
InterfaceForge report the ablation-relevant subsets:

```bash
mace_eval_configs \
  --configs=datasets/mace_roi/test.extxyz \
  --model=models/mace/artifacts/interfaceforge_mace.model \
  --output=test-predictions.extxyz
iface mace-roi evaluate test-predictions.extxyz test-mace-roi-metrics.json
```

The JSON contains energy-per-atom errors, global/ROI/non-ROI force-component
errors, force-vector RMSEs, and both eV and scaled thermodynamic-cycle
residuals. Override the four label/prediction keys on the evaluation command if
the extxyz uses a nonstandard prefix.

## Current compatibility and evaluation

The adapter supports `mace-torch>=0.3.17,<0.4`. Region weighting can use the
ordinary single-process generated jobs. A nonzero cycle loss is intentionally
rejected under distributed training because complete-cycle reduction across
ranks has not yet been implemented. A cycle larger than `batch_size` becomes
one oversized batch instead of being split.

For a publishable study, compare at least four matched-seed conditions:

| Model | ROI forces | Cycle loss |
|---|---:|---:|
| baseline MACE | no | no |
| region-only | yes | no |
| cycle-only | no | yes |
| MACE-ROI | yes | yes |

Report global and interface-only force errors alongside work of adhesion,
separation curves, unseen terminations and short interface MD stability. The
scientific claim should come from that ablation, not from the mechanism alone.
