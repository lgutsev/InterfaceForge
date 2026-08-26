# Mapped leaf-dataset campaigns

Use this template when the VASP reference trajectories live in several unrelated
directory trees but must become one synchronized MACE and DeePMD dataset.

The mapper discovers every nonempty `OUTCAR` below each configured source,
creates a clean logical leaf tree using hard links, invokes both collectors with
identical split settings, and then cross-audits their leaf/split membership, frame
counts, and source-frame membership digests. It stages only `OUTCAR`, `INCAR`, `POSCAR`, and `CONTCAR`, so
backup/restart subdirectories cannot make a valid VASP run appear non-terminal.

## Periodic SiN/TiN/TiO campaign

From the repository root:

```bash
launch_scripts/prepare_periodic_nitride_mlips.sh
```

That is a non-mutating dry run. To stage, collect both formats, and create the
visual audit:

```bash
launch_scripts/prepare_periodic_nitride_mlips.sh --execute --collect
```

The default source root is
`/ddnB/work/lgutsev/LATech_PROJS/Cer_Interface`. Override it without changing
the checked-in YAML:

```bash
CER_INTERFACE_BASE=/new/path/Cer_Interface \
  launch_scripts/prepare_periodic_nitride_mlips.sh --execute --collect
```

To intentionally rebuild existing dataset outputs:

```bash
launch_scripts/prepare_periodic_nitride_mlips.sh \
  --execute --collect --force-datasets
```


## LONI batch-launcher convention

Do not set `#SBATCH --mem` or pass `--mem` in InterfaceForge launchers intended
for LONI. Select the appropriate partition and task/CPU count, and allow LONI to
supply the node memory according to its site policy. Its Lua submission wrapper
may warn that manual `--mem` requests are unsupported.

## Choosing the split policy

The split policy is a scientific choice and must be set deliberately in
`collection.split_mode`.

### Broad, varied training data: `random-frame`

Use this when every chemistry, temperature, termination, composition, or other
source category must contribute to training:

```yaml
collection:
  split_mode: random-frame
  ratios: [0.8, 0.1, 0.1]
  seed: 20260730
```

Frames are deterministically shuffled separately within every leaf trajectory.
Every sufficiently populated leaf therefore contributes to train, validation,
and test. MACE and DeePMD receive identical source-frame indices, verified by
membership digests in the audit manifests.

This is the required policy for the periodic SiN/TiN/TiO campaign. With 48
OUTCARs containing 600 retained frames each, it produces approximately 480/60/60
frames per leaf and totals of 23,040 train, 2,880 validation, and 2,880 test
frames. This avoids the earlier failure mode where complete termination branches
were absent from training.

A random frame split can place temporally neighboring MD configurations in
different splits. Validation and test errors may therefore be optimistic. Keep
one or more independently generated trajectories outside the collected dataset
as an external generalization test.

### Leakage-resistant trajectory evaluation: `heritage`

Use this when validation/test must measure transfer to unseen trajectories or
complete campaign branches:

```yaml
collection:
  split_mode: heritage
```

The full parent lineage is indivisible, so related leaves remain in one split.
This reduces trajectory leakage, but with a small number of branches it can
assign an entire chemistry, termination, or temperature category away from
training. Always inspect the SVG/CSV audit before accepting this split.

Rule of thumb:

- Choose `random-frame` when maximizing training-set variety is the priority.
- Choose `heritage` when leakage-free evaluation is the priority.
- For production MLIPs, use `random-frame` for the main diverse dataset and a
  separate independent-trajectory challenge set for rigorous evaluation.

## Future campaigns

Copy `template.yaml`, define one `source` → logical `target` mapping per source
tree, choose a stable element order in `type_map`, and run:

```bash
iface-mapped-collect my-campaign.yaml
iface-mapped-collect my-campaign.yaml --execute --collect
```

The first command is always dry-run. Output layouts are directly compatible with
the defaults used by `iface train mace` and `iface train deepmd`:

```text
datasets/canonical/{train,valid,test}.extxyz
datasets/canonical/deepmd/{train,valid,test}/...
```

The audit writes JSON, CSV, Markdown, and a self-contained SVG dashboard below
the configured audit directory. A nonzero status means the two formats disagree,
a leaf failed conversion, or a requested split is empty.
