# Mapped leaf-dataset campaigns

Use this template when the VASP reference trajectories live in several unrelated
directory trees but must become one heritage-safe MACE and DeePMD dataset.

The mapper discovers every nonempty `OUTCAR` below each configured source,
creates a clean logical leaf tree using hard links, invokes both collectors with
identical split settings, and then cross-audits their membership, split assignment,
and frame counts. It stages only `OUTCAR`, `INCAR`, `POSCAR`, and `CONTCAR`, so
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
