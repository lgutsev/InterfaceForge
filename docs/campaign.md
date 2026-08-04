# Campaign format

`campaign.yaml` is the source of truth. Paths are resolved relative to that
file, which makes a campaign movable between a workstation and a cluster.

## Main sections

- `project`: name and human description.
- `profile`: local or Slurm resource profile.
- `reference`: VASP input templates. POTCAR is a local input, never distributed.
- `systems`: stable IDs, physical kinds, structure paths and optional metadata.
- `stages.vasp_mlff`: train, refit and stability settings.
- `dataset`: stride, split ratios, leakage strategy and type map.
- `models`: MACE and DeePMD generation settings. `models.mace.roi` enables the
  derived-data and loss adapter described in the [MACE-ROI guide](mace-roi.md).
- `exploration`: temperatures, strains and replicas.
- `validation`: intended downstream property checks.

The machine-readable contract is
[`schemas/campaign.schema.json`](../schemas/campaign.schema.json).

## Dataset strategies

`grouped` assigns every retained frame from an OUTCAR to one split. It is the
default because it provides the clearest defense against temporal leakage.

`guarded` divides each trajectory into contiguous blocks, rotates block order
between trajectories, and drops `guard_frames` between blocks. It is useful
when too few independent trajectories exist, but it should be treated as a
fallback.

In both modes, VASP force labels remain raw. Selective Dynamics becomes a
separate `move_mask` array.

## Portability

Put scheduler/account/container differences in `profiles/*.yaml`, not in
campaign data. Generated paths may be absolute because engine input formats
often require them; regenerate training jobs after moving a campaign.
