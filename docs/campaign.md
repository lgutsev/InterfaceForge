# Campaign format

> **Verification note:** the VASP-MLFF campaign path and the MACE/DeePMD data
> exporters have been exercised on real data. The complete multi-engine
> campaign lifecycle, portability, and restartability have not been verified.

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

## VASP MLFF accuracy profile

New campaign templates opt into VASP's accuracy-oriented two-stage recipe:

```yaml
stages:
  vasp_mlff:
    accuracy_profile: accurate
```

For `train`, this sets `ML_IALGO_LINREG=1`, `ML_SION1=0.3`, and
`ML_MRB2=12`. For `refit`, it sets the SVD solver
`ML_IALGO_LINREG=4`, restores `ML_SION1=0.5`, retains `ML_MRB2=12`, and
sets the refit sparsification default `ML_EPS_LOW=1E-11`. Remove the profile
to retain VASP defaults and the values in the reference INCAR.

The profile is applied while scaffolding a new campaign. InterfaceForge does
not retrofit these descriptor and solver choices into an existing continuation
database because the full on-the-fly training is intended to use a consistent
recipe from the start.

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
