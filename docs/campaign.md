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
- `validation`: intended downstream property checks, plus the structured
  `interfaces` / `references` / `reference_profiles` blocks (below).

## `validation` metadata

The property tools (`iface validate interface-energy`, `iface vasp adhesion
audit`) read three optional keys under `validation`:

```yaml
validation:
  interfaces:
    - match: "interface/*/*/N_Term/*"   # fnmatch against the leaf / system id
      stacking_axis: c                  # area plane for interface-energy
      n_interfaces: 2                    # equivalent interfaces per periodic cell
      orientation: "Si3N4(0001)/TiN(111)"
      termination: N
      polar_termination: true            # skip in interface-energy (see below)
  reference_profiles: [sharifi2026]      # bundled; `iface reference list`
  references:                            # or one-off values inline
    - key: in_house
      quantity: work_of_adhesion
      tolerance_j_per_m2: 0.4
      values:
        - {match: {termination: Ti}, value_j_per_m2: 3.10}
```

- **`interfaces`** entries are merged for each leaf (earlier entries win on a
  key). `stacking_axis` and `n_interfaces` become the defaults for
  `iface validate interface-energy` (an explicit `--stacking-axis` /
  `--n-interfaces` still wins). `polar_termination: true` makes that command
  skip the leaf — a (111)/(0001) polar-terminated slab is not an integer count
  of bulk formula units, so the bulk-referenced excess is undefined; use
  `iface validate adhesion` instead.
- **`reference_profiles`** names bundled literature profiles that
  `load_campaign` expands into `references`. `iface reference list` shows what
  is available; `iface reference show <name>` prints the expansion; `iface
  reference activate <name> -c campaign.yaml --write` splices the name into
  this list in place (comment-preserving, dry-run without `--write`).
- **`references`** are literature values for `work_of_adhesion`,
  `interface_energy`, or `surface_energy`. A computed value is compared to
  every entry whose `match` keys all appear (case-insensitively) in the
  interface metadata or the audited CSV row, and the delta plus a
  within-tolerance flag are written into the report. Hand-written entries win
  over a profile entry with the same `(key, quantity)`.

See [interface-energy.md](interface-energy.md) and the work-of-adhesion section
of [vasp.md](vasp.md).

The machine-readable contract is
[`schemas/campaign.schema.json`](../schemas/campaign.schema.json).

## VASP MLFF accuracy profile

VASP-MLFF generation is opt-in and defaults to disabled. Existing DFT-labelled
datasets should retain:

```yaml
stages:
  vasp_mlff:
    enabled: false
```

Omitting `enabled` is also treated as false. Set it to true only when
InterfaceForge should prepare and submit new VASP-MLFF reference runs.

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
