# Geometry-stratified validation

> **Verification note:** code-only. The classification and reporting
> machinery has automated coverage; it has not yet been run against a real
> trained committee's predictions.

A single pooled error number over a randomly sampled validation set can
hide a model that is excellent on the majority class and badly wrong on a
minority one — e.g. a MACE committee trained mostly on TiN/SiN interface
frames that is unreliable for an isolated bulk-like or surface
configuration (see [`--slab-mode static`](vasp.md#work-of-adhesion) for one
concrete case this already surfaced). This is a first step toward reporting
errors per geometry class instead of only one pooled figure, and applies to
any campaign, not just TiN/SiN.

## Classifying frames

`iface collect` now tags every frame it writes with:

- **`kind`**: `bulk`/`surface`/`interface`/`molecule`/`adsorbate`/`defect`/
  `other`, taken from the declared `kind` of whichever `campaign.yaml`
  system's `run_glob` matches that trajectory's path (relative to the
  dataset source root), e.g.:

  ```yaml
  systems:
    - id: tin-sin-interface
      kind: interface
      structure: structures/interface.vasp
      run_glob: "*/interface_*/*"
    - id: tin-bulk
      kind: bulk
      structure: structures/tin_bulk.vasp
      run_glob: "*/bulk_tin_*/*"
  ```

  A trajectory matching no system's `run_glob` (including every existing
  campaign that hasn't set it yet) is reported `unclassified` rather than
  guessed — `iface collect`'s output notes how many trajectories fell into
  that bucket, and old campaigns keep collecting exactly as before.
- **`tebeg_k`** and **`high_temperature`**: `tebeg_k` is read from the
  `INCAR` beside each `OUTCAR`; `high_temperature` compares it against the
  highest temperature in `exploration.temperatures`, falling back to a
  literal 600 K when none is configured.
- **`min_coordination_number`** / **`mean_coordination_number`**: per-atom
  coordination number using a covalent-radii cutoff (`ase.neighborlist.
  natural_cutoffs`, `mult=1.2`), so no per-element reference table is
  needed and this applies unchanged to a future NiO/phosphonate campaign.
  Computed per frame; degrades to empty (never raises) when a frame's
  geometry can't support a neighbor-list build.

All five are written both to `datasets/canonical/frames.csv` and into each
extxyz frame's `info` dict (`IF_kind`, `IF_tebeg_k`, `IF_high_temperature`,
`IF_min_coordination_number`, `IF_mean_coordination_number`).

## Reporting per-class errors

```bash
iface validate stratified predictions.csv results.csv
```

expects a CSV with `reference`/`predicted` columns plus whichever of
`kind`/`high_temperature`/`min_coordination_number` are available —
typically produced by joining `frames.csv` against a predictions file on
`run_id`/`source_frame`. It reports one row per class:

- `overall`;
- one `kind=...` row per distinct kind present;
- `high_temperature`;
- `low_coordination` — the threshold is the CSV's own
  `--low-coordination-percentile` (default 10th percentile) of
  `min_coordination_number`, not a hardcoded coordination number, so it
  needs no per-element convention and carries over to a different
  chemistry unchanged.

Each class is an **independent slice, not a mutually exclusive partition**:
a frame can count toward several rows at once (e.g. an interface frame that
is also high-temperature), which is how a real mixed campaign actually
looks. Any of the three optional columns may be absent — the corresponding
rows are simply omitted, not an error, since not every campaign has all of
them yet.

## What this does not yet do

- **Splitting** (`train`/`valid`/`test` assignment) is unchanged — still
  `assign_grouped`'s trajectory/category balancing, not yet stratified by
  these classes. Deliberately deferred to keep this pass scoped to
  reporting.
- **Physical tests per class** (a Wadh-sanity check for `interface`, a
  formation-energy-sign check for `defect`, and so on) are not built yet —
  they need per-class scientific criteria to be right rather than
  mechanical, so they're being proposed for review before anything is
  implemented.
