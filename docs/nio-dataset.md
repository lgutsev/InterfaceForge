# Canonical NiO dataset and leakage-safe splits

> **Verification note:** automated-test-only. The exporter, frame QC, grouping,
> split, verification and readiness audit are regression tested on synthetic
> VASP trees that mimic the NiO `Step1/` → `Step2_<T>K/` layout. They have not
> yet been run on the real LONI NiO + phosphonate trajectories. See
> [Verification and maturity](verification.md).

`iface dataset` turns the NiO (+ phosphonate) AIMD trees into **one** canonical
dataset that MACE, DeePMD/DPA and NequIP all train and test on, with one
split that never puts correlated frames on both sides of a train/test boundary.

## Why a NiO-specific exporter

The NiO campaigns are organised as stage trees below a project head:

```text
<head>/Step1/<OH>/<case>/          preheat (INCAR, POSCAR, OUTCAR[.gz], OSZICAR, CONTCAR)
<head>/Step2_300K/<OH>/<case>/     production AIMD from the Step1 CONTCAR
<head>/Step2_450K/<OH>/<case>/     ... same starting structure, higher temperature
<head>/Step2_600K/<OH>/<case>/     (each Step2 root may carry step2_sample.json)
```

Every `Step2_<T>K/<case>` run starts from the **same** `Step1/<case>` CONTCAR, so
the temperature series of one case is one correlated lineage. The generic
collectors (`iface collect`, `iface-mapped-collect`) group by parent directory
and would therefore split `Step2_300K/<case>` and `Step2_450K/<case>` into
different partitions. The NiO exporter instead groups whole **cases**.

## Commands

```bash
# Read-only: inventory, frame QC and the split an export would produce
iface dataset discover /path/to/NiO_head
iface dataset discover /path/to/NiO_head --full --output-json audit/nio_discover.json

# Write the canonical dataset (refuses a non-empty output unless --force)
iface dataset export /path/to/NiO_head --output datasets/canonical

# Re-check hashes, frame identity, split identity, extxyz <-> DeePMD agreement
iface dataset verify datasets/canonical

# Pre-GPU readiness audit (JSON + Markdown); see "Readiness audit" below
iface dataset readiness /path/to/NiO_head --output audit/nio_readiness \
    --dataset datasets/canonical -c campaign.yaml
```

Several project heads may be passed at once. Every command accepts the same
selection options (or a YAML/JSON `--config`, including a previous export's
`manifest.json`, so an export can be reproduced exactly):

| Option | Default | Meaning |
|---|---|---|
| `--stages` | `Step2` | `Step1`, `Step2`, `OPT` and/or `unstaged` trajectories to export. The default follows the documented `training` protocol (decorrelated Step2 frames); Step1 preheats are reported but not exported unless requested. |
| `--no-step2-sample` | honour it | Ignore `step2_sample.json` decorrelated frame indices. |
| `--stride` | 1 | Every Nth valid frame when no `step2_sample.json` selection applies. |
| `--include-incomplete` | off | Also export runs whose OUTCAR has no completion marker (still running, killed, crashed). |
| `--keep-scf-unconverged` | off | Keep MD steps whose electronic loop hit `NELM`. |
| `--keep-post-runaway` | off | Keep frames at/after a temperature runaway. |
| `--max-force`, `--max-md-temperature` | none | Optional explicit rejection thresholds (eV/Å, K). No threshold is invented by default. |
| `--ratios` | 0.8 0.1 0.1 | Target frame fractions. |
| `--seed` | 20260730 | Split seed. |
| `--group-by` | `case` | Leakage unit: `case`, `surface` (all ligand/anchor variants of one hydroxylated slab together) or `coverage` (whole OH levels). |
| `--stratify-by` | `ligand` | Balance splits within chemistry strata (`ligand`, `coverage`, `motif`, `pattern`, `anchor`, `stage`, `none`). |
| `--split-method` | `balanced` | `balanced` (largest frame deficit, stratum then global) or `hash` (per-group hash: adding cases never moves existing ones). |
| `--type-map` | sorted elements | Explicit element order; must cover every element present. |

## Discovery

Every `OUTCAR` / `OUTCAR.gz` below the given heads is a candidate trajectory.
`archive/`, `backup*`, `precondition/`, `smoke/`, `restart_archive_*`,
`refit_archive_*`, `stability_archive_*`, `.interfaceforge/` and `X*` branches
are skipped, matching the rest of InterfaceForge. A leaf does **not** have to be
a terminal directory: a conservative NiO Step1 run legitimately contains a
`precondition/` child.

For each trajectory the exporter records the stage (from the nearest
`Step1` / `Step2_<T>K` / `OPT` directory), the temperature (directory name,
otherwise INCAR `TEEND`/`TEBEG`, with its source), the case path below the stage
directory, and the chemistry parsed from the notebook naming convention
(`NiO_m110_Big_U46[_OH<pct>_<pattern>_<motif>][_<ligand>[_<anchor>]]`).
Unknown name tokens are listed as unparsed rather than guessed. Composition,
atom count and formula always come from the frames; a name that claims a ligand
while the frames contain no P (or the reverse) is reported.

## Frame quality control

Nothing is silently accepted or silently dropped. Every rejected frame is listed
in `rejected_frames.csv` with its reason, and every trajectory in
`trajectories.csv` with its status and warnings.

| Check | Action |
|---|---|
| OUTCAR ionic blocks vs frames ASE could parse | ASE silently ignores a truncated final ionic step; the exporter counts `TOTAL-FORCE` blocks itself and reports each unparsed step. |
| OUTCAR completion marker | Missing → trajectory `incomplete`, excluded unless `--include-incomplete`. |
| Missing/non-finite energy or forces, wrong force shape, degenerate cell | Frame rejected. |
| Atom identity/order changes within a run, or differs from POSCAR/CONTCAR | Frame rejected (DeePMD systems require a fixed order). |
| OSZICAR electronic iterations ≥ `NELM` | Frame rejected (SCF not converged). |
| Temperature runaway (the Step1 repair diagnostic: `T > max(1200 K, 4 × TEBEG)`) | Frames at and after the first runaway step rejected. |
| Energy-reference excursion only (`\|F − Fref\| > 50 eV`) | **Reported, not rejected** — [NiO AIMD policy](nio-aimd.md) documents known false positives for stable trajectories. |
| OSZICAR/OUTCAR step-count mismatch, OUTCAR error markers, missing constraints | Warning. |
| Byte-identical OUTCAR under two paths | Second copy excluded as a duplicate. |

## Step2 sampling provenance (`step2_sample.json`)

`step2_sample.json` (written by `iface vasp step2-sample`) selects the
decorrelated frames of each Step2 run. Once a Step2 root has one, it governs
every run in that root: the exporter never falls back to stride selection
behind your back. For each Step2 trajectory it records a `sample_state`:

| State | Meaning | Blocking |
|---|---|---|
| `absent` | No `step2_sample.json` in the Step2 root; stride selection applies | no |
| `disabled` | `--no-step2-sample` (or `use_step2_sample: false`); stride selection | no |
| `used` | A valid `OK` entry selects the frames | only if requested frames are missing |
| `pending` | Listed as not `OK` (or not listed) and the run has no frames yet | no |
| `invalid` | The file or this run's entry is malformed: not JSON, `runs` not a list, a run entry without `relative_path`, unexpected `format`, non-string `status`, `indices` missing or not a list, non-integer, negative or duplicate indices, `kept_frames` disagreeing with `indices`, or several entries for the run with different selections (identical repeats are accepted with a warning) | yes |
| `stale` | The run has frames, but the manifest does not list it or lists it as not `OK` | yes |

For a `used` trajectory the requested indices are partitioned exactly:

- `requested = present + missing`, where *present* means parsed from the OUTCAR;
- `present = selected + rejected_by_qc`;
- `exported = selected` when the trajectory is exported, otherwise 0.

A requested frame that is **missing** (beyond the parsed OUTCAR, or after the
parser stopped at a truncated/unreadable step) marks the manifest as stale:
the trajectory gets status `sampling_inconsistent`, `iface dataset export`
refuses to write the dataset, and `iface dataset readiness` reports it as
blocking. The error names the trajectory, the sampling file, the missing
frame(s) and the parsed frame range. An `invalid` or `stale` manifest gives
status `sampling_invalid` and blocks in the same way. A requested frame that
is present but fails QC (SCF ceiling, temperature runaway, non-finite labels,
missing virial, an explicit threshold) stays rejected; readiness lists it with
its reason under "Needs attention" and in the Step2 sampling section, but it
does not block.

`trajectories.csv` carries `sample_state`, `sample_manifest`,
`sample_manifest_sha256`, `sample_run_status`, the six `sample_count_*`
columns (`requested`, `present`, `rejected_by_qc`, `selected`, `missing`,
`exported`), the corresponding space-separated `sample_indices_*` lists and
`sample_errors`, so the outcome can be reconstructed without reopening the
sampling file. `step2_sample_indices` is kept as a legacy alias of
`sample_count_requested`. `manifest.json` has a compact `sampling` section
(state counts, totals and the sha256 of every sampling manifest used).

Reference labels follow the repository convention: `REF_energy` is ASE's
`energy(sigma->0)` in eV (TOTEN is kept as `REF_free_energy`), `REF_forces` are
raw VASP forces in eV/Å and are **never** zeroed on frozen atoms; constraints
are stored as a `move_mask` from POSCAR selective dynamics.

## Grouping and splitting

The split unit is a leakage group, never a frame:

1. The configured key (`case` by default) forms the base group, so every stage
   and temperature of a case lands in one split.
2. Groups are merged (union-find) when two trajectories start from an
   **identical structure**, or when one trajectory's final `CONTCAR` is another's
   starting `POSCAR` — the Step1 → Step2 hand-off is detected from the
   geometry itself, even if directories were renamed or moved between heads.
3. Groups are assigned deterministically (`balanced` or `hash`, seeded) and every
   requested split is guaranteed a group when there are enough groups.
4. An independent post-hoc check re-verifies group membership, identical
   starting structures and CONTCAR→POSCAR lineage across splits. An export with
   detected leakage is refused.

Under `--group-by case`, one hydroxylated slab decorated with different
ligands may appear in more than one split; the audit reports how many such
"surface families" are shared. Use `--group-by surface` for a stricter
ligand-transfer benchmark.

## Output (the single canonical dataset)

```text
datasets/canonical/
  train.extxyz valid.extxyz test.extxyz      MACE and NequIP (exact float64 round trip)
  deepmd/{train,valid,test}/<trajectory>/    DeePMD/DPA systems (+ move_mask.npy, system_meta.json,
                                             frame_map.csv with relative_leaf == extxyz IF_leaf)
  frames.csv                                 every exported frame with provenance
  trajectories.csv                           every discovered trajectory, status, QC counts
  rejected_frames.csv                        every rejected frame and why
  split_manifest.json                        groups, strata, seed, method, split hash, leakage report
  manifest.json                              config, type_map, file hashes, dataset hash, split hash,
                                             InterfaceForge commit, per-backend consumption hints
```

Each extxyz frame carries `frame_id` (`<trajectory>:<source_frame>`), `IF_leaf`,
`source_path`, `source_frame`, `IF_stage`, `IF_temperature_k`, `IF_case`,
`IF_group`, `IF_formula`, the parsed chemistry (`IF_coverage_pct`,
`IF_ligand`, ...) and, when available, the instantaneous MD temperature and SCF
iteration count. The **split hash** is a location-independent sha256 over the
sorted `(frame_id, split)` pairs; the **dataset hash** covers every data file.
Both are recorded by `iface train nequip` and by the NequIP committee bundle.

The layout is the `iface collect` layout, so `iface train mace` (default
`datasets/canonical/*.extxyz`), `iface train deepmd` (default
`datasets/canonical/deepmd`), `iface train nequip` (`models.nequip.dataset`),
`iface mlip-compare`, and `iface package dataset-archive --dedupe` /
`iface package materialize` all consume it unchanged.

## Readiness audit

`iface dataset readiness` answers, from files on disk and before any GPU time:

1. which trajectories are discoverable (by stage / temperature / status);
2. usable frames per system and temperature;
3. incomplete or problematic runs and the rejection reasons, plus, per Step2
   trajectory, whether `step2_sample.json` was used and how many requested
   frames were present, rejected by QC, exported or missing;
4. frames, trajectories and groups per split versus the target ratios;
5. leakage checks;
6. with `--dataset` (and optionally `-c campaign.yaml`): whether the enabled
   MACE, DeePMD and NequIP inputs are byte-identical to the canonical files.

It writes `readiness.json` and `readiness.md` and reports
`ready_for_gpu_smoke_tests` with explicit blocking items. "Ready" means the data
plumbing is consistent; it says nothing about label quality or model accuracy.
