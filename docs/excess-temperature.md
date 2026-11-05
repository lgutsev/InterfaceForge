# Excess-temperature audit

`iface audit excess-temperature` checks that a bulk-referenced interface excess

```
gamma = [ sum_i c_i E_i  -  sum_j n_j e_j  -  dN mu ] / D
```

compares energies taken at one temperature, and that thermal-mean bulk
references are physically sensible. It reads one JSON/YAML file, does no
structure I/O and runs no engine.

```bash
iface audit excess-temperature --input gamma_input.json [-o OUTDIR] [--force]
    [--tol-ratio 0.10] [--tol-t 1] [--tol-gamma 0.01]
```

`iface audit <root>` (the VASP-MLFF audit) is unchanged; `excess-temperature` is
routed to its own parser before the main one, because the older command takes
a positional run root.

## Why

When `e_j` are thermal means from MD at `T_ref`, each frame `E_i` must come
from MD at `T_ref` too. Classically each atom holds (3/2) k_B T of potential
energy, so a frame at `T_i` adds

```
d_gamma = sum_i c_i * n_atoms_i * (3/2) k_B (T_i - T_ref) / D      [eV/Å² × 16.02176634 → J/m²]
```

For a ~300-atom cell in an 800 Å² denominator, a 150 K mismatch is ~0.1 J/m².

## Input

Schema: [`schemas/excess_temperature_input.schema.json`](../schemas/excess_temperature_input.schema.json).

| Field | Meaning |
|---|---|
| `interface_frames[]` | `id`, `family`, `n_atoms`, `T_K` (number or `null` when not recorded), `coeff` (c_i), optional `provenance` |
| `bulk_refs[]` | `name`, `e_per_fu_eV`, `atoms_per_fu`, `T_K`, `kind` (`thermal_mean` or `static`), `n_fu_coeff` (n_j), optional `provenance` |
| `static_refs[]` | optional `name`, `e_per_fu_eV` — the 0 K energy of the same bulk, for C1 |
| `denominator_A2` | D in Å², including any multiplier (2A for two interfaces per cell, sum of w_i·2A_i for a weighted combination) |
| `gamma_values[]` | optional per-frame `family`, `gamma_J_m2`, `frame_id`, for C3 |

Several independent excesses (e.g. one per balanced combination, each with its
own D) go in one file as `{"schema_version": 1, "cases": [ {...}, {...} ]}`,
each case with an optional `label`. The overall verdict is the worst case
verdict (INCONSISTENT > UNDETERMINED > CONSISTENT).

Leave `T_K` null when the records do not state the frame's MD temperature. A
temperature read off a directory or family *name* is not a measurement; record
it in `provenance` instead.

## Checks

- **C1 equipartition.** For each `thermal_mean` bulk with a `static_refs` entry
  of the same name: Δ = e_thermal − e_static, expected = (3/2) k_B T ·
  atoms_per_fu, ratio = Δ/expected; PASS when |ratio − 1| ≤ `--tol-ratio`.
  A missing static reference leaves C1 `INCOMPLETE`; `static` bulks are not
  checked.
- **C2 temperature match.** T_ref is the common bulk temperature; bulk
  references that disagree are an input error (exit 2). Each frame gets
  ΔT and its d_gamma contribution; the case gets the total, the partial sum
  over frames with known T, and `sensitivity_J_m2_per_K` — the bias per kelvin
  if every frame were offset by the same amount (useful when T is unknown).
- **C3 spread.** With ≥ 2 `gamma_values` in a family: mean, sample standard
  deviation and standard error of the mean.

**Verdict.** CONSISTENT when every |ΔT| ≤ `--tol-t` and C1 passes;
INCONSISTENT when |d_gamma| > `--tol-gamma` or C1 fails; otherwise UNDETERMINED,
with the reason (null T_K, missing static reference, or ΔT above tolerance with
a bias that happens to cancel). Defaults and their rationale are in
[Agentic assignments § thresholds](agentic-assignments.md#default-thresholds-and-why).

Constants: k_B = 8.617333262×10⁻⁵ eV/K (`interface_mu.K_B_EV_PER_K`),
1 eV/Å² = 16.02176634 J/m² (`validation.EV_A2_TO_J_M2`).

## Outputs

In the output directory (default: the input file's directory):

- `excess_temperature_manifest.json` — input path and sha256, tolerances,
  constants, verdict, sha256 of the three audit files;
- `excess_temperature_audit.json` — the full result;
- `excess_temperature_audit.tsv` — one row per frame per case;
- `excess_temperature_audit.md` — the human-readable summary.

Existing outputs are never overwritten without `--force`. Each run appends an
`excess_temperature_audit` event to `<output>/.interfaceforge/state.json`.
stdout carries one JSON summary; errors print `ERROR: …` to stderr and exit 2.
The exit code is 0 for every verdict.

## Worked example

TiN bulk, 300 K MD mean −20.245090 eV/f.u., static −20.324136 eV/f.u.,
2 atoms/f.u.:

```
Δ        = −20.245090 − (−20.324136)             = 79.046 meV/f.u.
expected = 1.5 × 8.617333262e-5 × 300 × 2         = 77.556 meV/f.u.
ratio    = 1.019                                    → C1 PASS
```

Two 300-atom frames with c = +1, one from 300 K and one from 450 K MD, against
the 300 K reference, D = 803.297 Å²:

```
d_gamma = 300 × 1.5 × k_B × 150 / 803.297 eV/Å² × 16.02176634 = 0.116 J/m²  → INCONSISTENT
```

```json
{
  "interface_frames": [
    {"id": "f300", "family": "ideal", "n_atoms": 300, "T_K": 300, "coeff": 1},
    {"id": "f450", "family": "ideal", "n_atoms": 300, "T_K": 450, "coeff": 1}
  ],
  "bulk_refs": [
    {"name": "TiN", "e_per_fu_eV": -20.245090, "atoms_per_fu": 2, "T_K": 300,
     "kind": "thermal_mean", "n_fu_coeff": 140}
  ],
  "static_refs": [{"name": "TiN", "e_per_fu_eV": -20.324136}],
  "denominator_A2": 803.297
}
```

Both numbers are regression tests in `tests/test_excess_temperature.py`.
