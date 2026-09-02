# Bulk-referenced interfacial energy

`iface validate interface-energy` computes, for each unoxidized interface leaf in
the synchronized canonical dataset,

```
gamma_int(T) = ( <E_int>_T - x*<E/fu(TiN)>_T - y*<E/fu(SiN)>_T ) / (n_interfaces * A)
```

converted to J/m² (× 16.02176634). It runs entirely off the dataset — no extra
VASP or MLIP jobs.

```bash
iface validate interface-energy audit/interface_energy
```

Writes `interface_energy.{json,csv,md}` to that directory. It reads
`validation.interfaces` from the campaign for the per-interface stacking axis,
`n_interfaces`, orientation/termination labels, and the `polar_termination`
skip (see below).

## Polar-terminated interfaces are skipped

A `validation.interfaces` entry with `polar_termination: true` is excluded from
the γ_int table and listed under `skipped` in the report. A TiN(111) or
Si₃N₄(0001) slab stacks as alternating pure-element planes, so it is never an
integer number of bulk formula units: matching the interface's Ti and Si counts
to bulk TiN and Si₃N₄ leaves an unbalanced N (or Ti) excess that gets valued at
the full bulk per-formula-unit energy, and the result is off by one to two
orders of magnitude. The report's `nitrogen_balanced: false` flag catches this
even without the metadata, but the flag still emits the (meaningless) number;
`polar_termination` stops it at the source.

For those interfaces use `iface validate adhesion` — the work of adhesion
references the two relaxed slabs directly, so their non-stoichiometry cancels.
A grand-canonical γ(μ_N, μ_Ti) treatment that would give a meaningful band for
polar and oxidized interfaces is a planned follow-up.

## Method

- **Energies** are MD averages of the DFT `REF_energy` over post-equilibration
  frames, pooled across the train/valid/test splits per leaf and filtered by the
  original `source_frame` index (`--equilibration-frames`, default 100).
- **Formula-unit counts** `x`, `y` come from matching the interface composition
  against each bulk reference's *own* reduced formula (via the gcd of its atom
  counts), so a 1:1 nitride and Si₃N₄ are both handled. The predicted vs actual
  nitrogen count is reported and flagged if it does not balance.
- **Bulk references** are the same-temperature `bulk/TiN-Bulk_<T>K` and
  `bulk/SiN-Bulk_<T>K` leaves. This is only valid because every leaf in the
  dataset shares an identical INCAR apart from `TEBEG`/`TEEND` (same ENCUT,
  `IVDW`, POTCAR, k-density).
- **Area** is `|a × b|` for the two lattice vectors perpendicular to the
  stacking axis. The stacking axis comes from the matching
  `validation.interfaces` entry (`stacking_axis:`), else `--stacking-axis`,
  else the longest cell vector.
- **`n_interfaces`** comes from the matching `validation.interfaces` entry
  (`n_interfaces:`), else `--n-interfaces`, else 2 (a coherent periodic A/B/A
  stack).
- **Error bars** are block-average standard errors (`--blocks`, default 10)
  propagated through the excess and the division.

## What this is and is not

`gamma_int` here is an **approximation to the interface free energy**: it is the
excess *potential* energy over the bulk phases. The vibrational-entropy term
`-T·S_vib` is dropped; it largely cancels in an excess quantity when the
interface and bulk bonding environments are similar, but the result is not a
rigorous free energy.

It is also **not** the vacuum work of adhesion — `iface validate adhesion`
computes that separately by cleaving the interface into two slabs.

## MLIP-committee mode

```bash
iface validate interface-energy audit/interface_energy --predictions audit/mlip_compare
```

`--predictions` points at an `audit/mlip_compare*` directory. For each interface
the report then also carries an `mlip` block: the MACE committee's `gamma_int`
per member and for the ensemble mean, the DFT `gamma_int` recomputed on the
**exact same frames**, and `Δ = γ_MLIP − γ_DFT`. This is a much stronger check
than absolute-energy RMSE — the model has to get the interface-minus-bulk
*excess* right.

Because it reuses the `mlip-compare` prediction files, it is limited to the
test-split frames (60/leaf), so its error bars are wider than the default
DFT-on-all-600 value; `gamma_dft_same_frames` is the fair comparison target.

Oxidized interfaces (`O_x*`) are excluded: their excess oxygen has no bulk phase
to absorb it, so `gamma_int` becomes a function of the oxygen chemical potential
(a band between the TiO-referenced and O₂-referenced limits) rather than a
single number. That treatment is a planned follow-up.
