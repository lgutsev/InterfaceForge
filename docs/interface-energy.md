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

Writes `interface_energy.{json,csv,md}` to that directory.

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
  stacking axis, taken as the longest cell vector unless `--stacking-axis` is
  given.
- **`n_interfaces`** defaults to 2 (a coherent periodic A/B/A stack).
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

Oxidized interfaces (`O_x*`) are excluded: their excess oxygen has no bulk phase
to absorb it, so `gamma_int` becomes a function of the oxygen chemical potential
(a band between the TiO-referenced and O₂-referenced limits) rather than a
single number. That treatment is a planned follow-up.
