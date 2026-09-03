# Separation energy (DFT vs MLIP)

`iface validate separation-energy` computes, for one or more hand-built
interface structures,

```
gamma_sep = ( E(slab_a) + E(slab_b) - E(interface) ) / (n_interfaces * A)
```

in J/m². With `--reference free-surface` (the default: `slab_a`/`slab_b` are the
two half-slabs relaxed against their own vacuum surfaces) and `n_interfaces = 1`
this is exactly the **Dupré work of adhesion** — the quantity Sharifi et al.
(2026) report.

It is deliberately separate from `iface validate interface-energy`:

| | `interface-energy` | `separation-energy` |
|---|---|---|
| structures | the synchronized MD dataset | a few hand-built slabs |
| reference | bulk phases | the two isolated slabs |
| DFT vs MLIP | committee on test frames | committee **evaluated in place** |
| headline number | γ_int(T) | **γ_sep^MLIP − γ_sep^DFT** on identical geometry |

## Layout

Two layouts are accepted per interface:

**A — an `iface vasp adhesion prepare` tree (recommended for a polar interface).**
Cleave the interface itself; the half-slabs inherit whatever faces the cut
produces, so you never need a standalone (and, for TiN(111), ill-defined) surface
energy:

```bash
iface vasp adhesion prepare MD_Vac/Step2_Interface/SiN_TiN_N-term_450 \
  --method dft --slab-mode static --interface-sp \
  --output-dir adhesion/N_term_dft
```

`separation-energy` then reads `interface_static/`, `slabs/lower/`, `slabs/upper/`
from that tree's `manifest.json`. With `--slab-mode static` the quantity is the
ideal **work of separation** (frozen slabs); the MLIP evaluates the *identical*
frozen geometries.

**B — a plain directory** with `interface/ slab_a/ slab_b/` sub-directories, each a
VASP run directory. Use this when you have genuine relaxed free-surface half-slabs
(then γ_sep = the Dupré work of adhesion).

Run the sub-directories (or the adhesion tree's `slabs/*` + `interface_static/`)
as static VASP calculations for the DFT side.

## Running

```bash
iface validate separation-energy audit/separation \
  "interface/450K/Real/N_Term/SiN_TiN_N-term=adhesion/N_term_dft" \
  "interface/450K/Real/Ti_Term/SiN-TiN-Ti-term=adhesion/Ti_term_dft" \
  --mace-model models/mace_committee/seed_11/…_stagetwo.model \
  --mace-model models/mace_committee/seed_23/…_stagetwo.model \
  --mace-model models/mace_committee/seed_37/…_stagetwo.model \
  --mace-model models/mace_committee/seed_53/…_stagetwo.model \
  --deepmd-model models/deepmd/dpa2/model_000/frozen_model.pth \
  --deepmd-model models/deepmd/dpa2/model_001/frozen_model.pth \
  -c campaign.yaml
```

- **DFT** energies are read from each run's `OUTCAR` (`energy(sigma->0)` of the
  last ionic step) — for an adhesion tree that is `interface_static/OUTCAR` plus
  the two `slabs/*/OUTCAR`. An interface whose three runs have not all finished
  is reported but carries no `gamma_sep`.
- **`--mace-model` / `--deepmd-model`** (repeat for the committee) are evaluated
  on the same structures via ASE, in the current environment — run this where
  the committee's `mace-torch` / `deepmd-kit` is importable (i.e. on the
  cluster). The MLIP reads each run's `CONTCAR` if present, else `POSCAR`, so it
  always evaluates the *DFT* geometry — no independent relaxation. Each family
  reports its per-member γ_sep, the ensemble mean, the committee spread, and
  `Δ = γ_sep^ensemble − γ_sep^DFT`.
- The `LABEL=` prefix is fnmatched against `validation.interfaces`; its
  `orientation`/`termination` select which `validation.references`
  (`quantity: work_of_adhesion`) value to overlay, for both the DFT and the
  MLIP numbers.

## Output

`separation_energy.{json,csv,md}` plus a two-panel figure
(`separation_energy.{png,svg,pdf}`, same style as `iface mlip-compare`'s
`publication_rmse_summary`):

- **(a)** γ_sep per interface — DFT, each MLIP committee (± spread), and the
  literature diamond;
- **(b)** γ_sep^MLIP − γ_sep^DFT per interface, against a zero line.

The CSV has one row per (interface, source) with the literature Δ and
within-tolerance flag.

## Caveats

- `--reference free-surface` assumes the two `slab_a`/`slab_b` structures really
  are relaxed free-surface slabs. If they are frozen at the interface geometry
  the result is the ideal work of separation instead (an upper bound). The
  `reference` field records which you declared; it does not check.
- An MLIP trained mostly on the bonded interface can extrapolate poorly for an
  isolated, vacuum-exposed slab. A committee spread that blows up on `slab_a` or
  `slab_b` relative to `interface` is the tell.
- `--reference bulk` is accepted and recorded but the bulk-referenced scaling
  (formula-unit matching, `n_interfaces = 2`) is not yet implemented — use
  `iface validate interface-energy` for the bulk-referenced quantity.
