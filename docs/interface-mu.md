# Grand-canonical interfacial energy γ(Δμ) — the vacuum-free branch

`iface validate interface-mu` computes the interfacial energy of a **periodic,
vacuum-free** interface cell referenced to bulk phases plus an anion reservoir:

```
γ(Δμ_X) = [ E_int − Σ_C x_C·g_C − Δn·μ_X⁰ ] / (n_interfaces · A)  −  Δn·Δμ_X / (n_interfaces · A)

x_C = n_cation(C) / a_C        formula units of compound C in the cell
Δn  = n_X − Σ_C x_C·b_C        anion excess
```

**The slope in Δμ is exactly the stoichiometric imbalance.** A compensated cell
has `Δn = 0`, zero slope, and a single chemical-potential-independent γ. A polar
cell with `Δn ≠ 0` is a straight line across the window, and where two
terminations cross, the stable termination changes.

This is the treatment of **Hao, Delley, Veprek & Stampfl, PRL 97, 086102 (2006)**
for this exact system (TiN(111)/Si₃N₄), with the μ ↔ (T, p) mapping of
**Reuter & Scheffler, PRB 65, 035406 (2002)**.

## The two branches — do not mix them

InterfaceForge splits every property tool into two regimes. The split is
enforced: each command measures the periodic vacuum gap of its structures and
refuses one from the other branch (threshold 5 Å, `regime.VACUUM_THRESHOLD_A`).

| | **`bulk`** (vacuum-free) | **`free-surface`** (vacuum) |
|---|---|---|
| commands | `interface-mu`, `interface-energy` | `separation-energy`, `adhesion`, `vasp adhesion *` |
| structures | periodic cells, bulk phases | slabs, cleaved half-slabs |
| quantities | γ(Δμ), γ_int | work of adhesion / separation, surface energy |
| MLIP validity | in-distribution for a bulk + bonded-interface committee | **needs isolated surfaces in training** |

Why this matters, measured on this campaign: the MACE and DPA-2 committees
reproduce the DFT vacuum-free excess to **±0.005 J/m²**, and are off by
**−2 to −2.9 J/m²** on the cleaved work of separation — with a *small* committee
spread in both cases. A bulk-trained model gives no warning that it has left its
domain. Every payload now carries `regime` and `mlip_validity_domain`, and
`--reference bulk` has been removed from `separation-energy` (it is subsumed
here: a stoichiometric cell gives the same single number).

## Running

```bash
iface validate interface-mu audit/interface_mu \
  "N-term=MD_Period/N_term_static" \
  "Ti-term=MD_Period/Ti_term_static" \
  --phase TiN=bulk/TiN --phase Si3N4=bulk/Si3N4 \
  --phase N2=bulk/N2 --phase Ti=bulk/Ti_hcp --phase Si=bulk/Si_diamond \
  --anion N --n-interfaces 2 \
  --mace-model ... --deepmd-model ...
```

Each `--phase NAME=DIR` is a finished VASP run. The tool classifies them:
one **compound per cation** (TiN, Si₃N₄) sets the reference energies; the
**elemental anion molecule** (N₂) sets μ_X⁰ = E(N₂)/2 and the anion-rich limit;
one **elemental cation per compound** (Ti, Si) bounds the anion-poor limit via
ΔH_f(C)/b_C. The binding compound is reported.

Outputs: `interface_mu.{json,csv,md,png,svg,pdf}` — a γ vs Δμ_X plot, one line
per interface, DFT solid with the MLIP committee spread as a band.

**MLIPs are evaluated only on the periodic interface cells and the solid
compound references.** The molecular anion reference is a vacuum structure and
is never handed to a bulk-trained model; it enters the DFT and MLIP γ as the
same constant and cancels in `γ^MLIP − γ^DFT`. The slope is structural, so it is
identical for DFT and every MLIP — only the intercept moves.

`--allow-vacuum` skips the regime guard. Use it only knowing the result then
folds two free-surface energies into γ and is not comparable to a periodic one.

## Phase diagrams (pymatgen)

The Δμ window is bounded by whatever phase decomposes first. The pairwise bound
`Δμ ≥ max_C ΔH_f(C)/b_C` only asks about each cation's *elemental* phase — right
for a binary, but in Ti–Si–N the window can instead be cut by a silicide or a
ternary. `--window hull` (the **default**) builds the convex hull over every
`--phase` you supply and intersects the stability ranges of the compounds the
interface is actually made of, naming the phase that binds each side.

```bash
# which phases exist in the system (MP IDs; energies are NOT taken from MP)
iface phases suggest Ti Si N

# the hull and the window, from your own runs
iface phases hull --anion N --compound TiN --compound Si3N4 \
  --phase TiN=bulk/TiN --phase Si3N4=bulk/Si3N4 \
  --phase N2=bulk/N2 --phase Ti=bulk/Ti_hcp --phase Si=bulk/Si_diamond \
  --phase Ti2N=bulk/Ti2N --phase TiSi2=bulk/TiSi2 --phase Ti5Si3=bulk/Ti5Si3
```

`phases hull` puts every `--phase` on the hull and takes the interface's own
constituents separately as `--compound`, so a competing phase is just another
`--phase`. `interface-mu` infers the constituents instead, so there it needs
`--aux-phase` to tell a competing phase apart from a reference — see
[`--phase` vs `--aux-phase`](#--phase-vs---aux-phase). Either way the competing
phases go on the hull as **auxiliary constraints**, can tighten the window, and
appear by name under `competing_stable_phases`.

Two hard rules the tooling enforces:

- **Every element needs an elemental reference.** Without it the chemical-potential
  scale for that element is undefined, and the hull refuses to build.
- **A reference phase above the hull is rejected** as a reservoir — it would
  decompose. `e_above_hull_ev_per_atom` is reported per phase, so a bad structure
  or a settings mismatch shows up immediately.

**Never mix Materials Project energies into your hull.** MP uses different
cutoffs, no dispersion, and fitted anion corrections; a hull built from a mix of
MP and your PBE+IVDW numbers is meaningless. `iface phases suggest` returns MP
IDs so you know *which* phases to compute — recompute all of them yourself.
It uses `mp_api` for a live query when installed (`pip install
'interfaceforge[phases-mp]'` + `MP_API_KEY`), and otherwise returns the verified
built-in list for this chemical system.

Sanity check worth doing once: run both `--window hull` and `--window pairwise`.
On a system with no competing ternary they agree exactly; if they differ, the
hull is right and the difference tells you which phase you had been ignoring.

## Bulk phases to calculate

Same settings as the interfaces (520 eV, `IVDW=11`, consistent k-point density,
same functional). Relax each fully (`ISIF=3`) — these are reservoir references,
not strained blocks. Materials Project IDs are given so there is no polymorph
ambiguity.

### Required now, for γ(Δμ_N)

| Phase | Structure | Space group | MP ID | Role |
|---|---|---|---|---|
| **TiN** | rocksalt (B1) | Fm‑3m (225) | [mp-492](https://legacy.materialsproject.org/materials/mp-492/) | compound reference, Ti side |
| **β‑Si₃N₄** | β nitride | P6₃/m (176) | [mp-988](https://legacy.materialsproject.org/materials/mp-988/) | compound reference, Si side |
| **N₂** | isolated molecule | — | *none — see below* | μ_N⁰, N‑rich limit |
| **Ti** | hcp (α‑Ti) | P6₃/mmc (194) | [mp-46](https://legacy.materialsproject.org/materials/mp-46/) | ΔH_f(TiN) → N‑poor bound |
| **Si** | diamond | Fd‑3m (227) | [mp-149](https://legacy.materialsproject.org/materials/mp-149/) | ΔH_f(Si₃N₄) → N‑poor bound |

> ⚠ **Ti hcp is `mp-46`, not `mp-72`.** `mp-72` is ω‑Ti (P6/mmm, 191), a
> different polymorph — using it silently shifts the N‑poor bound.

**N₂ is not an MP bulk entry.** Compute it yourself: one N₂ molecule in a
≥ 12 Å cubic box, Γ‑point only, same ENCUT. `μ_N⁰ = E(N₂)/2`. (Keep the usual
caveat that PBE overbinds N₂ by ~0.5 eV; if you want the window pinned to
experiment, substitute the measured ΔH_f instead.)

### Competing phases, which tighten the N-poor bound

None of these is a phase the interface is *made of* — each is a phase it could
decompose *toward*, so they enter the hull through `--aux-phase` (see below) and
can only ever raise the N‑poor bound.

| Phase | Structure | Space group | MP ID | Role |
|---|---|---|---|---|
| **Ti₂N** | ε‑Ti₂N, tetragonal | P4₂/mnm (136) | [mp-8282](https://legacy.materialsproject.org/materials/mp-8282/) | sits between Ti and TiN: TiN plausibly decomposes to Ti₂N before elemental Ti |
| **Ti₅Si₃** | hexagonal (Mn₅Si₃‑type, D8₈) | P6₃/mcm (193) | [mp-2108](https://legacy.materialsproject.org/materials/mp-2108/) | the most stable Ti silicide — TiSi₂ alone may not be the binding one |
| TiSi₂ | C54 | Fddd (70) | [mp-2582](https://legacy.materialsproject.org/materials/mp-2582/) | Hao et al.'s N‑poor μ_Si bound |
| TiSi | orthorhombic | Pnma (62) | [mp-7092](https://legacy.materialsproject.org/materials/mp-7092/) | Ti‑Si hull completeness |
| Ti₅Si₄ | tetragonal | P4₁2₁2 (92) | [mp-505527](https://legacy.materialsproject.org/materials/mp-505527/) | Ti‑Si hull completeness |
| Ti₃Si | tetragonal | P4₂/n (86) | [mp-980420](https://legacy.materialsproject.org/materials/mp-980420/) | Ti‑Si hull completeness |

> The error here is **one‑sided**: an omitted stable phase can only make the
> window look *too wide*, never too narrow. So this is safe to do incrementally
> — add a phase, re‑run `phases hull`, and either the window tightens or the
> phase comes out above the hull at your settings and nothing changes. Both
> `iface phases hull` and `iface validate interface-mu --window hull` report
> `missing_known_phases`: the phases of *your* chemical system that the hull was
> built without. A window reported alongside a non‑empty list is an upper limit,
> not the answer.

### The oxidation limit: μ_O for a phase that contains no O

Asking for the Δμ_O *window* of TiN is the wrong question, and pymatgen cannot
answer it — its range routine divides by the compound's amount of the open
element, so an O-free compound raises `ZeroDivisionError`. The right question for
the N/O ordering work is the **oxidation limit**: how O-rich the reservoir can get
before the nitrides stop being the stable phases at all.

`phases hull` detects this and switches method. The grand potential of an O-free
phase is flat in μ_O while every O-bearing competitor's falls, so above some μ_O
the compound is undercut and stays undercut — a monotone, one-sided bound, found
by bisecting a `GrandPotentialPhaseDiagram`. The lower side is genuinely
unbounded (`dmu_min_ev: null`): taking O away cannot destabilise a phase that
contains none.

```bash
iface phases hull --anion O --compound TiN --compound Si3N4 \
  --phase TiN=Wadh/TiN_mp492 --phase Si3N4=Wadh/Si3N4_mp988 \
  --phase TiO2=Wadh/TiO2_mp2657 --phase SiO2=Wadh/SiO2_mp7000 \
  --phase TiO=Wadh/TiO_mp2664 --phase Ti2O3=Wadh/Ti2O3_mp458 \
  --phase N2=Wadh/N2_gas --phase O2=Wadh/O2_gas \
  --phase Ti=Wadh/Ti_mp46 --phase Si=Wadh/Si_mp149
```

You get a limit per compound, the decomposition at that limit, and the binding
one:

```
"method": "grand-potential-open-element",
"dmu_max_ev": -5.25, "dmu_max_set_by": "TiN",
"per_compound": [
  {"compound": "TiN",   "dmu_limit_ev": -5.25, "decomposition_at_limit": ["N2", "TiO2"]},
  {"compound": "Si3N4", "dmu_limit_ev": -5.00, "decomposition_at_limit": ["N2", "SiO2"]}
]
```

Read that as: above Δμ_O = −5.25 eV, TiN gives way to TiO₂ + N₂, so **every
substitutional-O configuration in the ordering study has to sit below the tighter
of the two limits** — above it the interface is not an oxygen-doped nitride, it is
an oxide. The values above are the toy numbers the tests pin, not your DFT.

Two rules the dispatch enforces. A compound that is not stable even at
Δμ = −15 eV is refused: at that point the reservoir is as poor as it can
meaningfully get, so the problem is the phase, not the oxygen. And naming a
compound that contains the anion together with one that does not is refused
rather than half-answered — the first has a two-sided range, the second only an
upper limit, so they are separate runs.

### `--phase` vs `--aux-phase`

`--phase` is for the references the interface decomposes *into*: one compound per
cation (TiN, Si₃N₄), the elemental anion (N₂), one elemental cation each (Ti, Si).
`--aux-phase` is for everything that merely competes — a competing nitride, a
reduced oxide, a second polymorph. Auxiliary phases go on the convex hull and can
cut the window, but are never decomposed into.

The distinction is not cosmetic. Ti₂N contains the anion and one cation, so it
*looks* like a compound reference; passing it as `--phase` would demand that it
coexist with TiN and would count every Ti atom twice in the decomposition. That
is refused with an error pointing here, rather than silently producing a wrong
γ. The same applies to a second elemental reference (ω‑Ti next to hcp Ti).

```bash
iface validate interface-mu audit/mu \
  MD_Period_1=interfaces/MD_Period_1 \
  --phase TiN=phases/TiN_mp492 \
  --phase Si3N4=phases/Si3N4_mp988 \
  --phase N2=phases/N2_gas \
  --phase Ti=phases/Ti_mp46 \
  --phase Si=phases/Si_mp149 \
  --aux-phase Ti2N=phases/Ti2N_mp8282 \
  --aux-phase Ti5Si3=phases/Ti5Si3_mp2108 \
  --aux-phase TiSi2=phases/TiSi2_mp2582
```

### Needed for the oxide / N-O ordering work (paper 1, μ_O)

The same machinery runs with `--anion O`. For TiOₓN_y you will need:

| Phase | Structure | Space group | MP ID |
|---|---|---|---|
| **TiO₂ rutile** | rutile | P4₂/mnm (136) | [mp-2657](https://legacy.materialsproject.org/materials/mp-2657/) |
| TiO₂ anatase | anatase | I4₁/amd (141) | [mp-390](https://legacy.materialsproject.org/materials/mp-390/) |
| **TiO** | rocksalt | Fm‑3m (225) | [mp-2664](https://legacy.materialsproject.org/materials/mp-2664/) |
| Ti₂O₃ | corundum | R‑3c (167) | [mp-458](https://legacy.materialsproject.org/materials/mp-458/) |
| Ti₃O₅ | monoclinic (Magnéli n=3) | C2/m (12) | [mp-1147](https://legacy.materialsproject.org/materials/mp-1147/) |
| Ti₄O₇ | triclinic (Magnéli n=4) | P‑1 (2) | [mp-12205](https://legacy.materialsproject.org/materials/mp-12205/) |
| **SiO₂ α‑quartz** | α‑quartz | P3₁21 (152) | [mp-7000](https://legacy.materialsproject.org/materials/mp-7000/) |
| **O₂** | isolated molecule | — | *none — compute it* |

**O₂ must be spin-polarised** (`ISPIN=2`, triplet ground state) in a ≥ 12 Å box.
A non-spin-polarised O₂ is wrong by >1 eV and will corrupt the whole μ_O window.

**Every solid here runs `ISPIN=1`, including the reduced oxides.** Ti³⁺ is d¹, so
Ti₂O₃, Ti₃O₅ and Ti₄O₇ look like candidates for a moment — they are not. MP's
own workflow initialises MAGMOMs and runs `ISPIN=2`, and all three converge to
non-magnetic: 0.003, 0.001 and 0.010 μ_B respectively, ordering `NM`. The d
electrons spin-pair (Ti–Ti dimerisation along **c** in Ti₂O₃). So this is settled
by calculation, not assumption, and needs no rerun. The one thing that would
overturn it is **+U on Ti**, which localises the d electrons and can produce
moments where plain GGA gives none; MP applies no U to Ti–O, so if this campaign
does, re-check all three before trusting their hull positions.

O₂ is not a counterexample to any of that: MP has no isolated-molecule entry, so
its silence on O₂ says nothing. The triplet is real and you compute it yourself.

Bold rows are the minimum set. TiO₂ rutile is the stable Ti oxide and normally
the binding bound; anatase is a polymorph of the same composition, so pass it (if
at all) as `--aux-phase` — at your settings it will almost certainly come out
above the hull, which is correct physics, not an error. TiO and Ti₂O₃ matter only
if your TiOₓN_y is reduced enough that they become the competing phase; the same
goes for the Magnéli phases Ti₃O₅ and Ti₄O₇, which the completeness check now
asks for on any μ_O window. Ti₄O₇ is only 0.007 eV/atom
above MP's own hull (it decomposes to TiO₂ + Ti₃O₅ there), so at 520 eV with
`IVDW=11` it may land either side; `e_above_hull_ev_per_atom` decides, not the
built-in list. Ti₃O₅ has several C2/m entries on MP — `mp-1147` is the stable
one.

### The molecular references, in full

Both are single molecules in a large box, Γ-point only, at the campaign's ENCUT.
μ_X⁰ = E(X₂)/2.

```
# N2 -- closed-shell singlet
SYSTEM = N2 molecule
ENCUT  = 520
ISPIN  = 1          # or ISPIN=2; it must converge to 0 muB
ISMEAR = 0 ; SIGMA = 0.03
IBRION = 2 ; NSW = 60 ; ISIF = 2     # relax the bond, not the box
EDIFF  = 1E-7 ; EDIFFG = -1E-3
LREAL  = .FALSE.
```

```
# O2 -- triplet ground state; this is the one that goes wrong silently
SYSTEM  = O2 molecule
ENCUT   = 520
ISPIN   = 2
MAGMOM  = 2*1.0
NUPDOWN = 2         # pins S=1: without it the SCF can fall into the singlet
ISYM    = 0
ISMEAR  = 0 ; SIGMA = 0.03
IBRION  = 2 ; NSW = 60 ; ISIF = 2
EDIFF   = 1E-7 ; EDIFFG = -1E-3
LREAL   = .FALSE.
```

KPOINTS is Γ only for both, and the box must be ≥ 12 Å in every direction (a
molecule is charge-neutral and non-polar, so no dipole correction is needed;
`IDIPOL`/`LDIPOL` are not required).

`interface-mu` checks this for you rather than trusting it. The total cell moment
is read from the last SCF step of the reference's OUTCAR and compared with the
molecule's ground-state multiplicity: an O₂ that is unpolarised, or below half
the expected 2 μ_B, is **refused** — with `--allow-spin-mismatch` to override,
which then stamps the warning across the report instead. An N₂ carrying a moment
is flagged as `CHECK`. The measured value appears in the report and under
`reference_phases.<name>.spin`, so the old manual check

```bash
grep mag OUTCAR | tail -1        # O2 must show 2.00, N2 must show 0
```

is no longer the thing standing between you and a 1 eV error in every γ.

### Already available

Your campaign's `bulk/TiN-Bulk*`, `bulk/SiN-Bulk*` and `bulk/TiO-Bulk*` MD
trajectories are **not** substitutes: those are finite-temperature MD snapshots
at a fixed cell. γ(Δμ) needs relaxed 0 K reference energies. Use the MD ones
with `iface validate interface-energy` (which MD-averages consistently on both
sides) instead.
