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

### Optional, tightens the N-poor bound

| Phase | Structure | Space group | MP ID | Role |
|---|---|---|---|---|
| TiSi₂ | C54 | Fddd (70) | [mp-2582](https://legacy.materialsproject.org/materials/mp-2582/) | Hao et al.'s N‑poor μ_Si bound |

### Needed for the oxide / N-O ordering work (paper 1, μ_O)

The same machinery runs with `--anion O`. For TiOₓN_y you will need:

| Phase | Structure | Space group | MP ID |
|---|---|---|---|
| **TiO₂ rutile** | rutile | P4₂/mnm (136) | [mp-2657](https://legacy.materialsproject.org/materials/mp-2657/) |
| TiO₂ anatase | anatase | I4₁/amd (141) | [mp-390](https://legacy.materialsproject.org/materials/mp-390/) |
| **TiO** | rocksalt | Fm‑3m (225) | [mp-2664](https://legacy.materialsproject.org/materials/mp-2664/) |
| Ti₂O₃ | corundum | R‑3c (167) | [mp-458](https://legacy.materialsproject.org/materials/mp-458/) |
| **SiO₂ α‑quartz** | α‑quartz | P3₁21 (152) | [mp-7000](https://legacy.materialsproject.org/materials/mp-7000/) |
| **O₂** | isolated molecule | — | *none — compute it* |

**O₂ must be spin-polarised** (`ISPIN=2`, triplet ground state) in a ≥ 12 Å box.
A non-spin-polarised O₂ is wrong by >1 eV and will corrupt the whole μ_O window.

Bold rows are the minimum set. TiO₂ rutile is the stable Ti oxide and normally
the binding bound; TiO and Ti₂O₃ matter only if your TiOₓN_y is reduced enough
that they become the competing phase.

### Already available

Your campaign's `bulk/TiN-Bulk*`, `bulk/SiN-Bulk*` and `bulk/TiO-Bulk*` MD
trajectories are **not** substitutes: those are finite-temperature MD snapshots
at a fixed cell. γ(Δμ) needs relaxed 0 K reference energies. Use the MD ones
with `iface validate interface-energy` (which MD-averages consistently on both
sides) instead.
