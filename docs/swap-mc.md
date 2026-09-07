# N/O swap Monte Carlo interface generator

> **Verification note:** automated-test only. The search, archive, selection,
> and export logic are regression tested with an analytic energy model. No real
> MACE/DeePMD search has been run through it, and no ordering prediction has been
> checked against DFT.

`iface swap-mc` searches for low-energy oxygen arrangements in an oxynitride
interface at **fixed oxygen content**: N and O are exchanged on an explicit set
of anion sites, each proposed arrangement is relaxed with a committee, and a
Metropolis walk over the relaxed structures keeps the search near low energy. It
produces a provenance-stamped archive of candidate interfaces and a diverse
shortlist for DFT verification — the input to a "random vs searched" ordering and
adhesion comparison.

## Method and attribution

The workflow follows **PAIPAI** — S. Zhu & R. Arróyave, *"Ground-State Structure
Search of Defective High-Entropy Alloys Using Machine-Learning Potentials and
Monte Carlo Sampling"*, Computational Materials Science **270** (2026) 114752;
<https://github.com/siyazhu/PAIPAI> (source reviewed at commit `00ff18b8`).

InterfaceForge adapts it:

| PAIPAI | InterfaceForge `swap-mc` |
|---|---|
| host swaps over any unlike pair in a group | N↔O only, on explicit anion sites; Si/Ti sublattice identities fixed |
| GRACE calculator | InterfaceForge MACE committee (mean PES) or a single DeePMD model |
| relaxed swap configurations | fixed cell; the interface's own `FixAtoms` constraints kept |
| fast worker (0.10 eV/Å, 30 steps) / slow worker (0.01 eV/Å, 400 steps) | `--screen-*` / `--refine-*`, same defaults; calibrate against your ordering-energy differences |
| `--temp` used as `exp(-ΔE/temp)` | `--kt-ev`, documented as an **eV energy scale, not Kelvin** |
| queued asynchronous low-energy search | independent seeded Metropolis walks |

**This is a low-energy structure search, not equilibrium sampling.** The archived
populations are biased toward low energy by the greedy screen relaxation and must
not be read as equilibrium site occupancies. Relative energies come from one MLIP
potential-energy surface; DFT-check the shortlist before any ordering claim.
These caveats are repeated in every `manifest.json` and `summary.md`.

## 1. Inspect the sites

```bash
iface swap-mc sites interface/SiN_TiN_O25.vasp \
  --cation Si --cation Ti --stacking-axis c \
  --interface-band 6 --move-classes layer
```

Prints the eligible anion indices, the move classes (layers by default, or
`region` / `none`), the nearest-cation slab label per site, the detected cation
contact plane, the fixed oxygen count, and any anions excluded because they are
frozen. `--z-window LO HI`, `--region {all,lower,upper,interface}`, and repeated
`--layer N` narrow the set further. Frozen substrate atoms are never eligible.

## 2. Run the search

Needs an importable committee environment (`mace-torch` or `deepmd-kit`), so run
it where `iface validate separation-energy` runs. Use `--dry-run` anywhere else
to resolve the sites and write `plan.json` without importing a backend.

```bash
iface swap-mc run interface/SiN_TiN_O25.vasp runs/order/O25 \
  --interface-band 6 --move-classes layer \
  --steps 2000 --kt-ev 0.05 --searches 3 \
  --screen-fmax 0.10 --screen-steps 30 \
  --refine-fmax 0.01 --refine-steps 400 --refine-accepted \
  --random-baseline 20 \
  --mace-model models/mace_committee/seed_11/…_stagetwo.model \
  --mace-model models/mace_committee/seed_23/…_stagetwo.model \
  --mace-model models/mace_committee/seed_37/…_stagetwo.model \
  --mace-model models/mace_committee/seed_53/…_stagetwo.model \
  --device cuda
```

In an isolated DeePMD environment where `iface` is not installed, the module has
a standalone entry point: `python -m interfaceforge.swap_mc run ...` with
`PYTHONPATH` pointed at the repository `src/` (same pattern as
`python -m interfaceforge.separation_energy`).

- `--start-mode keep` (default) starts each seed from the input arrangement;
  `randomize` scrambles it at the same count; `introduce --introduce-oxygen K`
  places K oxygen on an all-nitride eligible set.
- A MACE committee drives the walk with its **mean** PES; committee force spread
  is recorded at the final geometry of every refined/baseline candidate as an
  independent "needs DFT" flag. `--deepmd-model` takes a single model.
- `--random-baseline N` relaxes N random arrangements at the same composition and
  archives them tagged `random-baseline` for the paper comparison.

Output in `runs/order/O25/`:

| File | Contents |
|---|---|
| `manifest.json` | input hash, config, resolved sites, relaxer identity + model hashes, consensus best, per-seed best energies and pairwise Hamming distance, citation, caveats |
| `candidates.jsonl` | one row per distinct arrangement: substituent site list, screen/refine energy, convergence, committee σ, which seeds saw it |
| `candidates/cand_*.extxyz` | relaxed geometry for the lowest-energy `--geom-cap` (40) candidates plus every refined / baseline / initial one |
| `trajectory_seed_*.csv` | per-step proposed swap, ΔE, accept/reject, current and best energy |
| `summary.md` | the top table and the caveats |

## 3. Shortlist for DFT

```bash
iface swap-mc select runs/order/O25 -n 12 --low-fraction 0.6
```

Writes `dft_shortlist.{json,csv}`: `round(n·low_fraction)` lowest-energy
arrangements (the search's own favorable picks — the ones whose MLIP energy must
be confirmed), filled out by farthest-point picks in occupation-Hamming space
(a spread over the archive), the initial and random-baseline arrangements, and
the few highest committee-spread candidates. Only candidates with an archived
geometry are eligible; raise `--geom-cap` on the run if too few qualify.

## 4. Export to VASP

```bash
iface swap-mc export runs/order/O25 runs/order/O25_dft
```

Writes one directory per shortlisted candidate with a `POSCAR` (relaxed geometry,
frozen layers preserved) and `ordering.json`. From there:

- **DFT ordering benchmark:** `iface vasp opt-prepare` over the exported POSCARs
  (fixed cell, same frozen layers) and compare relative energies and rankings
  with the same relaxation convention.
- **Adhesion contrast:** `iface vasp adhesion prepare` per candidate, then
  `iface validate separation-energy` random vs searched at identical composition.

`--cand-id CAND_ID` (repeatable) exports specific candidates instead of the
shortlist.

## What to check before trusting a result

- Screen convergence rate (in `manifest.json` / `summary.md`): a low rate means
  the ΔE the walk accepted on are not converged — raise `--screen-steps` or
  loosen `--screen-fmax` and re-run.
- Seed agreement: if the seeds' best arrangements are far apart in Hamming
  distance, the search has not converged — add seeds or steps.
- DFT vs MLIP on the shortlist: the failure mode is the search finding an
  arrangement that looks favorable only because the model underestimates its
  energy. A good aggregate test RMSE does not cover this.
- `--kt-ev` is an energy scale in eV. It sets how far above the current energy a
  swap can be accepted; it is not a temperature.
