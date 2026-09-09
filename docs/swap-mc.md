# N/O swap Monte Carlo interface generator

> **Verification note:** automated-test only. The search, archive, selection,
> export, and the DFT benchmark preparation, launch, collection and comparison
> are regression tested with an analytic energy model and synthetic VASP output.
> No real MACE/DeePMD search has been run through it, and no ordering prediction
> has yet been checked against real DFT.

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

For DeePMD, InterfaceForge requests the native neighbor-list backend explicitly.
This avoids a TorchScript deserialization failure in the optional vesin path of
the LONI DeePMD 3.2.0b0 module, at the cost of slower neighbor-list construction.
MACE remains the preferred driving backend for long swap-MC searches.

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
frozen layers preserved) and `ordering.json` (role, archive roles, MLIP energies,
committee spread). `--cand-id CAND_ID` (repeatable) exports specific candidates
instead of the shortlist.

## 5. Prepare the DFT benchmark

```bash
iface swap-mc dft-prepare runs/order/O25_dft runs/order/O25_vasp \
  --reference /path/to/converged/interface/run \
  --stage static --stage relax
```

The `--reference` directory is an existing converged run for the same interface.
Its INCAR supplies every electronic setting (ENCUT, PREC, ISPIN, LDAU, smearing,
parallelization), its KPOINTS is copied verbatim, and its POTCAR is reused when
the species order matches. Nothing in it is modified.

**What this guarantees, and why it is the whole point.** Ordering energies are
differences of tens of meV between structures with the same atoms in the same
box. They are meaningful only if the candidates differ in *nothing else*, so
`dft-prepare` enforces that rather than trusting it:

| Check | Behaviour |
|---|---|
| composition | identical across candidates, or refuse |
| cell | identical within 1e-4 A, or refuse |
| frozen layers | same selective-dynamics count per species, or refuse |
| species blocks | every POSCAR rewritten into one canonical (alphabetical) order so a single POTCAR is valid everywhere; selective-dynamics flags travel with their atom |
| INCAR | byte-identical apart from `SYSTEM`; verified by hashing the body, and the hash recorded in the manifest |
| KPOINTS / POTCAR | one shared file, hard-linked into every run |
| `ISYM` | forced to 0. Different N/O arrangements have different symmetry, and symmetry-reduced k-point sets would make their energies inconsistent at exactly the meV scale being measured |
| ML tags | removed; this is a DFT benchmark |

Two stages are prepared, under `static/` and `relax/`:

- **`static`** (`IBRION=-1`, `NSW=0`) scores the MLIP geometries as they are. It
  separates the energy *ranking* from any geometry error and is the cheap first
  test. Run it on the whole shortlist.
- **`relax`** (`IBRION=2`, `ISIF=2`, fixed cell, same frozen layers) re-relaxes
  each arrangement, which is the quantity the MLIP search itself approximated.
  Run it on the subset that matters.

Useful options: `--ediff`, `--ediffg`, `--nsw`, `--launcher NAME` (default
`runvasp.sh` then `run.slurm` from the reference), `--potcar-root` when a POTCAR
must be assembled, and `--mlip-energy {best,refine,screen}` to choose which
archived MLIP energy is carried into the comparison.

## 6. Launch

```bash
iface swap-mc dft-launch runs/order/O25_vasp --stage static
iface swap-mc dft-launch runs/order/O25_vasp --stage static --execute
```

The dry run is the default. Preflight re-hashes every prepared input and refuses
to submit if one changed since `dft-prepare`, because editing a single run's
INCAR silently breaks the comparison. A directory that already holds
`OUTCAR`/`OSZICAR` is reported as skipped, never resubmitted. `--limit N` bounds
the batch. Submission writes `ordering_dft_launch_<stage>.{json,tsv}`.

## 7. Collect and compare

```bash
iface swap-mc dft-collect runs/order/O25_vasp --stage static
iface swap-mc dft-compare runs/order/O25_vasp --stage static
```

`dft-collect` reads `energy(sigma->0)` from the last completed ionic step of each
run with the same parsing `iface audit` uses, and writes
`static/ordering_dft_energies.{json,csv}`. Unfinished runs are reported, not
dropped; in the relax stage a run that never reached `EDIFFG` is marked unusable.

`dft-compare` runs the collection itself, then compares both methods through
`dE_i = E_i - E_reference` evaluated on the *same* configuration, so the
arbitrary offset between a DFT total energy and an MLIP energy cancels exactly
and no fitted shift is applied. The reference is the initial arrangement when the
export contains it, else the lowest-MLIP random baseline, else the highest-DFT
candidate; `--reference-cand CAND_ID` overrides it, and `--per-atom` reports
meV/atom.

Outputs in `static/`: `ordering_comparison.{json,csv,md}` and a DFT-vs-MLIP
scatter `ordering_comparison.png` grouped by searched / random / initial.

The report answers three separate questions:

1. **Does the model rank these arrangements as DFT does?** Pearson r, Spearman
   rho, pairwise order agreement, and MAE/RMSE/max residual next to the DFT
   spread they have to be judged against. A max residual comparable to the whole
   DFT spread means this model does not resolve the ranking, and the report says
   so.
2. **Is the search's own best arrangement real?** Whether the MLIP minimum is
   also the DFT minimum, its DFT rank, and how far above the DFT best it sits.
3. **Is the headline claim true?** `searched_vs_random` gives the best searched
   arrangement minus the mean random arrangement under both methods, with the
   sign agreement. A disagreeing sign is the signature of a search exploiting
   model error: add those configurations to training and repeat.

## 8. The other compositions

Validate the workflow at one composition (O25) with the full DFT check, then run
the MLIP searches for the others. Keep a small DFT spot-check at each, two to
four runs covering the searched minimum and a random control, because agreement
at one oxygen content does not establish agreement at another, especially after
a search deliberately leaves the random-arrangement distribution the model was
trained near. Compositions whose eligible sites are all N or all O have no swaps
to make and serve as endpoint controls.

## What to check before trusting a result

- Screen convergence rate (in `manifest.json` / `summary.md`): a low rate means
  the ΔE the walk accepted on are not converged — raise `--screen-steps` or
  loosen `--screen-fmax` and re-run.
- Seed agreement: if the seeds' best arrangements are far apart in Hamming
  distance, the search has not converged — add seeds or steps.
- DFT vs MLIP on the shortlist: the failure mode is the search finding an
  arrangement that looks favorable only because the model underestimates its
  energy. A good aggregate test RMSE does not cover this. `dft-compare` reports
  the sign agreement of the searched-vs-random gain, which is the direct test.
- `--kt-ev` is an energy scale in eV. It sets how far above the current energy a
  swap can be accepted; it is not a temperature.
