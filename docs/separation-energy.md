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

### Isolated MACE and DeePMD environments

Do not load MACE's conda environment and LONI's compiled DeePMD module into the
same Python process. Run the same structures once per backend and merge their
small JSON results afterward. The merge process imports neither MACE nor
DeePMD and verifies that the interface specs, directories, atom counts, areas,
reference convention, and completed DFT energies agree.

MACE job (inside `/project/lgutsev/env/mace_env`):

```bash
iface validate separation-energy audit/separation/stages/mace \
  "interface/300K/MD_Vac/N_Term/SiN_TiN_N-term=adhesion/N_term_dft" \
  "interface/300K/MD_Vac/Ti_Term/SiN-TiN-Ti-term=adhesion/Ti_term_dft" \
  --mace-model models/mace_committee/seed_11/…_stagetwo.model \
  --mace-model models/mace_committee/seed_23/…_stagetwo.model \
  --mace-model models/mace_committee/seed_37/…_stagetwo.model \
  --mace-model models/mace_committee/seed_53/…_stagetwo.model \
  --device cuda --json-only -c campaign.yaml
```

DeePMD job (after loading `deepmd-kit/r9.3-deepmd3.2.0.b.0-gpu` in a clean
shell):

```bash
# Point only at the repository's pure-Python source; do not add another env's
# complete site-packages directory to PYTHONPATH.
export PYTHONPATH=/absolute/path/to/InterfaceForge/src${PYTHONPATH:+:$PYTHONPATH}

python -m interfaceforge.separation_energy audit/separation/stages/deepmd \
  "interface/300K/MD_Vac/N_Term/SiN_TiN_N-term=adhesion/N_term_dft" \
  "interface/300K/MD_Vac/Ti_Term/SiN-TiN-Ti-term=adhesion/Ti_term_dft" \
  --deepmd-model models/deepmd/dpa2/model_000/frozen_model.pth \
  --deepmd-model models/deepmd/dpa2/model_001/frozen_model.pth \
  --deepmd-model models/deepmd/dpa2/model_002/frozen_model.pth \
  --deepmd-model models/deepmd/dpa2/model_003/frozen_model.pth \
  --json-only -c campaign.yaml
```

For a PyTorch member whose export failed, `model.ckpt.pt` can be supplied in
place of `frozen_model.pth`; DeePMD's inference backend loads both formats. The
LONI launcher prefers the frozen artifact and falls back to the checkpoint with
a warning. This is appropriate for the ASE comparison, but a checkpoint is not
a substitute for validating a frozen model in a downstream deployment engine.

To retry the four DPA-2 exports as an idempotent Slurm array (without
overwriting exports that already exist), run from the campaign root:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/freeze_missing_deepmd_dpa2.sbatch
```

Use `python -m interfaceforge.separation_energy` in the DeePMD job so the
interpreter supplied by the module imports the small evaluator directly. The
explicit repository `src/` path exposes InterfaceForge without exposing a
second environment's NumPy/PyTorch/CUDA packages. A useful job preflight is:

```bash
command -v python
python -c "import interfaceforge, deepmd; print(interfaceforge.__file__, deepmd.__file__)"
```

Finally, in the lightweight InterfaceForge development environment:

```bash
iface validate separation-energy audit/separation \
  --merge-json audit/separation/stages/mace/separation_energy.json \
  --merge-json audit/separation/stages/deepmd/separation_energy.json \
  -c campaign.yaml
```

This writes the usual single `separation_energy.{json,csv,md,png,svg,pdf}` set.
The two GPU jobs can run concurrently; submit the merge job with Slurm
`afterok` dependencies if the whole workflow should complete automatically.

### Bundled LONI submitter

From the campaign root, run the preflight and then submit:

```bash
bash /path/to/InterfaceForge/launch_scripts/submit_separation_energy.sh --dry-run
bash /path/to/InterfaceForge/launch_scripts/submit_separation_energy.sh
```

The wrapper discovers the current ENCUT-tagged MACE layout and the legacy
layout, or accepts `MACE_COMMITTEE_ROOT` as either the seed directory's parent
or the directory containing `mace_committee/`. Preflight prints the selected
models and rejects missing files before any submission. It cannot verify
training completion or runtime loading on the login node.

Each submission writes model lists and scheduler logs to a new
`audit/separation/runs/run.XXXXXXXX/` and passes that directory to all three
jobs. Final reports are written there too, so retries cannot mix backend
partials from different submissions. If a later submission fails, inspect
`jobs.tsv` and the raw scheduler logs before retrying. See
[launcher details](../launch_scripts/README.md#sintin-separation-energy-comparison)
for discovery rules and overrides. Direct CLI calls retain their explicitly
chosen output directories.

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
