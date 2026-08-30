# NiO(110) surface hydroxylation — training-data generator

One self-contained notebook that builds **partially hydroxylated NiO(110) slabs**
(and phosphonate-ligand-decorated variants) as input for VASP+U relaxation / AIMD.
This is a standalone application notebook — it is *not* part of the
InterfaceForge package and does not modify it. It deliberately does **not**
generate POTCAR files; use the project's normal POTCAR generator after export.

## Why

Every existing `NiO_m110_Big_U46*` interface directory uses a fully bare
Ni²⁺/O²⁻ termination. Real NiOₓ surfaces are partially hydroxylated (surface
`Ni–OH` from dissociative water), which is the mechanism by which phosphonic-acid
SAM anchors are believed to bind. This notebook generates the training data to
close that gap.

Out of scope: Ni-vacancy / Ni³⁺ / NiOOH non-stoichiometry, global (ice-rule)
proton ordering, and the separate AIMD sampling-policy change.

## The notebook

`NiO_m110_hydroxylation.ipynb`, run top to bottom. It is **self-contained** —
the helper functions are inlined in the §0 Toolkit cells.
`nio_hydroxylation_utils.py` is the *build source* for those cells. Edit it and
run `python sync_notebook_toolkit.py`; the notebook does not import it.


| § | what |
|---|---|
| 0 | Toolkit — inlined helper functions (I/O, site finding, motifs, H-bonds, freeze, phosphonate, export). Run once; not normally edited. |
| 1–2 | Load bare slab → exposed-Ni inventory (canonical, frozen order). |
| 3 | Coverage sweep: 0/25/50/75/100 % × {clustered, scattered} × {capped, dissoc}, with selective-dynamics freezing applied. |
| 4–5 | H-bond diagnostic + manual H-orientation override. |
| 6–7 | Visualise (incl. frozen/mobile split) + sanity checks. |
| 8 | Export the hydroxylation cases as runnable VASP folders + manifest. |
| 9–12 | Phosphonate ligand: anchor detection (P–OH vs P=O) → surface-docked orientation → placement at **pristine (0 % OH) / bare / OH-boundary** → export. |
| 13 | Batch loop over **surfaces × passivants × cases × anchor-positions**. |

## Setup

```bash
pip install -r requirements.txt
```

**Everything needed for generation is bundled in `inputs/`** (the bare
`CONTCAR`, four passivant `.xyz` files, and `inputs/vasp_template/` with the
working `INCAR` + `KPOINTS`). Add POTCAR afterwards with the existing project
generator. The sweep cell (§3) is meant to be re-run after you
tweak a parameter, the exposed-Ni inventory (§2), or an `H` orientation (§5).

Geometry note: minimum-image distances use fractional coords + the full cell
matrix, so a non-orthogonal (hexagonal) cell is handled correctly; periodicity
is in-plane only.

## Two hydroxylation motifs (both generated at every intermediate coverage)

- **capped** (`_capped`) — an exploratory OH-only fragment placed straight up
  (+z) from a selected Ni at a 2.0 Å placeholder `Ni–O` distance. It adds OH,
  not H2O, and must not be compared as a neutral water-dissociation state.
- **dissoc** (`_dissoc`) — the same `Ni–OH` **plus** a proton on the nearest
  distinct surface lattice O, including valid first-shell acceptors. A
  deterministic maximum matching assigns one lattice O per selected Ni. If a
  complete matching is impossible, generation stops rather than silently
  producing a mixed motif. Thus every `_dissoc` case contains exactly one H2O
  equivalent per selected Ni.

The generator does **not** presuppose which motif wins — the downstream VASP+U
stage decides per local environment.

## Hydrogen orientation

A candidate `O···O` H-bond graph (2.5–3.3 Å window) is built over all hydroxyl
oxygens; each hydroxyl's single `H` is assigned **greedily, shortest edge first**,
toward its best available acceptor (donate once, accept many). Isolated
hydroxyls get a modest outward tilt (≈25° from +z). This is a **local starting
guess, not a ground-state search** — VASP+U ionic relaxation refines proton
positions. Every pattern prints a diagnostic:
`satisfied near-linear O–H···O contacts / candidate pairs`, plus `H···H` clashes.
Section 5 of the notebook is a manual override table.

## Selective dynamics (partial freeze)

Configured in the sweep cell (§3):

The interface must be **anchored from the bottom** — every exported slab has a
frozen bottom region (`F F F`), everything above relaxes (`T T T`).

- `FREEZE_MODE = "auto"` *(default)* — keep the input POSCAR's `F/T` flags if it has
  any; otherwise freeze the bottom `FREEZE_VALUE` (1) layer. Never leaves a slab
  un-anchored. Added OH / ligand atoms stay mobile. Tiles correctly under
  `SURFACE_SUPERCELL`. The compromise slab freezes 40/200 atoms, exactly the
  same 20% bottom plane as the original 24/120 slab.
- `FREEZE_MODE = "inherit"` — keep the CONTCAR's flags; **error** if it has none.
- `FREEZE_MODE = "bottom"` + `FREEZE_SUBMODE` ∈ `{layers, thickness, fraction,
  count}` — always re-derive a bottom band.
- `FREEZE_MODE = "window"` + `FREEZE_LO/HI` — the direct equivalent of the
  `atom.x`-window freezer: fix every atom with `z < LO` or `z > HI` (set `HI`
  huge to freeze only the bottom; lower it to also pin the top). Use `FREEZE_AXIS`
  if the normal isn't `z`.
- `FREEZE_ONLY = ("Ni", "O")` — hydroxyl / adsorbate atoms never frozen.
- `FREEZE_MODE = None` — strip all constraints.

`provenance.json` records the frozen atom count + indices and the freeze
parameters; the manifest has an `n_frozen` column.

## Phosphonate ligand docking

- **Anchor** = a P with exactly 3 O (≤1.95 Å) and 1 C (≤2.10 Å). Each anchor O is
  classed **P–OH** (carries an H) or **P=O**. Bis-phosphonate (DCZ-4P) → the
  lowest-atom-index P; all P indices recorded in the manifest.
- **Orientation** (`orient_phosphonate`) — **surface → phosphonate head → body**:
  1. rotate so **P → (centroid of the body atoms) points up (+z)** — the whole
     body stands vertical, anchor O's below P so the head faces the surface;
  2. spin about the vertical through P so the **P=O faces the nearest surface
     hydroxyl H** in a boundary case;
  3. only if the upright molecule is taller than the available vacuum does it
     **auto-tilt about a horizontal axis through P** (realistic tilted-SAM
     geometry); `tilt_deg` in the manifest. In practice none of the four
     passivants need this — see the vacuum note below.
- **Placement** (`place_ligand`): the neutral molecule's terminal P=O oxygen is
  placed directly over the target exposed Ni at `OH_HEIGHT` (≈2.05 Å). This
  guarantees that the requested starting Ni–O distance is real rather than
  placing P over Ni and leaving every anchor O laterally displaced. Steric
  clashes are resolved by a full-azimuth/contact-tilt search about that fixed
  O; the final Ni–O distance is audited and never sacrificed to a vertical
  lift. Every candidate must also pass species-aware hard contact floors
  (`H–Ni ≥ 2.10 Å`, `O–Ni ≥ 1.75 Å`, `O–O ≥ 2.30 Å`, plus pair-specific
  floors for C/N/P). This prevents a protonated P–OH group from being forced
  into an artificial Ni–H or compressed second Ni–O bond. If no valid rigid
  orientation exists, generation stops and exports nothing for that case.
  The contact search permits up to a 40° rigid SAM tilt about the bound O,
  consistent with tilted phosphonate layers, but it never relaxes a chemical
  distance floor merely to make a case pass.
- **Vacuum / cell**: the slab sits in the *middle* of its cell, so the real
  headroom is the gap to the slab's **periodic image** (`slab_bottom + c`), not
  the distance to the `c` plane. `vacuum_fit` reports `vacuum_gap` (to the image);
  `ok` needs no overlaps *and* `fits_vacuum`.
- **`LIGAND_CELL_C`** (§9), applied to ligand cases only (bare / OH-only keep the
  CONTCAR cell):
  - `48.0` *(default)* — one uniform cell for all supplied passivants.
  - `"auto"` — keep the molecule upright, then lengthen `c` until the gap from
    the ligand top to the next slab image is at least `LIGAND_PERIODIC_GAP`
    (12 Å by default). The old implementation tilted before extending, which
    could create an unintended geometry.
  - `None` — never change `c`; the vacuum audit must still pass.
  `extend_vacuum` rescales only the `c` lattice vector; atom positions unchanged.

### Anchor positions

- **`bare`** — over an exposed Ni with no hydroxyl (acid condenses with bare Ni²⁺).
- **`boundary`** — bound to a bare exposed Ni that is the nearest neighbour of a
  hydroxylated Ni; P=O is oriented toward the adjacent `Ni–OH` H. It is not
  placed at the geometric midpoint, which produced a non-bonded starting state.

At 100% OH coverage there is no bare-Ni chemisorption site. Those structures
therefore use a separately labelled molecular `surface-O–H···O=P` mode: one
surface OH is directed outward as the donor and the neutral phosphonate P=O is
placed at a 1.80 Å H···O starting distance. The export audit requires
H···O = 1.45–2.20 Å, O–H···O ≥ 150°, and Ni···O ≥ 2.80 Å so a hydrogen-bonded
case cannot be misreported as Ni–O anchoring. Explicit deprotonated
mono-/bidentate products and proton-transfer/water-elimination states remain
separate chemical states rather than outcomes assumed during local relaxation.

### §13 batch

Set `RUN_BATCH = True`. Loops `SURFACES × BATCH_LIGANDS × BATCH_CASES ×
BATCH_POSITIONS`, re-deriving the exposed-Ni inventory and coverage pattern per
surface, docking each ligand, checking overlaps + vacuum, exporting, and writing
`generated/manifest_batch.csv`. The manifest and provenance include the
limiting chemical-contact pair and its margin above the applicable floor.
The default grid combines two explicit modes. `BATCH_CASES` covers bare-Ni
chemisorption: pristine plus clustered/scattered × capped/dissociated at 25%,
50%, and 75% OH. `BATCH_HBOND_CASES` covers molecular hydrogen-bond adsorption
on the capped and dissociated 100%-OH surfaces. With the four bundled ligands
this produces 108 ligand-decorated structures: 100 Ni–O cases plus eight
surface-OH···O=P cases.

## Naming

```
NiO_m110_Big_U46                                             bare (0 %)
NiO_m110_Big_U46_OH50_clustered_capped                       hydroxylated
NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary      + ligand
```

`OH<pct>` = coverage fraction of the exposed-Ni inventory; `<pattern-id>` ∈
{clustered, scattered, full}; `<motif>` ∈ {capped, dissoc}; ligand token has
hyphens stripped; default anchor position ∈ {bare, boundary, hbond}. `hbond`
always denotes the non-chemisorbed OH100 molecular state.

## Export — runnable VASP optimization folders

`RUN_TEMPLATE` (§1) points at the supplied working `INCAR` + `KPOINTS`
(optionally `runvasp.sh`). Every generated case is ready for the project's
POTCAR-generation/launch step and is grouped by OH type:

```
generated/
  OH0/    NiO_m110_Big_U46/                       bare pristine (0 %)
          NiO_m110_Big_U46_<ligand>/              ligand on pristine NiO
  OH25/   NiO_m110_Big_U46_OH25_clustered_capped/
          NiO_m110_Big_U46_OH25_clustered_dissoc/
          NiO_m110_Big_U46_OH25_scattered_capped/
          NiO_m110_Big_U46_OH25_scattered_dissoc/
  OH50/   ...  (+ the <case>_<ligand>_<bare|oh|boundary>/ folders from §12)
  OH75/   ...   OH100/  ...
```

One directory per coverage — no deeper nesting. The folder name carries the
pattern / motif / ligand, so it stays unambiguous. **§8 writes the hydroxylation
folders, §12 the ligand folders** (each prints a `... INCAR in N/N` line); §13 is
only the optional batch expansion.

Each leaf:

```
POSCAR          element-ordered (alphabetical = ASE sort=True) via a *stable*
                sort, so POSCAR / MAGMOM / LDAU* agree and an external POTCAR
                generator receives an unambiguous order. Ni-up stays before
                Ni-down. Selective-dynamics F/T.
KPOINTS         copied from RUN_TEMPLATE
INCAR           template copied, then rewritten for THIS case's species order --
                MAGMOM, LDAUL, LDAUU, LDAUJ are per POSCAR species so they change
                with the element set:
                   bare        Ni O          LDAUL = 2 -1
                   hydroxylated H Ni O        LDAUL = -1 2 -1
                   ligand      C H N Ni O P   LDAUL = -1 -1 -1 2 -1 -1
                Ni always L=2, U=4.6; SYSTEM = case name; ISIF=2 for every slab;
                decorated slabs receive a structure-specific DIPOL centre.
POTCAR          intentionally not written; generate it from the ordered POSCAR
                using the existing project workflow.
structure.xyz   convenience (carries the magnetic moments + freeze constraint)
provenance.json coverage / pattern / motif / sites / H-bond score / ligand info /
                freeze / supercell / magmom / input hashes / run_dir / run_inputs
```

The template `INCAR` is the **passivated-case** relaxation
(`Phosphonate on NiO(110) AFM-II` — PBE + Dudarev U(Ni 3d)=4.6 + D3 + dipole).

`manifest.csv` / `manifest_ligands.csv` carry a **`path`** column.

### Batch POTCAR, audit, and launch on LONI

Use `manifest_batch.csv` as the allowlist when the OH-free pilot runs are being
launched separately. From the InterfaceForge repository root:

```bash
iface vasp opt-prepare notebooks/nio_m110_hydroxylation/generated \
  --manifest notebooks/nio_m110_hydroxylation/generated/manifest_batch.csv \
  --exclude-prefix OH0 \
  --launcher-template launch_scripts/runvasp.sh \
  --dry-run

iface vasp opt-prepare notebooks/nio_m110_hydroxylation/generated \
  --manifest notebooks/nio_m110_hydroxylation/generated/manifest_batch.csv \
  --exclude-prefix OH0 \
  --launcher-template launch_scripts/runvasp.sh
```

The second command runs the local `POTCAR_gen` inside each selected leaf only
when its POTCAR is absent or empty. Existing nonempty POTCAR files are never
replaced. It installs the production VASP 6.5.1 launcher, audits the complete selected
batch, and writes `opt_manifest.json` plus `opt_audit.{json,tsv,md}` without
submitting anything.

`--exclude-prefix OH0` is essential here because `manifest_batch.csv` includes
the pristine ligand cases; it keeps the manually launched OH-free branch out
of this audited batch. The exclusion is remembered for later `--audit-only`
runs.

After reviewing `generated/opt_audit.md`:

```bash
iface vasp opt-launch notebooks/nio_m110_hydroxylation/generated
iface vasp opt-launch notebooks/nio_m110_hydroxylation/generated --execute
```

The first command is another full preflight and dry run. `--execute` is the
only form that submits; it records every Slurm job ID and blocks duplicate
batch launches.

## Magnetism

The slab is AFM-II (NiO type-II order: Ni-O-Ni 180° superexchange pairs
antiparallel). Moments are taken, in order of preference, from: the input file
(`.extxyz`), `RUN_TEMPLATE/INCAR`'s `MAGMOM` line (tiled for a supercell), or a
frame-independent 2-colouring of the Ni `<100>` graph (`assign_afm_ii_moments`).
`compact_magmom` re-encodes them per case (`30*2 30*-2 72*0` — added OH / ligand
atoms trail as `N*0.0`).

## Surface size — 200-atom compromise

The old 120-atom, ~12 × 12 Å cell was only slightly too small, while its 2×2
repeat jumped to 480 atoms. `build_nio110_slab.py` now uses the AFM-compatible
20-fold matrix `[[4,0],[0,5]]`, built from the bundled 4.1863376 Å bulk POSCAR:

| slab | in-plane cell | atoms | effective linear scale vs old | frozen |
|---|---:|---:|---:|---:|
| old 12-fold | ~12.21 × 12.21 Å | 120 | 1.000 | 24 |
| **default compromise** | **16.75 × 14.80 Å** | **200** | **1.291** | **40** |
| old 2×2 repeat | ~24.4 × 24.4 Å | 480 | 2.000 | 96 |

For pristine-surface docking, the final deterministic orientations give minimum
all-atom ligand-to-ligand image gaps of 5.19 Å (Me-4PACz), 9.40 Å (MeO-2PACz),
9.15 Å (MeO-4PADBC), and 7.14 Å (DCZ-4P), including the contact-search tilts.
Every ligand export independently recalculates this value and refuses to write
a case below `MIN_PERIODIC_LIGAND_GAP = 3.5 Å`, so hydroxylated/boundary cases
cannot silently reintroduce touching mirror images.

`SURFACE_SUPERCELL = (1, 1, 1)` is therefore the default. The supplied slab
retains the original five-layer thickness and one frozen bottom plane. Rebuild
it from `inputs/POSCAR_bulk` with `build_nio110_slab.py` when changing the
lattice constant, thickness, or transformation matrix.

```bash
cd notebooks/nio_m110_hydroxylation
python build_nio110_slab.py
```

For a production campaign, first relax the generated **bare** 200-atom slab
once at the final 520 eV settings. Then replace the bundled compromise POSCAR
(or point `SURFACE_FILE`) at that relaxed CONTCAR before generating all
hydroxylated/passivated cases. This amortizes the clean-surface relaxation
instead of repeating it independently in every decorated optimization.

## Acceptance

- The generated **0 % case has identical atom positions, cell, and (with
  `FREEZE_MODE = "inherit"`) selective-dynamics flags to the bare input slab**
  (checked in §7).
- Every hydroxylated case passes: correct `O–H` bond lengths, no atom overlaps,
  cell/pbc unchanged, `Ni–OH` count == requested fraction, total atom count ==
  bare + added O + added H. Every dissociated case additionally requires one
  distinct protonated lattice O per selected Ni.
- Every exported Ni-containing run has an explicit, spin-balanced MAGMOM of the
  exact POSCAR length. Both interactive and batch paths call the same validator.
- Every slab uses fixed-cell `ISIF=2`; the vacuum direction is never relaxed.
- Each pattern has a top+side plot and a printed H-bond diagnostic score.

## Reference conventions this mirrors

- Exposed-Ni rule: `NiO_MD_Passivation/src/nio_md_prep/analysis/interfacial.py`
  `_static_exposed_ni_sites` (upper-half Ni, O-coordination < 6).
- Phosphonate anchor rule: `NiO_MD_Passivation/src/nio_md_prep/chemistry.py`
  `phosphonate_roles` (P bonded to exactly 3 O + 1 C).
- POTCAR generation after export: `interfaceforge/vasp.py`.
- Naming stem `NiO_m110_Big_U46[_<Ligand>]`: InterfaceForge VASP workflow
  (`docs/vasp.md`, `tests/test_aimd_protocols.py`). "U46" = `LDAUU = 4.6` on Ni-d.
