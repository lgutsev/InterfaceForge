# NiO(110) surface hydroxylation — training-data generator

One self-contained notebook that builds **partially hydroxylated NiO(110) slabs**
(and phosphonate-ligand-decorated variants) as input for VASP+U relaxation / AIMD.
This is a standalone application notebook — it is *not* part of the
InterfaceForge package and does not modify it (it only optionally *reuses*
InterfaceForge's licensed-PAW `assemble_potcar` helper on export).

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
`nio_hydroxylation_utils.py` is the *build source* for those cells (edit there,
rerun the builder); the notebook does not import it.


| § | what |
|---|---|
| 0 | Toolkit — inlined helper functions (I/O, site finding, motifs, H-bonds, freeze, phosphonate, export). Run once; not normally edited. |
| 1–2 | Load bare slab → exposed-Ni inventory (canonical, frozen order). |
| 3 | Coverage sweep: 0/25/50/75/100 % × {clustered, scattered} × {capped, dissoc}, with selective-dynamics freezing applied. |
| 4–5 | H-bond diagnostic + manual H-orientation override. |
| 6–7 | Visualise (incl. frozen/mobile split) + sanity checks. |
| 8 | Export the hydroxylation cases as runnable VASP folders + manifest. |
| 9–12 | Phosphonate ligand: anchor detection (P–OH vs P=O) → surface-docked orientation → placement at **pristine (0 % OH) / bare / OH / boundary** → export. |
| 13 | Batch loop over **surfaces × passivants × cases × anchor-positions**. |

## Setup

```bash
pip install -r requirements.txt
```

**Everything is bundled in `inputs/`** (the bare `CONTCAR`, the four passivant
`.xyz`, and `inputs/vasp_template/` with the working `INCAR` + `KPOINTS`) — the
notebook is reproducible as-is, just run it top to bottom. Only `POTCAR` is not
bundled (VASP-licensed): set `POTCAR_ROOT` in §1, or drop a Ni/O `POTCAR` into
`inputs/vasp_template/`. The sweep cell (§3) is meant to be re-run after you
tweak a parameter, the exposed-Ni inventory (§2), or an `H` orientation (§5).

Geometry note: minimum-image distances use fractional coords + the full cell
matrix, so a non-orthogonal (hexagonal) cell is handled correctly; periodicity
is in-plane only.

## Two hydroxylation motifs (both generated at every intermediate coverage)

- **capped** (`_capped`) — an `O` placed straight up (+z) from a selected Ni at a
  2.0 Å placeholder `Ni–O` distance, with its own `H`.
- **dissoc** (`_dissoc`) — the same `Ni–OH` **plus** a proton on the nearest
  *free* surface lattice O beyond the Ni's first shell (`acceptor_search_radius`,
  default 4.0 Å) — a genuine dissociated water pair. If none is free it reduces
  to a capped hydroxide and prints a warning; `n_lattice_protons` in the manifest
  records how many true pairs formed.

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

- `FREEZE_MODE = "auto"` *(default)* — keep the CONTCAR's `F/T` flags if it has
  any; otherwise freeze the bottom `FREEZE_VALUE` (2) layers. Never leaves a slab
  un-anchored. Added OH / ligand atoms stay mobile. Tiles correctly under
  `SURFACE_SUPERCELL` (24 frozen → 96 for a 2×2).
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
     hydroxyl H** (`oh` / `boundary`), leaving the P–OH oxygens toward bare NiO;
  3. only if the upright molecule is taller than the available vacuum does it
     **auto-tilt about a horizontal axis through P** (realistic tilted-SAM
     geometry); `tilt_deg` in the manifest. In practice none of the four
     passivants need this — see the vacuum note below.
- **Placement** (`place_ligand`): P over the target `(x, y)`, the **lowest anchor
  oxygen** `OH_HEIGHT` (≈2.05 Å) above the exposed-Ni plane. Clash-lift in +z if
  needed (`docking_lift`). The molecule is shifted into the cell as a rigid body
  — never `wrap()`, which would tear it across the periodic boundary.
- **Vacuum / cell**: the slab sits in the *middle* of its cell, so the real
  headroom is the gap to the slab's **periodic image** (`slab_bottom + c`), not
  the distance to the `c` plane. `vacuum_fit` reports `vacuum_gap` (to the image);
  `ok` needs no overlaps *and* `fits_vacuum`.
- **`LIGAND_CELL_C`** (§9), applied to ligand cases only (bare / OH-only keep the
  CONTCAR cell):
  - `"auto"` *(default)* — lengthen `c` per case just enough that the whole
    molecule sits **inside** the cell with `LIGAND_VACUUM_TOP` (3 Å) above its
    top atom. Typical result: Me-4PACz ~38–40 Å, MeO-2PACz ~36–38, MeO-4PADBC
    ~41–42, DCZ-4P ~45–47 (from the 35.9 Å CONTCAR). `c` varies per case — fine
    for MLIP training; the manifest / provenance record it.
  - a number — force that `c` for every ligand case (uniform cell, e.g. `48`).
  - `None` — never touch `c`; a tall molecule then pokes past the box into the
    (empty) bottom vacuum — physically fine, just untidy in a viewer.
  `extend_vacuum` rescales only the `c` lattice vector; atom positions unchanged.

### The three anchor positions

- **`bare`** — over an exposed Ni with no hydroxyl (acid condenses with bare Ni²⁺).
- **`oh`** — over a hydroxylated exposed Ni (`Ni–OH`).
- **`boundary`** — the in-plane **midpoint between a hydroxylated exposed-Ni and
  its nearest bare exposed-Ni neighbour**: the anchor straddles the edge of an OH
  island, one acid −OH over bare Ni, the P=O turned toward the adjacent `Ni–OH`'s
  H. Highest-value case, never dropped.

### §13 batch

Set `RUN_BATCH = True`. Loops `SURFACES × BATCH_LIGANDS × BATCH_CASES ×
BATCH_POSITIONS`, re-deriving the exposed-Ni inventory and coverage pattern per
surface, docking each ligand, checking overlaps + vacuum, exporting, and writing
`generated/manifest_batch.csv`.

## Naming

```
NiO_m110_Big_U46                                             bare (0 %)
NiO_m110_Big_U46_OH50_clustered_capped                       hydroxylated
NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary      + ligand
```

`OH<pct>` = coverage fraction of the exposed-Ni inventory; `<pattern-id>` ∈
{clustered, scattered, full}; `<motif>` ∈ {capped, dissoc}; ligand token has
hyphens stripped; anchor position ∈ {bare, oh, boundary}.

## Export — runnable VASP optimization folders

`RUN_TEMPLATE` (§1) points at a working run (`INCAR` + `KPOINTS` + `POTCAR`,
optionally `runvasp.sh`). Every generated case is a launchable folder, grouped
by OH type:

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
                sort, so POSCAR / MAGMOM / LDAU* / POTCAR all agree on one order;
                Ni-up stays before Ni-down. Selective-dynamics F/T.
KPOINTS         copied from RUN_TEMPLATE
INCAR           template copied, then rewritten for THIS case's species order --
                MAGMOM, LDAUL, LDAUU, LDAUJ are per POSCAR species so they change
                with the element set:
                   bare        Ni O          LDAUL = 2 -1
                   hydroxylated H Ni O        LDAUL = -1 2 -1
                   ligand      C H N Ni O P   LDAUL = -1 -1 -1 2 -1 -1
                Ni always L=2, U=4.6; SYSTEM = case name;
                ISIF=2 + dipole for decorated slabs; pristine 0 % -> ISIF=3, LDIPOL=.FALSE.
POTCAR          sliced from the template POTCAR (or POTCAR_ROOT) into that same
                element order. inputs/vasp_template/POTCAR is git-ignored but kept
                locally; set POTCAR_ROOT if you relocate the notebook.
structure.xyz   convenience (carries the magnetic moments + freeze constraint)
provenance.json coverage / pattern / motif / sites / H-bond score / ligand info /
                freeze / supercell / magmom / input hashes / run_dir / run_inputs
```

The template `INCAR` is the **passivated-case** relaxation
(`Phosphonate on NiO(110) AFM-II` — PBE + Dudarev U(Ni 3d)=4.6 + D3 + dipole).

`manifest.csv` / `manifest_ligands.csv` carry a **`path`** column.

## Magnetism

The slab is AFM-II (NiO type-II order: Ni-O-Ni 180° superexchange pairs
antiparallel). Moments are taken, in order of preference, from: the input file
(`.extxyz`), `RUN_TEMPLATE/INCAR`'s `MAGMOM` line (tiled for a supercell), or a
frame-independent 2-colouring of the Ni `<100>` graph (`assign_afm_ii_moments`).
`compact_magmom` re-encodes them per case (`30*2 30*-2 72*0` — added OH / ligand
atoms trail as `N*0.0`).

## Surface size — passivant images

The supplied ~12 × 12 Å cell puts a standing passivant's periodic images only
**5–6.6 Å** apart (ring-to-ring). Measured minimum image contact:

| supercell | in-plane | Me-4PACz | MeO-2PACz | MeO-4PADBC | DCZ-4P |
|---|---|---|---|---|---|
| `(1,1,1)` | ~12 Å  | 6.4 Å | 6.6 Å | 5.3 Å | 5.0 Å |
| `(2,2,1)` | ~24 Å  | 16.8 Å | 17.5 Å | 15.4 Å | 15.7 Å |
| `(3,3,1)` | ~36 Å  | 28.6 Å | 29.1 Å | 27.0 Å | 27.3 Å |

**`SURFACE_SUPERCELL = (2, 2, 1)` is the default** (480 atoms, Ni240 O240) — it
`repeat()`s the *relaxed* CONTCAR (AFM pattern and freeze constraint tile with
it; only the adsorbate region then needs relaxing), 480 atoms. Drop to `(1,1,1)` for
bare / hydroxylated-only runs. To rebuild the surface from scratch
(different size / thickness / symmetric slab) use `build_nio110_slab.py`.

## Acceptance

- The generated **0 % case has identical atom positions, cell, and (with
  `FREEZE_MODE = "inherit"`) selective-dynamics flags to the bare input slab**
  (checked in §7).
- Every hydroxylated case passes: correct `O–H` bond lengths, no atom overlaps,
  cell/pbc unchanged, `Ni–OH` count == requested fraction, total atom count ==
  bare + added O + added H. The sweep asserts this before export.
- Each pattern has a top+side plot and a printed H-bond diagnostic score.

## Reference conventions this mirrors

- Exposed-Ni rule: `NiO_MD_Passivation/src/nio_md_prep/analysis/interfacial.py`
  `_static_exposed_ni_sites` (upper-half Ni, O-coordination < 6).
- Phosphonate anchor rule: `NiO_MD_Passivation/src/nio_md_prep/chemistry.py`
  `phosphonate_roles` (P bonded to exactly 3 O + 1 C).
- VASP run-dir packaging + `assemble_potcar`: `interfaceforge/vasp.py`.
- Naming stem `NiO_m110_Big_U46[_<Ligand>]`: InterfaceForge VASP workflow
  (`docs/vasp.md`, `tests/test_aimd_protocols.py`). "U46" = `LDAUU = 4.6` on Ni-d.
