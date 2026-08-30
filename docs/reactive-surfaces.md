# Reactive magnetic surface campaigns

`iface surface` models chemically valid surface **states**, not isolated file
edits. A state records its occupied reactive sites, coverage pattern, reaction
motif, proton balance, magnetic initialization, frozen atoms, parent state, and
optional molecular binding mode. This is intentionally separate from
`iface vasp geom`, which remains the generic structure-editing layer.

## Commands

```bash
iface surface init -o surface_campaign.yaml
iface surface analyze slab.vasp --metal Ni --anion O
iface surface cell-optimize primitive_slab.vasp \
  --adsorbate Me4PACz.xyz \
  --min-multiplier 12 --max-multiplier 24 --max-atoms 240 \
  --min-translation 14 --min-image-gap 3.5 --max-aspect 1.3 \
  --translation-parity 1 0 --freeze-bottom-layers 1 \
  -o compromise.vasp
iface surface plan surface_campaign.yaml
iface surface build surface_campaign.yaml
iface surface audit generated_surface_campaign -o surface_audit.csv
iface surface select candidates.csv labeling_queue.csv --count 40 \
  --feature-column coverage --feature-column reaction_coordinate \
  --state-column motif --state-column initial_binding --max-per-state 5
```

`plan` performs the complete geometry generation and docking audit in memory.
It writes nothing. `build` exports element-ordered POSCAR/extxyz structures,
rewritten INCAR/KPOINTS files when a template is supplied, per-run
`provenance.json`, a campaign manifest, and `state_graph.json`. The graph
contains explicit coverage-growth, lattice-protonation, and adsorbate-binding
edges, so downstream selection does not infer reaction ancestry from directory
names. POTCAR is never generated.

## Cell optimization

The optimizer enumerates two-dimensional Hermite-normal-form transformation
matrices and rejects candidates that violate any configured constraint:

- atom budget and determinant range;
- shortest in-plane lattice translation;
- maximum aspect ratio;
- adsorbate-to-periodic-image clearance over sampled azimuths;
- collinear AFM translation parity.

For a phosphonic acid, the molecule is first recognized by its P(O)3-C anchor
and rotated into the intended head-down/body-up surface frame. This prevents a
raw XYZ stored on its side from spuriously forcing a huge slab. For NiO(110),
`--translation-parity 1 0` expresses that one primitive `[001]` translation
flips AFM-II spin; both generated surface vectors must therefore have even
first coefficients.

With the bundled primitive NiO(110) geometry and Me4PACz, the documented
constraints select:

```text
matrix       [[4, 0], [0, 5]]
atoms        200
cell         16.745 x 14.801 A
image gap    5.194 A
```

## Reactive states

Schema version 1 supports two initial hydroxylation motifs:

- `terminal_hydroxyl`: one exploratory OH fragment per occupied metal site;
- `dissociated_water`: one terminal metal-OH plus one proton on a **distinct**
  nearby lattice oxygen per occupied metal site.

The dissociated-water builder uses maximum-cardinality bipartite matching. It
fails instead of exporting a state when every selected metal cannot receive a
distinct lattice-oxygen proton acceptor. Thus a folder labelled
`dissociated_water` always contains exactly one H2O equivalent per occupied
site.

At intermediate coverage, `clustered` grows a compact island and `scattered`
uses farthest-point selection. The clean and full-coverage endpoints are
deduplicated.

## Adsorbates and magnetism

Phosphonic acids are recognized from a P atom bonded to three O and one C.
Two explicitly different neutral starting modes are available:

- `direct`: terminal phosphonate O over an unoccupied exposed metal site;
- `hbond`: terminal phosphonate O placed beyond a surface O-H donor.

Docking searches molecular azimuth, enforces a hard contact floor, audits the
ligand's in-plane self-image gap, and lengthens only the surface-normal cell
vector when the requested slab-image vacuum is insufficient.

`magnetism.mode: superexchange` constructs nearly linear
magnetic-metal--bridge--magnetic-metal edges, two-colours that graph, rejects
frustrated or unbalanced cells, and carries the resulting per-atom moments
through reaction-state construction, ligand addition, and element ordering.

## Audit

Every exported leaf contains enough provenance to classify it later. The audit
prefers the relaxed `CONTCAR`, reports detached protons and phosphonate
metal-bound/non-chemisorbed character, extracts the final `TOTEN`, and compares
the last OUTCAR magnetization table against the initial spin signs. Missing
OUTCAR data is reported as `MISSING`, not as a successful spin check.

## Mechanism-aware labeling selection

`iface surface select` reuses InterfaceForge's uncertainty-plus-descriptor-
diversity ranking but places a quota on the combined reactive-state key. This
prevents a large population of easy clean-surface frames from consuming the
entire DFT labeling budget while dissociated, H-bonded, reconstructed, or
spin-changed states remain unseen. If `--state-column` is omitted, recognized
coverage, motif, binding, and spin columns present in the CSV are combined
automatically. The selected CSV records its state key and selection diagnostics.

## Verification boundary

The bundled NiO test exercises the real 200-atom slab and Me4PACz molecule. It
checks 20 exposed sites, 50-up/50-down AFM-II ordering, 40 frozen atoms, 15
reactive states, 15 decorated states, VASP input rewriting, periodic-image
clearance, and recovery of the 200-atom compromise from the primitive slab.
No generated VASP relaxation has yet been completed and scientifically
reviewed; the subsystem is therefore automated-test verified rather than
human-validated for production chemistry.
