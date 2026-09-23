# Vacuum-aligned slab band edges

This example compares pristine and passivated asymmetric VASP slabs without
assuming that their raw Kohn-Sham eigenvalues share a common zero. It extracts
the two vacuum sides separately from each `LOCPOT`, checks plateau flatness,
aligns the VBM and CBM to the selected physical side, and subtracts the
configured pristine reference.

The default example maps every `MAPI_MAI_Surf_*` case to `MAPI_MAI_Surf` and
every `MAPI_PbI2_Surf_*` case to `MAPI_PbI2_Surf`. It selects the high-z vacuum
side. Edit `slab_alignment.json` if the passivant faces the other side.

## Run interactively

Install plotting support and copy the configuration to the calculation root:

```bash
pip install -e ".[slab-align]"
cp examples/vasp/slab-alignment/slab_alignment.json /path/to/calculations/
cd /path/to/calculations
iface vasp slab-align . --run-sumo
```

Each matching immediate child needs `LOCPOT` for the flatness audit. `OUTCAR`
and `vasprun.xml` are additionally required for band-edge alignment; a missing
electronic-structure file no longer prevents the LOCPOT flatness decision.
`INCAR` is inspected when present. The command writes:

- `band_edge_alignment.tsv`, `.json`, and `.txt` in the root;
- `dipole_flatness_audit.tsv` and `.txt` plus `relaunch_review_queue.txt` in
  the root;
- `locpot.dat`, an annotated `vacuum_profile.png`, and a simpler
  work-function-style `Workfunction.png` in every analyzed child;
- `sumo_dosplot.log` and a dedicated `sumo_dos_data/` directory when
  `--run-sumo` is requested.

This is a post-processing command: it never runs VASP and never writes
`INCAR`, `POSCAR`, `CONTCAR`, `KPOINTS`, `POTCAR`, `WAVECAR`, `CHGCAR`,
`LOCPOT`, `OUTCAR`, `DOSCAR`, or `vasprun.xml`. SUMO receives the source
`vasprun.xml` with `--filename` and is executed from `sumo_dos_data/`, so its
generated files cannot land beside VASP restart files.

The audit recognizes the expected sawtooth discontinuity introduced by
VASP's dipole correction. Each plot highlights only the configured physical
surface's selected plateau, marks that side's correction-plane step when
present, and moves the fit boundary to exclude the complete short transition.
The detector uses the spatial extent of the change: a localized reset is not
a reason to relaunch, while a gradual change distributed through the vacuum
remains part of the fit and fails the flatness gate.

Flatness triage is automatic. No calculation is submitted and no `INCAR` is
ever overwritten:

- `OK` writes `LOCPOT_FLATNESS_OK` and needs no dipole improvement;
- `SUSPECT_FLATNESS`, `FAILED_FLATNESS`, or a VASP vacuum cross-check issue
  writes `RELAUNCH_REVIEW_REQUIRED` and, when an INCAR-only recovery is
  defensible, a proposed `INCAR.dipole_fix`;
- `FAILED_ANALYSIS` writes `LOCPOT_AUDIT_FAILED` because a safe proposal could
  not be generated.

The proposed `INCAR.dipole_fix` is a fresh static calculation (`NSW=0`,
`IBRION=-1`, `ISTART=0`, `ICHARG=2`) and therefore ignores potentially stale
`WAVECAR` and `CHGCAR` files. It also sets `PREC=Accurate`, `AMIN=0.01`,
`NELM=200`, and `EDIFF=1E-6` for the slower charge redistribution that can accompany
`LDIPOL`; enables `LVHAR` and VASP 6.4.3+'s `LVACPOTAV` field-free vacuum
analysis; and requests projected DOS with `LORBIT=11`. The original `INCAR`
is never replaced automatically.

When `LVACPOTAV` output is present, the audit records VASP's upper and lower
vacuum levels and cross-checks the selected value against the independently
fitted LOCPOT plateau. VASP warnings about a missing field-free region or
excess vacuum charge trigger manual review. If no field-free region exists,
the workflow recommends increasing the vacuum and deliberately does not write
an `INCAR.dipole_fix`; changing `DIPOL` alone cannot repair that geometry.

Use `--no-write-dipole-fixes` to run the same audit and marker generation
without creating proposed INCAR files.

The reported values are

\[
E_\mathrm{CBM}^{\mathrm{vac}} = \varepsilon_\mathrm{CBM} - V_\mathrm{vac}
\]

and

\[
\Delta E_\mathrm{CBM} =
E_{\mathrm{CBM,pass}}^{\mathrm{vac}} -
E_{\mathrm{CBM,pristine}}^{\mathrm{vac}},
\]

with the same definitions for the VBM. A positive delta therefore means a
shift upward, toward vacuum. The vacuum terms cancel only when the fitted
plateaus are actually equal; the command always measures both instead of
assuming this.

## Run on the `single` partition

Copy or submit the repository launcher from the calculation root:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/run_slab_alignment_single.sbatch
```

Set `SLAB_ALIGNMENT_CONFIG` to select another configuration filename. The
launcher runs the automatic flatness audit, creates the review queue and
proposed INCAR files for flagged cases, and then runs `sumo-dosplot` in each
matched child. A nonzero job exit is deliberate when a folder fails analysis
or the strict flatness gate; inspect `relaunch_review_queue.txt` rather than
blindly resubmitting every folder.

## Debugging sequence

1. Run one calculation first with `--only MAPI_MAI_Surf`.
2. Inspect its `vacuum_profile.png`; confirm the shaded selected-side window is
   atom-free and flat.
3. Check `selected_swing_eV` and `selected_std_eV` in
   `dipole_flatness_audit.tsv`. A failed flatness gate prevents reference
   subtraction.
4. Open `relaunch_review_queue.txt`. For each listed folder compare the current
   and proposed inputs with `diff -u FOLDER/INCAR FOLDER/INCAR.dipole_fix`.
   InterfaceForge never applies the proposal or relaunches VASP.
5. Confirm the last OUTCAR and `vasprun.xml` Fermi energies agree. A mismatch
   marks that folder failed.
6. Inspect SUMO PDOS before interpreting a global CBM shift. A BPDCA-derived
   unoccupied state below the perovskite CBM is not the same as movement of the
   perovskite-derived CBM.
7. After the reference passes, run the full root and compare each passivated
   case only with its configured termination reference.

The parser reads raw LOCPOT values directly in eV. It does not use ASE's
charge-density rescaling. For a periodic slab it cuts the largest atom-free gap
at its midpoint and fits the low-z and high-z halves independently; the two
physical sides are never merged.

The proposed fractional `DIPOL_z` is the periodic mass-weighted ionic center
and is a reviewable heuristic, not proof that a misplaced `DIPOL` caused the
residual field. The proposal preserves existing `DIPOL_x` and `DIPOL_y`, fixes
`IDIPOL = 3`, and should be accepted only after checking the structure,
compactness, SCF convergence, and vacuum thickness.

## Tight static SCF for flagged slabs

A sloped vacuum with correct `LDIPOL`/`IDIPOL` is often an electronic
convergence problem rather than a dipole-placement problem. A relaxation run at
`EDIFF=1E-4` with `AMIN=0.10` on a >50 A cell is exactly the case where VASP
warns about charge sloshing along the long vector. Moving `DIPOL` does not
repair an unconverged density. `slab-tight-scf` prepares a controlled test:

```bash
iface vasp slab-tight-scf . --config slab_alignment_fapi.json --dry-run   # plan only
iface vasp slab-tight-scf . --config slab_alignment_fapi.json --copy '*.sbatch'
```

The command reads `band_edge_alignment.json` and parses every daughter's OUTCAR
for EDIFF, NELM, AMIN, the per-ionic-step SCF convergence, ionic-relaxation
termination, final forces, and the charge-sloshing and vacuum-charge warnings.
By default, a parent geometry that exhausted NSW or retains a final maximum
force above 0.05 eV/A is reported but not blocked; use `--require-relaxed` to
turn failed ionic convergence into a hard block, and `--force-warn` to change
the force-warning threshold. Geometry-warning parents are also prepared under
`relax_continue/<folder>/` by default. Those continuation runs start from
`CONTCAR`, preserve the physical model, and use `EDIFF=1E-6`,
`EDIFFG=-0.03 eV/A`, `NSW=200`, `NELM=200`, and `AMIN=0.01`. Use
`--no-relax-restarts` to disable this or the `--relax-*` options to adjust it.
The command then writes `tight_scf/<folder>/` for every flagged daughter (`SUSPECT_*`,
`FAILED_FLATNESS`, or a VASP vacuum warning), plus that daughter's configured
reference as a same-settings control. Each new folder contains:

- `POSCAR` copied from the parent `CONTCAR`, which keeps the final geometry and
  cell;
- the parent `KPOINTS`, `POTCAR`, and `WAVECAR` (`--wavecar none` gives an
  atomic start);
- `INCAR`, changed only in `NSW=0`, `IBRION=-1`, `ISTART`/`ICHARG`
  (self-consistent, never `ICHARG=11`), `EDIFF=1E-7`, `NELM=200`,
  `AMIN=0.01`, `LDIPOL`, `IDIPOL`, `LVHAR`, `LVACPOTAV`, and `LCHARG`. The
  functional, cutoff, `LREAL`, and `DIPOL` are kept so that any change can be
  attributed to convergence;
- `INCAR.parent` and `TIGHT_SCF_PROVENANCE.json`, which holds the audit
  metrics (including cell-tilt status), parent SCF/ionic diagnostics, geometry
  warning policy, and the tag-level INCAR diff.

Parents are never modified and nothing is submitted. The continuation tree is
independent of tight-SCF selection, so a slab can be skipped for vacuum repair
yet still receive a geometry restart when it exhausted NSW or retains large
residual forces. Folders that pass the audit, or whose LOCPOT audit failed, are
skipped from the tight-SCF tree unless `--select all` is given. A parent without
a final OUTCAR timing block, or with a CONTCAR/POSCAR
count or POTCAR species-order mismatch, is `BLOCKED` with a reason in
`tight_scf/tight_scf_plan.txt`. Existing output folders are never
overwritten; `--overwrite` refreshes inputs only while no OUTCAR is present.
`--dipol suggested` moves DIPOL to the audit's ionic centre, but change one
variable at a time.

For LOCPOT-heavy auditing and planning, use the scheduler rather than the head
node. The combined planning launcher reruns `slab-align` before
`slab-tight-scf --dry-run`, so the plan cannot consume a stale audit:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/run_slab_repair_plan_single.sbatch
```

It auto-detects `slab_alignment_fapi.json` when present (otherwise
`slab_alignment.json`), or accepts `SLAB_ALIGNMENT_CONFIG`. After reviewing the
plan, prepare the unrun repair inputs through the scheduler as well:

```bash
sbatch /path/to/InterfaceForge/launch_scripts/run_slab_repair_prepare_single.sbatch
```

The preparation launcher refreshes only destinations without an `OUTCAR`,
copies `runvasp.sh` when present, and prepares both `tight_scf/` and
`relax_continue/`.

After the runs finish, re-audit the new tree with the same configuration and
compare `selected_swing_eV` and the work function against the parent audit:

```bash
iface vasp slab-align tight_scf --config slab_alignment_fapi.json --no-write-dipole-fixes
```

If the slope persists at tight convergence, inspect the planar-averaged CHGCAR
for a genuinely charge-free vacuum region, and then test additional vacuum.

## Publication figures for the selected structures

After the final four calculations pass the flatness audit, copy the dedicated
manifest to the calculation root and submit the publication post-processing on
the `single` partition:

```bash
cp /path/to/InterfaceForge/examples/vasp/slab-alignment/slab_publication.json .
sbatch /path/to/InterfaceForge/launch_scripts/run_slab_publication_single.sbatch
```

The default manifest intentionally includes only these comparisons:

- `MAPI_MAI_Surf` versus `MAPI_MAI_Surf_BPDCA_D`;
- `MAPI_PbI2_Surf` versus `MAPI_PbI2_Surf_BPDCA_B`.

`iface vasp slab-publish` refuses a selected structure whose physical-side
plateau is not accepted by the flatness thresholds. It runs `sumo-dosplot`
with its Fermi-level shift disabled and the same configured Gaussian broadening
for all four cases, then moves each DOS energy axis onto the
same vacuum reference used for the VBM and CBM. The ligand projection is based
on atoms present in the passivated structure but absent from its pristine
reference; this prevents methylammonium C/N/H from being mislabeled as BPDCA.
The two calculations may use different surface-normal cell lengths when they
retain the same surface-normal direction; independently padding each model to a
target vacuum thickness does not invalidate atom matching. A small in-plane cell
relaxation (an `ISIF` 3/4 run nudges the surface lattice by a fraction of an
angstrom) is also tolerated up to `in_plane_cell_tolerance_angstrom` (default
0.35 A) and matched against the averaged in-plane cell. A genuine in-plane
lattice change -- a different supercell or a rotation -- is a multi-angstrom
component difference and is still rejected.

SUMO writes its raw `*_dos.dat` energy column relative to the internally
adjusted Fermi level (the VBM for a semiconductor), even when `--no-shift` is
used for SUMO's own plot. InterfaceForge restores the vacuum reference by
adding the independently calculated vacuum-aligned VBM. It also expands every
framework species into explicit species-local atom indices because bare atom
selectors such as `Pb,I` produce empty projections in current SUMO releases.
An electronic figure is refused if appreciable total DOS remains inside the
eigenvalue-defined gap after alignment.

The command writes `vacuum_validation.{pdf,png,svg}` and
`electronic_alignment.{pdf,png,svg}` under `publication_figures`. The first is
a 2x2 selected-plateau validation figure. Each vacuum panel is cropped to the
configured physical side: it shows the fitted plateau and
`vacuum_context_angstrom` of the adjacent surface-side approach, but excludes
the opposite vacuum and the dipole-correction reset. Because the plotted data
are subset before drawing, the deep potential oscillations inside the slab do
not set the panel's y scale. The second is a 2x3 figure containing
pristine and passivated Pb/I/BPDCA PDOS plus the vacuum-aligned VBM/CBM diagram
for each termination. It also writes `publication_band_edges.tsv` and a JSON
manifest recording the exact folders, atom selections, and interpretation
guard. Positive band-edge deltas mean movement upward, toward vacuum.

The publication launcher is post-processing only as well. SUMO runs from
each calculation's `publication_dos_data/` directory with `vasprun.xml` as an
explicit absolute input. Existing VASP files are never moved, renamed,
deleted, or replaced.

To remake figures from existing `publication_dos_data` without rerunning SUMO:

```bash
iface vasp slab-publish . --config slab_publication.json
```

### Surface-normal axis

`slab-align` supports vacuum along x/a, y/b, or z/c. Cell tilt is now a
two-level quality-control check rather than a hard 0.1-degree cutoff. By
default, tilts above 0.1 degrees are retained and marked `TILT_WARNING`, while
tilts above 1.0 degree are rejected as `TILT_FAILURE` for manual review.
Both thresholds are configurable with `tilt_warn_degrees` and
`tilt_fail_degrees`. Distances use the exact perpendicular repeat length
(cell volume divided by in-plane area), and `normal_tilt_degrees`,
`tilt_status`, and `normal_length_A` are recorded in JSON and TSV. These
analysis thresholds are not a certification of VASP dipole-correction
validity; inspect the recorded IDIPOL, geometry, and vacuum plateau.
Old configs default to z. For an x-normal slab use:

```json
{
  "axis": "x",
  "side": "high-x",
  "tilt_warn_degrees": 0.1,
  "tilt_fail_degrees": 1.0,
  "references": [
    {"prefix": "FAPI_FAI_Surf", "reference": "FAPI_FAI_Surf"}
  ]
}
```

Use `low-x` for the opposite face (and analogously `high-y`/`low-y`).
An explicit side can also select the axis without an `axis` key; conflicting
axis/side settings are rejected. Averaging, periodic vacuum detection, ionic
center, plot labels and `locpot.dat` coordinates follow this selection.
JSON/TSV include `axis`, `suggested_DIPOL_normal`, and `dipole_axis_status`.
The legacy `suggested_DIPOL_z` is populated only for z; the legacy profile
`c_length_A` field is the selected length and has a `normal_length_A` alias.

Generated `INCAR.dipole_fix` uses IDIPOL=1/2/3 for x/y/z and changes only the
corresponding DIPOL component. The recorded OUTCAR IDIPOL takes precedence
for the direction audit; a mismatch is flagged even with a flat potential.
An unavailable OUTCAR direction is marked UNKNOWN. VASP vacuum-level
crosschecks are performed only when its recorded direction matches.
Changing the analysis axis does not repair a calculation performed with the
wrong dipole direction: obtain a self-consistent static calculation with the
correct IDIPOL before reporting the corrected work function.

This axis option applies to `slab-align`; it does not generalize the separate
`slab-publish` structure/DOS layout workflow.
