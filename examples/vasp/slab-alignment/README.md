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
- `SUSPECT_FLATNESS` or `FAILED_FLATNESS` writes
  `RELAUNCH_REVIEW_REQUIRED` and a proposed `INCAR.dipole_fix`;
- `FAILED_ANALYSIS` writes `LOCPOT_AUDIT_FAILED` because a safe proposal could
  not be generated.

The proposed `INCAR.dipole_fix` is a fresh static calculation (`NSW=0`,
`IBRION=-1`, `ISTART=0`, `ICHARG=2`) and therefore ignores potentially stale
`WAVECAR` and `CHGCAR` files. It also sets `PREC=Accurate`, `AMIN=0.01`, `NELM=200`, and
`EDIFF=1E-6` for the slower charge redistribution that can accompany
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
