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
- `sumo_dosplot.log` when `--run-sumo` is requested.

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
