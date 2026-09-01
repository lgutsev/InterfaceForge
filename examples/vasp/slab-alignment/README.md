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

Each matching immediate child must contain `LOCPOT`, `OUTCAR`, and
`vasprun.xml`. `INCAR` is inspected when present. The command writes:

- `band_edge_alignment.tsv`, `.json`, and `.txt` in the root;
- `locpot.dat` and `vacuum_profile.png` in every analyzed child;
- `sumo_dosplot.log` when `--run-sumo` is requested.

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
launcher runs `sumo-dosplot` in each matched child after alignment.

## Debugging sequence

1. Run one calculation first with `--only MAPI_MAI_Surf`.
2. Inspect its `vacuum_profile.png`; confirm the shaded selected-side window is
   atom-free and flat.
3. Check `selected_swing_eV` and `selected_std_eV` in the TSV. A failed
   flatness gate prevents reference subtraction.
4. Compare `current_LDIPOL`, `current_IDIPOL`, and `current_DIPOL`. To generate
   a proposed correction without touching `INCAR`, rerun with
   `--write-dipole-fixes`; this writes `INCAR.dipole_fix` only for a non-flat
   selected side.
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
