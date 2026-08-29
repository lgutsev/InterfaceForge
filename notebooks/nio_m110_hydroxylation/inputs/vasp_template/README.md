# vasp_template

Template run inputs, copied into every generated case folder by the notebook
(`RUN_TEMPLATE`, §1).

| file | in the export | note |
|---|---|---|
| `INCAR` | copied, with `MAGMOM` rewritten for the exact case atom count/order, `SYSTEM` set, `ISIF = 2` for every slab, and a structure-specific `DIPOL` for decorated slabs | the AFM-II relaxation recipe: PBE + Dudarev U(Ni 3d) = 4.6 eV + D3 |
| `KPOINTS` | copied verbatim | Γ-only (the cell is large) |
| `POTCAR` | **not generated** | run the project's existing POTCAR generator against the element-ordered POSCAR |

The user-supplied reference template has the same INCAR and KPOINTS committed
here. Its POSCAR/CONTCAR are reference structures, not export templates: the
notebook writes a new ordered POSCAR for every generated chemistry.

Any local `POTCAR`, `WAVECAR`, or `CHGCAR` files remain git-ignored.
