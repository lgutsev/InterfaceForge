# vasp_template

Template run inputs, copied into every generated case folder by the notebook
(`RUN_TEMPLATE`, §1).

| file | in the export | note |
|---|---|---|
| `INCAR` | copied, with `MAGMOM` rewritten for the case's atom count/order, `SYSTEM` set, and `ISIF = 2` for every decorated slab (the clean 0 % slab keeps `ISIF = 3`) | the AFM-II relaxation recipe: PBE + Dudarev U(Ni 3d) = 4.6 eV + D3 |
| `KPOINTS` | copied verbatim | Γ-only (the cell is large) |
| `POTCAR` | **not bundled** (VASP-licensed). Drop `PAW_PBE Ni` + `PAW_PBE O` here for the Ni/O-only pristine case, or set `POTCAR_ROOT` in §1 to a per-element PAW tree (`<root>/Ni/POTCAR`, `<root>/C/POTCAR`, …) for the hydroxyl / ligand cases | POSCAR element order |

`POTCAR` (and any `WAVECAR` / `CHGCAR` you leave here) are git-ignored.
