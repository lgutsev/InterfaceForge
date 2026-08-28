# inputs

Bundled with the notebook so it is reproducible as-is — `NiO_m110_hydroxylation.ipynb`
reads everything here by relative path.

| file | what |
|---|---|
| `CONTCAR` | bare NiO(110) slab — 120 atoms (Ni₆₀O₆₀), AFM-II, hexagonal ~12 × 12 Å cell, selective dynamics (bottom plane `F F F`). z is the surface normal. |
| `Me4PACz.xyz` | Me-4PACz passivant (mono-phosphonic acid) |
| `MeO-2PACz.xyz` | MeO-2PACz passivant |
| `MeO-4PADBC.xyz` | MeO-4PADBC passivant |
| `DCZ-4P.xyz` | DCZ-4P **bis**-phosphonate — the lowest-atom-index P is used as the anchor |
| `vasp_template/` | `INCAR` + `KPOINTS` from the working AFM-II relaxation; see its README |

The notebook tiles `CONTCAR` to a `SURFACE_SUPERCELL` (default `(2,2,1)` → ~24 Å,
480 atoms) so standing passivants don't interact with their periodic images.

To use a different / freshly-built surface, drop a `POSCAR` / `CONTCAR` / `.extxyz`
here and point `SURFACE_FILE` at it (an `.extxyz` may carry its own magnetic
moments and freeze constraint — e.g. the output of `../build_nio110_slab.py`).
