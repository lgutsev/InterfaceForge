# inputs

Bundled with the notebook so it is reproducible as-is — `NiO_m110_hydroxylation.ipynb`
reads everything here by relative path.

| file | what |
|---|---|
| `NiO_110_AFM_compromise.POSCAR` | default bare NiO(110) slab — 200 atoms (Ni₁₀₀O₁₀₀), AFM-II, rectangular 16.75 × 14.80 Å cell, five physical layers, bottom plane `F F F` (40/200 atoms). |
| `NiO_110_AFM_compromise.extxyz` | same slab with initial magnetic moments and the freeze constraint retained explicitly. |
| `POSCAR_bulk` | supplied relaxed conventional NiO cell (`a = 4.1863376 Å`) used by `../build_nio110_slab.py`. |
| `CONTCAR` | legacy 120-atom ~12 × 12 Å slab, retained only for comparison/backward compatibility. Its passivant images are too close for production ligand runs. |
| `Me4PACz.xyz` | Me-4PACz passivant (mono-phosphonic acid) |
| `MeO-2PACz.xyz` | MeO-2PACz passivant |
| `MeO-4PADBC.xyz` | MeO-4PADBC passivant |
| `DCZ-4P.xyz` | DCZ-4P **bis**-phosphonate — the lowest-atom-index P is used as the anchor |
| `vasp_template/` | `INCAR` + `KPOINTS` from the working AFM-II relaxation; see its README |

The notebook uses the 200-atom compromise slab directly with
`SURFACE_SUPERCELL = (1,1,1)`. Do not apply the old `(2,2,1)` repeat unless a
deliberate 800-atom calculation is wanted. Every ligand export audits its
in-plane periodic self-image distance and rejects a gap below 3.5 Å.

To use a different / freshly-built surface, drop a `POSCAR` / `CONTCAR` / `.extxyz`
here and point `SURFACE_FILE` at it (an `.extxyz` may carry its own magnetic
moments and freeze constraint — e.g. the output of `../build_nio110_slab.py`).
