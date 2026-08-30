"""Build a bare AFM-II NiO(110) slab for the hydroxylation notebook.

Improvements over the `fcc110("Ni", ...) + manual O sublattice` recipe:

* **rocksalt NiO(110)** with the correct 1:1 Ni:O stoichiometry and a non-polar
  (Tasker type 1) termination;
* an **AFM-compatible 20-fold in-plane cell** (16.75 x 14.80 A for the supplied
  bulk POSCAR): 29% larger in effective linear size than the original 12-fold
  cell, but only 200 atoms rather than the 480-atom 2x2 repeat;
* the original **five physical layers and one frozen bottom layer**, preserving
  the established 1x1 slab/freezer balance instead of silently thickening it;
* **AFM-II moments from the linear Ni-O-Ni superexchange graph**, with
  assertions: zero net moment and every defining 180-degree pair antiparallel;
* writes **POSCAR** (``sort=False`` so the compact ``MAGMOM`` stays valid) **and
  ``slab.extxyz``** (carries the moments into the notebook) plus a ``MAGMOM`` /
  INCAR starter block.

The old 12-fold matrix ``[[2,-3],[2,3]]`` is only about 12.2 x 12.2 A.  Repeating
that cell 2x2 jumps directly to 480 atoms.  The default rectangular matrix
``[[4,0],[0,5]]`` is the tractable intermediate: its determinant is 20, both
[001] coefficients are even (so AFM-II is periodic), and passivant images no
longer touch.  The notebook consumes this output directly with
``SURFACE_SUPERCELL = (1, 1, 1)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc110, make_supercell
from ase.constraints import FixAtoms
from ase.io import read, write
from nio_hydroxylation_utils import assign_afm_ii_moments  # same directory

# ----------------------------------------------------------------------------
# knobs
# ----------------------------------------------------------------------------
BULK_POSCAR    = Path("inputs/POSCAR_bulk")
A_NIO          = None        # None -> mean cell length from BULK_POSCAR
N_LAYERS       = 5           # same physical thickness as the original 1x1 slab
SUPERCELL_P    = np.array([   # 20-fold, rectangular, AFM-II-periodic compromise
    [4, 0, 0],
    [0, 5, 0],
    [0, 0, 1],
])
VACUUM_TOTAL   = 30.0        # same center(vacuum=15 A) convention as the original
FREEZE_LAYERS  = 1           # same single bottom (110) layer as the original;
                             # 24/120 old -> 40/200 compromise, both exactly 20%
                             # bottom layer held fixed -- the interface is
                             #   anchored from the bottom (F F F); everything above
                             #   relaxes (T T T)
MU_NI          = 2.0         # bohr magneton, high-spin Ni(2+) d8
OUT_DIR        = Path("inputs")
OUT_STEM       = "NiO_110_AFM_compromise"

if A_NIO is None:
    bulk = read(BULK_POSCAR)
    lengths = np.asarray(bulk.cell.lengths(), dtype=float)
    if not np.allclose(lengths, lengths.mean(), rtol=0.0, atol=1e-3):
        raise ValueError(
            f"{BULK_POSCAR} is not cubic within 0.001 A: {lengths.tolist()}")
    A_NIO = float(lengths.mean())

# ----------------------------------------------------------------------------
# 1. rocksalt NiO(110) primitive slab  (Ni FCC(110) + O displaced by cell[0]/2
#    along [001] -- puts O at the rock-salt edge centres; this is the same
#    construction as the original recipe, which gives a clean (110) cut).
# ----------------------------------------------------------------------------
ni = fcc110("Ni", size=(1, 1, N_LAYERS), a=A_NIO, vacuum=None, orthogonal=True, periodic=True)
ox = Atoms(["O"] * len(ni), positions=ni.positions + ni.cell[0] / 2, cell=ni.cell, pbc=ni.pbc)
prim = ni + ox
prim.set_cell(ni.cell)

# ----------------------------------------------------------------------------
# 2. intermediate AFM-compatible in-plane supercell.
#    AFM-II changes sign after one primitive [001] translation, so the first
#    coefficient of each periodic surface vector must be even.  SUPERCELL_P's
#    two coefficients are 4 and 0; both preserve the spin phase.
# ----------------------------------------------------------------------------
P = np.asarray(SUPERCELL_P, dtype=int)
if P.shape != (3, 3) or not np.array_equal(P[2], [0, 0, 1]):
    raise ValueError("SUPERCELL_P must be a 3x3 in-plane slab transformation")
if np.any(P[:2, 0] % 2):
    raise ValueError(
        "SUPERCELL_P breaks AFM-II periodicity: both [001] coefficients must be even")
slab = make_supercell(prim, P)
slab.set_pbc((True, True, False))
slab.wrap()
slab.center(vacuum=VACUUM_TOTAL / 2, axis=2)

# ----------------------------------------------------------------------------
# 3. AFM-II moments -- 2-colour the Ni-O-Ni 180° superexchange graph
# ----------------------------------------------------------------------------
assign_afm_ii_moments(slab, mu=MU_NI)          # prints up/down/net; [CHECK] -> use NUPDOWN
sym = np.array(slab.get_chemical_symbols())
ni = np.where(sym == "Ni")[0]
o_ = np.where(sym == "O")[0]
moments = slab.get_initial_magnetic_moments()

# ----------------------------------------------------------------------------
# 4. reorder Ni(up) | Ni(down) | O  -> compact MAGMOM
# ----------------------------------------------------------------------------
up = ni[moments[ni] > 0]
dn = ni[moments[ni] < 0]
order = np.concatenate([up, dn, o_])
slab = slab[order]
slab.set_initial_magnetic_moments(
    [MU_NI] * len(up) + [-MU_NI] * len(dn) + [0.0] * len(o_))

magmom = f"{len(up)}*{MU_NI} {len(dn)}*{-MU_NI} {len(o_)}*0.0"
if len(up) != len(dn):
    raise RuntimeError(f"unbalanced AFM-II cell: {len(up)} up / {len(dn)} down")

# ----------------------------------------------------------------------------
# 4b. anchor the slab from the bottom: freeze the lowest FREEZE_LAYERS (110) layers
# ----------------------------------------------------------------------------
z = slab.positions[:, 2]
planes = np.sort(np.unique(np.round(z - z.min(), 1)))          # distinct (110) sub-planes
if not 0 <= FREEZE_LAYERS <= len(planes):
    raise ValueError(f"FREEZE_LAYERS={FREEZE_LAYERS} outside 0..{len(planes)}")
if FREEZE_LAYERS:
    zcut = planes[FREEZE_LAYERS - 1] + z.min() + 0.05
    frozen = np.where(z <= zcut)[0]
else:
    frozen = np.array([], dtype=int)
slab.set_constraint(FixAtoms(indices=frozen.tolist()))
expected_frozen = len(slab) * FREEZE_LAYERS // N_LAYERS
if len(frozen) != expected_frozen:
    raise RuntimeError(
        f"freezer mismatch: expected {expected_frozen} atoms in "
        f"{FREEZE_LAYERS}/{N_LAYERS} layers, found {len(frozen)}")

# ----------------------------------------------------------------------------
# 5. write POSCAR (order-preserving, selective dynamics) + extxyz + starter
# ----------------------------------------------------------------------------
OUT_DIR.mkdir(exist_ok=True)
write(OUT_DIR / f"{OUT_STEM}.POSCAR", slab, format="vasp", direct=True,
      vasp5=True, sort=False)
write(OUT_DIR / f"{OUT_STEM}.extxyz", slab)          # feed this to the notebook

(OUT_DIR / f"{OUT_STEM}.INCAR_starter").write_text(
    f"# starter tags for {OUT_STEM} -- merge into your production INCAR\n"
    f"ISPIN  = 2\n"
    f"MAGMOM = {magmom}\n"
    f"LDAU   = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2 -1\nLDAUU = 4.6 0.0\nLDAUJ = 0.0 0.0\n"
    f"LMAXMIX = 4\n"
    f"LDIPOL = .TRUE.\nIDIPOL = 3        # asymmetric slab -> dipole correction\n",
    encoding="utf-8")

print("NiO(110) AFM-II slab")
print(f"  a (bulk)      : {A_NIO} A  ({BULK_POSCAR})")
print(f"  layers        : {N_LAYERS}")
print(f"  supercell P   : {P[:2, :2].tolist()}   ({int(round(np.linalg.det(P)))}-fold)")
print(f"  cell lengths  : {np.round(slab.cell.lengths(), 2)}")
print(f"  cell angles   : {np.round(slab.cell.angles(), 1)}")
print(f"  atoms         : {len(slab)}   ({len(up)} Ni-up, {len(dn)} Ni-down, {len(o_)} O)")
print(f"  frozen (bottom): {len(frozen)} atoms  ({FREEZE_LAYERS} layers, F F F)")
print(f"  min image dist : {min(slab.cell.lengths()[:2]):.1f} A")
print(f"  effective scale: {np.sqrt(abs(np.linalg.det(P)) / 12.0):.3f} x old 12-fold cell")
print(f"  MAGMOM        : {magmom}")
print(f"  wrote {OUT_DIR / (OUT_STEM + '.POSCAR')} , .extxyz , .INCAR_starter")
