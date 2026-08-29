"""Build a bare AFM-II NiO(110) slab for the hydroxylation notebook.

Improvements over the `fcc110("Ni", ...) + manual O sublattice` recipe:

* **rocksalt NiO(110)** with the correct 1:1 Ni:O stoichiometry and a non-polar
  (Tasker type 1) termination;
* **in-plane supercell sized to a target minimum image distance** so a standing
  phosphonate SAM does not interact with its periodic neighbour (the 12x12 A
  cell gave only ~5-6 A ring-to-ring);
* **more layers** (default 7) for (110) surface-energy / work-function
  convergence -- 5 leaves only ~4 relaxing layers once the bottom is frozen;
* **AFM-II moments from the (111) plane index**, with assertions: zero net
  moment, and every nearest-neighbour Ni pair antiparallel (the defining
  type-II superexchange motif);
* writes **POSCAR** (``sort=False`` so the compact ``MAGMOM`` stays valid) **and
  ``slab.extxyz``** (carries the moments into the notebook) plus a ``MAGMOM`` /
  INCAR starter block.

Cheaper alternative: if you already have a *relaxed* small-cell CONTCAR, just
set ``SURFACE_SUPERCELL = (2, 2, 1)`` in the notebook -- ``atoms.repeat`` tiles
the relaxed slab and its AFM pattern, and only the adsorbate region then needs
relaxing.  Rebuild from scratch (this script) when you want a different size,
thickness, or a symmetric slab.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc110, make_supercell
from ase.constraints import FixAtoms
from ase.io import write
from nio_hydroxylation_utils import assign_afm_ii_moments  # same directory

# ----------------------------------------------------------------------------
# knobs
# ----------------------------------------------------------------------------
A_NIO          = 4.1946      # A, experimental rocksalt NiO (or read from a POSCAR)
N_LAYERS       = 7           # physical (110) layers (each has equal Ni + O)
TARGET_MIN_IMG = 20.0        # A, desired minimum in-plane lattice length
VACUUM_TOTAL   = 40.0        # A, total vacuum along the surface normal
                             #   (>= tallest standing ligand + ~15 A; DCZ-4P ~20 A)
FREEZE_LAYERS  = 2           # bottom (110) layers held fixed -- the interface is
                             #   anchored from the bottom (F F F); everything above
                             #   relaxes (T T T)
MU_NI          = 2.0         # bohr magneton, high-spin Ni(2+) d8
OUT_DIR        = Path("inputs")
OUT_STEM       = "NiO_110_AFM"

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
# 2. near-square, AFM-compatible in-plane supercell >= TARGET_MIN_IMG
#    P0 = [[2,-3],[2,3]] is the original 12-fold cell (~12 A); scale it by n.
# ----------------------------------------------------------------------------
la = np.linalg.norm(2 * prim.cell[0] - 3 * prim.cell[1])
n = int(np.ceil(TARGET_MIN_IMG / la))
P = np.array([[2 * n, -3 * n, 0], [2 * n, 3 * n, 0], [0, 0, 1]])
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
    f"LDAU   = .TRUE.\nLDAUTYPE = 2\nLDAUL = 2 -1\nLDAUU = 6.0 0.0\nLDAUJ = 0.0 0.0\n"
    f"LMAXMIX = 4\n"
    f"LDIPOL = .TRUE.\nIDIPOL = 3        # asymmetric slab -> dipole correction\n",
    encoding="utf-8")

print("NiO(110) AFM-II slab")
print(f"  a (bulk)      : {A_NIO} A")
print(f"  layers        : {N_LAYERS}")
print(f"  supercell P   : [[{2*n},{-3*n}],[{2*n},{3*n}]]   ({int(round(np.linalg.det(P)))}-fold)")
print(f"  cell lengths  : {np.round(slab.cell.lengths(), 2)}")
print(f"  cell angles   : {np.round(slab.cell.angles(), 1)}")
print(f"  atoms         : {len(slab)}   ({len(up)} Ni-up, {len(dn)} Ni-down, {len(o_)} O)")
print(f"  frozen (bottom): {len(frozen)} atoms  ({FREEZE_LAYERS} layers, F F F)")
print(f"  min image dist : {min(slab.cell.lengths()[:2]):.1f} A")
print(f"  MAGMOM        : {magmom}")
print(f"  wrote {OUT_DIR / (OUT_STEM + '.POSCAR')} , .extxyz , .INCAR_starter")
