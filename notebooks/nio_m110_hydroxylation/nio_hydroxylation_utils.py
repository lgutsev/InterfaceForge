"""Toolkit for the NiO(110) hydroxylation notebook.

Build source for the inlined "0. Toolkit" section of
``NiO_m110_hydroxylation.ipynb``. The notebook is self-contained and does NOT
import this; edit here, then rerun the notebook builder. Kept in the repo so
the toolkit stays diff-able and testable.

* Structures are ASE ``Atoms``; pymatgen only for symmetry-equivalent site
  labels (lazy import).
* All minimum-image geometry uses fractional coords + the full cell matrix, so
  non-orthogonal (hexagonal) slabs are correct.  Periodicity is in-plane only
  (a, b); the surface normal (c) is never wrapped.
* Windows-safe: ``pathlib`` throughout, no Slurm / symlinks / shell.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

# --------------------------------------------------------------------------
# constants (placeholder geometry -- refined later by VASP+U relaxation)
# --------------------------------------------------------------------------
D_NI_O_PLACEHOLDER = 2.00     # Ang, capping O above a surface Ni along +z
D_O_H = 0.97                  # Ang, hydroxyl O-H bond length
NI_O_BOND_CUTOFF = 2.55       # Ang, rock-salt NiO first-shell cutoff
HBOND_OO_MIN = 2.50           # Ang, O...O window for a candidate H-bond
HBOND_OO_MAX = 3.30           # Ang
HBOND_HO_MAX = 2.50           # Ang, satisfied H...O contact
HBOND_ANGLE_MIN = 140.0       # deg, satisfied O-H...O angle (near linear)
HH_CLASH = 1.50               # Ang, H...H steric clash threshold
DEFAULT_H_TILT_DEG = 25.0     # isolated hydroxyl: modest outward tilt from +z
D_NI_OP = 2.05                # Ang, target surface-Ni ... phosphonate-O distance

# Hard lower bounds for unintended ligand--slab contacts. These are rejection
# floors, not target bond lengths. Dissociative and multidentate products must
# be constructed explicitly rather than created by a steric-search accident.
CONTACT_MINIMA = {
    ("C", "H"): 1.45,
    ("C", "Ni"): 1.80,
    ("C", "O"): 1.65,
    ("H", "H"): 1.60,
    ("H", "N"): 1.35,
    ("H", "Ni"): 2.10,
    ("H", "O"): 1.45,  # permits a strong O--H...O hydrogen bond
    ("H", "P"): 1.55,
    ("N", "Ni"): 1.75,
    ("N", "O"): 1.55,
    ("Ni", "O"): 1.75,
    ("Ni", "P"): 2.00,
    ("O", "O"): 2.30,
    ("O", "P"): 1.80,
}
DEFAULT_CONTACT_MIN = 1.25


# ==========================================================================
# minimum-image geometry (in-plane periodic, any cell shape)
# ==========================================================================
def _cell_array(cell) -> np.ndarray:
    return np.asarray(getattr(cell, "array", cell), dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def mic_delta(p: np.ndarray, q: np.ndarray, cell) -> np.ndarray:
    """Minimum-image ``p - q`` (Cartesian, Å), wrapping only the in-plane
    lattice vectors a and b -- correct for non-orthogonal cells."""
    C = _cell_array(cell)
    df = np.asarray(p, float) - np.asarray(q, float)
    df = df @ np.linalg.inv(C)
    df[..., :2] -= np.round(df[..., :2])
    return df @ C


def inplane_dmatrix(atoms: Atoms, idx: Sequence[int]) -> np.ndarray:
    """Pairwise in-plane (a-b) minimum-image distance matrix for ``atoms[idx]``."""
    C = _cell_array(atoms.cell)
    frac = atoms.get_scaled_positions()[np.asarray(idx)]
    df = frac[:, None, :] - frac[None, :, :]
    df[..., :2] -= np.round(df[..., :2])
    df[..., 2] = 0.0
    return np.linalg.norm(df @ C, axis=-1)


def _rotmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if float(np.dot(a, b)) > 0 else np.diag([1.0, -1.0, -1.0])
    c = float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def _rot_about_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotmat_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation by ``theta`` rad about unit ``axis`` (Rodrigues)."""
    a = _unit(np.asarray(axis, float))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ==========================================================================
# I/O
# ==========================================================================
def load_structure(path: str | Path) -> Atoms:
    """Read a POSCAR/CONTCAR or extended-xyz slab.  Selective-dynamics ``T/F``
    flags are preserved as an ASE constraint (kept by ``FREEZE_MODE='inherit'``)."""
    path = Path(path)
    atoms = ase_read(path)
    if not atoms.cell.rank:
        raise ValueError(f"{path} has no cell; a periodic slab needs lattice vectors.")
    atoms.pbc = (True, True, True)
    return atoms


def load_molecule(path: str | Path) -> Atoms:
    """Read a molecule (xyz or POSCAR).  Cell / constraints dropped."""
    atoms = ase_read(Path(path))
    atoms.set_cell(None)
    atoms.pbc = False
    atoms.set_constraint()
    atoms.center()
    return atoms


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ==========================================================================
# surface analysis -- exposed / reactive Ni inventory
# ==========================================================================
@dataclass
class SurfaceModel:
    atoms: Atoms
    ni_indices: np.ndarray
    o_indices: np.ndarray
    exposed_ni: np.ndarray            # atom indices, sorted, canonical order
    top_z: float                     # z of the exposed Ni plane
    z_normal: int = 2

    @property
    def n_exposed(self) -> int:
        return int(len(self.exposed_ni))

    def coverage_count(self, fraction: float) -> int:
        return int(round(fraction * self.n_exposed))


def analyse_surface(atoms: Atoms, *, top_layer_tol: float = 0.80,
                    coord_cutoff: float = NI_O_BOND_CUTOFF,
                    require_undercoordinated: bool = True) -> SurfaceModel:
    """Exposed Ni = upper-half Ni that is under-coordinated in O (bulk rock-salt
    Ni is 6-coordinate); mirrors ``_static_exposed_ni_sites``.  Sorted by atom
    index -> a stable canonical selection."""
    z = atoms.positions[:, 2]
    sym = np.array(atoms.get_chemical_symbols())
    ni_idx = np.where(sym == "Ni")[0]
    o_idx = np.where(sym == "O")[0]
    if len(ni_idx) == 0 or len(o_idx) == 0:
        raise ValueError("surface does not contain both Ni and O atoms")

    ni_z = z[ni_idx]
    mid = float(np.median(ni_z))
    top_z = float(ni_z.max())
    top_layer = ni_idx[ni_z >= top_z - top_layer_tol]

    if require_undercoordinated:
        from ase.neighborlist import neighbor_list

        i_list, _ = neighbor_list("ij", atoms, {("Ni", "O"): coord_cutoff})
        coord = np.bincount(i_list, minlength=len(atoms))
        upper = ni_idx[ni_z >= mid]
        under = {int(a) for a in upper if coord[a] < 6}
        exposed = sorted(set(int(a) for a in top_layer) | under)
        exposed = [a for a in exposed if z[a] >= mid]
    else:
        exposed = sorted(int(a) for a in top_layer)

    if not exposed:
        raise ValueError("no exposed upper-surface Ni sites identified")
    return SurfaceModel(atoms=atoms, ni_indices=ni_idx, o_indices=o_idx,
                        exposed_ni=np.array(exposed, dtype=int), top_z=top_z)


def equivalent_site_labels(atoms: Atoms, site_indices: Sequence[int]) -> dict[int, int]:
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        struct = AseAtomsAdaptor.get_structure(atoms)
        eq = SpacegroupAnalyzer(struct, symprec=0.1).get_symmetry_dataset()["equivalent_atoms"]
        return {int(i): int(eq[i]) for i in site_indices}
    except Exception as exc:  # pragma: no cover
        print(f"  [equivalent_site_labels] symmetry analysis skipped: {exc}")
        return {int(i): int(i) for i in site_indices}


# ==========================================================================
# coverage patterns
# ==========================================================================
def select_sites(surface: SurfaceModel, fraction: float, arrangement: str, *,
                 seed: int = 0) -> np.ndarray:
    """``clustered`` = compact island (nearest-neighbour accretion), ``scattered``
    = farthest-point spread, ``all`` / ``none`` = endpoints."""
    exposed = surface.exposed_ni
    n = surface.coverage_count(fraction)
    if n <= 0:
        return np.array([], dtype=int)
    if n >= len(exposed):
        return np.array(sorted(exposed), dtype=int)

    D = inplane_dmatrix(surface.atoms, exposed)
    rng = np.random.default_rng(seed)
    if arrangement in ("all", "none"):
        chosen = list(range(n))
    elif arrangement in ("clustered", "scattered"):
        want_min = arrangement == "clustered"
        chosen = [int(rng.integers(len(exposed)))]
        while len(chosen) < n:
            rest = [k for k in range(len(exposed)) if k not in chosen]
            dmin = [min(D[k, c] for c in chosen) for k in rest]
            chosen.append(rest[int(np.argmin(dmin)) if want_min else int(np.argmax(dmin))])
    else:
        raise ValueError(f"unknown arrangement {arrangement!r}")
    return np.array(sorted(exposed[k] for k in chosen), dtype=int)


# ==========================================================================
# hydroxylation motifs
# ==========================================================================
@dataclass
class Hydroxyl:
    o_pos: np.ndarray
    kind: str                       # "ni_oh" | "lattice_oh"
    parent_ni: int | None = None
    parent_o: int | None = None
    h_pos: np.ndarray | None = None
    h_assigned: bool = False


def build_capped_hydroxide(surface: SurfaceModel, site_indices: Sequence[int], *,
                           d_ni_o: float = D_NI_O_PLACEHOLDER) -> list[Hydroxyl]:
    """Motif A: an O directly above each selected Ni along +z."""
    return [Hydroxyl(o_pos=surface.atoms.positions[ni] + np.array([0.0, 0.0, d_ni_o]),
                     kind="ni_oh", parent_ni=int(ni)) for ni in site_indices]


def build_dissociated_pair(surface: SurfaceModel, site_indices: Sequence[int], *,
                           d_ni_o: float = D_NI_O_PLACEHOLDER,
                           acceptor_search_radius: float = 3.2,
                           acceptor_min_radius: float = 1.5,
                           surface_o_band: float = 1.6,
                           require_complete: bool = True) -> list[Hydroxyl]:
    """Build one water-balanced dissociation pair per selected surface Ni.

    Each pair contains a new ``Ni-OH`` plus a proton on a *distinct* nearby
    surface lattice O.  Surface O atoms in the Ni first coordination shell are
    valid proton acceptors; excluding them incorrectly removes the usual
    nearest-neighbour proton-transfer product on an oxide surface.

    A deterministic augmenting-path matching is used instead of greedy nearest
    assignment.  This avoids silently losing lattice protons at high coverage.
    By default an incomplete matching is an error because a folder labelled
    ``dissoc`` must contain one full H2O equivalent per selected Ni site.
    """
    atoms = surface.atoms
    pos = atoms.positions
    cell = atoms.cell
    o_z = pos[surface.o_indices, 2]
    surf_o = surface.o_indices[o_z >= surface.top_z - surface_o_band]

    sites = [int(i) for i in site_indices]
    hydroxyls = build_capped_hydroxide(surface, sites, d_ni_o=d_ni_o)
    options: list[list[tuple[float, int]]] = []
    for ni in sites:
        cands: list[tuple[float, int]] = []
        for o in surf_o:
            d_ni = float(np.linalg.norm(mic_delta(pos[o], pos[ni], cell)))
            if d_ni < acceptor_min_radius or d_ni > acceptor_search_radius:
                continue
            cands.append((d_ni, int(o)))
        options.append(sorted(cands))

    # Maximum-cardinality bipartite matching.  Visit the most constrained Ni
    # first, while each Ni's candidate order still prefers the closest O.
    matched_o: dict[int, int] = {}

    def augment(site_pos: int, seen_o: set[int]) -> bool:
        for _distance, oxygen in options[site_pos]:
            if oxygen in seen_o:
                continue
            seen_o.add(oxygen)
            previous = matched_o.get(oxygen)
            if previous is None or augment(previous, seen_o):
                matched_o[oxygen] = site_pos
                return True
        return False

    for site_pos in sorted(range(len(sites)), key=lambda i: (len(options[i]), i)):
        augment(site_pos, set())

    assigned = {site_pos: oxygen for oxygen, site_pos in matched_o.items()}
    missing = [sites[i] for i in range(len(sites)) if i not in assigned]
    if missing and require_complete:
        raise ValueError(
            "incomplete dissociated-water motif: could not assign a distinct "
            f"surface lattice O to {len(missing)}/{len(sites)} selected Ni sites "
            f"within {acceptor_min_radius}-{acceptor_search_radius} Å; "
            f"unmatched Ni indices: {missing}"
        )
    if missing:
        print(f"  [build_dissociated_pair] WARNING: incomplete matching for {missing}")
    for site_pos in sorted(assigned):
        oxygen = assigned[site_pos]
        hydroxyls.append(Hydroxyl(o_pos=pos[oxygen].copy(), kind="lattice_oh",
                                  parent_o=oxygen))
    return hydroxyls


# ==========================================================================
# hydrogen-bond graph + greedy H assignment
# ==========================================================================
def assign_hydrogens(hydroxyls: list[Hydroxyl], cell, *, d_oh: float = D_O_H,
                     default_tilt_deg: float = DEFAULT_H_TILT_DEG,
                     default_azimuth_deg: float = 0.0) -> list[tuple[int, int]]:
    """Greedy directed H-bond assignment: candidate O...O edges shortest-first;
    each not-yet-donating hydroxyl points its single H at its best available
    acceptor (donate once, accept many).  A starting guess, not a ground state."""
    n = len(hydroxyls)
    opos = [h.o_pos for h in hydroxyls]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(mic_delta(opos[i], opos[j], cell)))
            if HBOND_OO_MIN <= r <= HBOND_OO_MAX:
                edges.append((r, i, j))
    edges.sort()

    realised: list[tuple[int, int]] = []
    for _r, i, j in edges:
        for a, b in ((i, j), (j, i)):
            if hydroxyls[a].h_assigned:
                continue
            direction = _unit(mic_delta(opos[b], opos[a], cell))
            if direction[2] < np.sin(np.radians(5)):
                direction[2] = np.sin(np.radians(15))
                direction = _unit(direction)
            hydroxyls[a].h_pos = opos[a] + d_oh * direction
            hydroxyls[a].h_assigned = True
            realised.append((a, b))
            break

    th, az = np.radians(default_tilt_deg), np.radians(default_azimuth_deg)
    dflt = _unit(np.array([np.cos(az) * np.sin(th), np.sin(az) * np.sin(th), np.cos(th)]))
    for h in hydroxyls:
        if not h.h_assigned:
            h.h_pos = h.o_pos + d_oh * dflt
    return realised


def override_hydrogen(hydroxyls: list[Hydroxyl], index: int, *, tilt_deg: float,
                      azimuth_deg: float, d_oh: float = D_O_H) -> None:
    th, az = np.radians(tilt_deg), np.radians(azimuth_deg)
    d = np.array([np.cos(az) * np.sin(th), np.sin(az) * np.sin(th), np.cos(th)])
    hydroxyls[index].h_pos = hydroxyls[index].o_pos + d_oh * _unit(d)
    hydroxyls[index].h_assigned = True


@dataclass
class HBondDiagnostic:
    candidate_pairs: int
    satisfied_contacts: int
    hh_clashes: int
    score: float = field(init=False)

    def __post_init__(self):
        self.score = (self.satisfied_contacts / self.candidate_pairs
                      if self.candidate_pairs else 1.0)


def hbond_diagnostic(hydroxyls: list[Hydroxyl], cell) -> HBondDiagnostic:
    n = len(hydroxyls)
    opos = [h.o_pos for h in hydroxyls]
    hpos = [h.h_pos for h in hydroxyls]
    cand = sat = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(mic_delta(opos[i], opos[j], cell)))
            if not (HBOND_OO_MIN <= r <= HBOND_OO_MAX):
                continue
            cand += 1
            for a, b in ((i, j), (j, i)):
                if hpos[a] is None:
                    continue
                oh = mic_delta(hpos[a], opos[a], cell)
                ho = mic_delta(opos[b], hpos[a], cell)
                if np.linalg.norm(ho) > HBOND_HO_MAX:
                    continue
                # Conventional donor-O--H...acceptor-O angle is measured at H:
                # H->donor O is ``-oh`` and H->acceptor O is ``ho``.
                ang = np.degrees(np.arccos(np.clip(np.dot(-_unit(oh), _unit(ho)), -1, 1)))
                if ang >= HBOND_ANGLE_MIN:
                    sat += 1
                    break
    clash = 0
    for i in range(n):
        for j in range(i + 1, n):
            if hpos[i] is None or hpos[j] is None:
                continue
            if np.linalg.norm(mic_delta(hpos[i], hpos[j], cell)) < HH_CLASH:
                clash += 1
    return HBondDiagnostic(cand, sat, clash)


# ==========================================================================
# assembly
# ==========================================================================
def _extend(base: Atoms, extra: Atoms) -> Atoms:
    """``base + extra`` keeping ``base``'s FixAtoms constraint and initial
    magnetic moments (new atoms get moment 0, so an AFM slab's MAGMOM survives
    hydroxylation / ligand decoration)."""
    n0 = len(base)
    cons = base.constraints
    mom = base.get_initial_magnetic_moments()
    out = base + extra
    out.set_constraint(cons)
    if np.any(mom):
        full = np.zeros(len(out))
        full[:n0] = mom
        out.set_initial_magnetic_moments(full)
    return out


def assemble(atoms: Atoms, hydroxyls: list[Hydroxyl]) -> Atoms:
    """bare slab + hydroxyl O and H atoms.  FixAtoms constraint and initial
    magnetic moments preserved (new atoms appended after, moment 0, mobile)."""
    out = atoms.copy()
    add_sym, add_pos = [], []
    for h in hydroxyls:
        if h.kind == "ni_oh":
            add_sym.append("O")
            add_pos.append(h.o_pos)
        if h.h_pos is not None:
            add_sym.append("H")
            add_pos.append(h.h_pos)
    if add_pos:
        out = _extend(out, Atoms(add_sym, positions=np.array(add_pos)))
    out.wrap()
    return out


def sanity_check(bare: Atoms, built: Atoms, hydroxyls: list[Hydroxyl],
                 expected_sites: int, *, min_separation: float = 0.75) -> dict:
    from ase.neighborlist import neighbor_list

    issues: list[str] = []
    oh_lengths = [float(np.linalg.norm(h.h_pos - h.o_pos))
                  for h in hydroxyls if h.h_pos is not None]
    if oh_lengths and not np.allclose(oh_lengths, D_O_H, atol=0.05):
        issues.append(f"O-H bond lengths off target: {np.round(oh_lengths, 3)}")
    d = neighbor_list("d", built, min_separation)
    if len(d):
        issues.append(f"{len(d)} atom pair(s) closer than {min_separation} Å")
    if not np.allclose(bare.cell.array, built.cell.array):
        issues.append("cell changed between bare and built")
    if tuple(built.pbc) != (True, True, True):
        issues.append(f"pbc not (T,T,T): {tuple(built.pbc)}")
    n_ni_oh = sum(1 for h in hydroxyls if h.kind == "ni_oh")
    if n_ni_oh != expected_sites:
        issues.append(f"Ni-OH count {n_ni_oh} != requested {expected_sites}")
    added_H = sum(1 for h in hydroxyls if h.h_pos is not None)
    if len(built) != len(bare) + n_ni_oh + added_H:
        issues.append("atom count != bare + added O + added H")
    return {"ok": not issues, "issues": issues, "n_ni_oh": n_ni_oh,
            "added_O": n_ni_oh, "added_H": added_H, "oh_lengths": oh_lengths}


# ==========================================================================
# phosphonate anchor detection + surface-docked orientation
# ==========================================================================
@dataclass
class PhosphonateAnchor:
    p_index: int
    o_indices: list[int]             # all three anchor O
    oh_o_indices: list[int]          # P-OH  (carry an H)
    eq_o_indices: list[int]          # P=O   (terminal, no H)
    c_index: int
    all_p_indices: list[int]


def _bonded_hydrogens(mol: Atoms, o_index: int, cutoff: float = 1.20) -> list[int]:
    pos, sym = mol.positions, mol.get_chemical_symbols()
    d = np.linalg.norm(pos - pos[o_index], axis=1)
    return [int(j) for j in np.where((d > 0) & (d <= cutoff))[0] if sym[j] == "H"]


def find_phosphonate_anchor(mol: Atoms, *, po_max: float = 1.95, pc_max: float = 2.10,
                            oh_max: float = 1.20, bond_scale=None) -> PhosphonateAnchor:
    """Anchor = a P with exactly 3 O (<= ``po_max`` Å) and 1 C (<= ``pc_max`` Å);
    mirrors ``chemistry.phosphonate_roles``.  Distances are ranked directly (ASE
    ``natural_cutoffs`` over-counts on carbazole phosphonic acids).  Each anchor
    O is classed P-OH (has a bonded H) or P=O.  Multiple phosphonates ->
    lowest-atom-index P; all P indices recorded."""
    sym = np.array(mol.get_chemical_symbols())
    pos = mol.positions
    cands = []
    for i in np.where(sym == "P")[0]:
        d = np.linalg.norm(pos - pos[i], axis=1)
        o = [int(j) for j in np.argsort(d) if sym[j] == "O" and d[j] <= po_max]
        c = [int(j) for j in np.argsort(d) if sym[j] == "C" and d[j] <= pc_max]
        if len(o) == 3 and len(c) == 1:
            cands.append((int(i), sorted(o), c[0]))
    if not cands:
        raise ValueError("no phosphonate group found (need a P with exactly 3 O within "
                         f"{po_max} Å and 1 C within {pc_max} Å)")
    cands.sort()
    p_index, o_indices, c_index = cands[0]
    oh = [o for o in o_indices if _bonded_hydrogens(mol, o, oh_max)]
    eq = [o for o in o_indices if o not in oh]
    return PhosphonateAnchor(p_index, o_indices, oh, eq, c_index,
                             [p for p, _, _ in cands])


def orient_phosphonate(mol: Atoms, anchor: PhosphonateAnchor, *,
                       po_toward_xy: Sequence[float] | None = None,
                       max_body_height: float | None = None,
                       lean_azimuth_deg: float | None = None,
                       max_tilt_deg: float = 80.0) -> Atoms:
    """Orient the molecule as **surface -- phosphonate head -- body**:

    1. rotate so **P -> (centroid of body atoms) points up (+z)** -- the whole
       body stands vertical (centroid, not just the P->C bond, keeps bent /
       bulky / long bis-phosphonate bodies upright); the anchor O then sit
       below P so the head faces the surface;
    2. if there is a P=O and ``po_toward_xy`` is given, spin about the vertical
       through P so the P=O in-plane projection points that way (toward a
       surface hydroxyl H), leaving the P-OH oxygens toward the bare NiO;
    3. if ``max_body_height`` is given and the upright molecule is taller than
       that (P to highest atom, along +z), **tilt about a horizontal axis
       through P** just enough to fit -- a realistic tilted-SAM geometry for a
       slab with limited vacuum.  Lean direction = ``lean_azimuth_deg`` (default
       the way the body already leans); capped at ``max_tilt_deg``.
    """
    out = mol.copy()
    P = out.positions[anchor.p_index].copy()
    body = np.setdiff1d(np.arange(len(out)), [anchor.p_index, *anchor.o_indices])
    body_dir = (out.positions[body].mean(axis=0) - P) if len(body) else \
               (out.positions[anchor.c_index] - P)
    out.positions = (_rotmat(body_dir, np.array([0.0, 0.0, 1.0])) @ (out.positions - P).T).T + P

    if min(out.positions[o, 2] for o in anchor.o_indices) > P[2]:
        print("  [orient_phosphonate] NOTE: every anchor O ended up above P -- "
              "check the molecule / anchor choice.")

    if anchor.eq_o_indices and po_toward_xy is not None:
        d = out.positions[anchor.eq_o_indices[0]] - out.positions[anchor.p_index]
        cur = _unit(np.array([d[0], d[1], 0.0]))
        tgt = _unit(np.array([po_toward_xy[0], po_toward_xy[1], 0.0]))
        if np.linalg.norm(cur) > 1e-6 and np.linalg.norm(tgt) > 1e-6:
            ang = np.arctan2(cur[0] * tgt[1] - cur[1] * tgt[0], np.dot(cur, tgt))
            out.positions = (_rot_about_z(ang) @ (out.positions - P).T).T + P

    if max_body_height is not None:
        rel = out.positions - P
        if rel[:, 2].max() > max_body_height:
            if lean_azimuth_deg is None:
                bc = rel[body].mean(axis=0) if len(body) else rel[anchor.c_index]
                phi = np.arctan2(bc[1], bc[0]) if np.hypot(bc[0], bc[1]) > 1e-6 else 0.0
            else:
                phi = np.radians(lean_azimuth_deg)
            axis = np.array([-np.sin(phi), np.cos(phi), 0.0])

            def top_after(theta):
                return (_rotmat_axis(axis, theta) @ rel.T).T[:, 2].max()

            hi = np.radians(max_tilt_deg)
            if top_after(hi) > max_body_height:
                theta = hi
                print(f"  [orient_phosphonate] WARNING: even a {max_tilt_deg:.0f}° tilt "
                      "leaves the body above the vacuum limit -- enlarge the cell.")
            else:
                lo = 0.0
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if top_after(mid) > max_body_height else (lo, mid)
                theta = hi
            out.positions = (_rotmat_axis(axis, theta) @ rel.T).T + P
    return out


def ligand_tilt_deg(mol_or_positions, anchor: PhosphonateAnchor) -> float:
    """Angle (deg) between P -> body-centroid and +z, for an oriented molecule."""
    pos = getattr(mol_or_positions, "positions", np.asarray(mol_or_positions))
    P = pos[anchor.p_index]
    body = np.setdiff1d(np.arange(len(pos)), [anchor.p_index, *anchor.o_indices])
    v = pos[body].mean(axis=0) - P if len(body) else pos[anchor.c_index] - P
    return float(np.degrees(np.arccos(np.clip(v[2] / (np.linalg.norm(v) + 1e-9), -1, 1))))


def extend_vacuum(atoms: Atoms, new_length: float, *, axis: int = 2) -> Atoms:
    """Return a copy with the ``axis`` lattice vector rescaled to ``new_length``
    (Å), atom Cartesian positions unchanged -- i.e. add vacuum above the slab so
    a tall standing SAM has room.  Only lengthens; a shorter request is ignored."""
    out = atoms.copy()
    C = out.cell.array.copy()
    cur = float(np.linalg.norm(C[axis]))
    if new_length > cur:
        C[axis] *= new_length / cur
        out.set_cell(C, scale_atoms=False)
    return out


def vacuum_fit(structure: Atoms, n_slab: int, *, axis: int = 2,
               min_gap: float = 2.0) -> dict:
    """Check the ligand doesn't run into the slab's periodic image.

    The slab sits in the *middle* of the cell (vacuum on both sides), so the
    real headroom for a standing ligand is the gap between its highest atom and
    the slab's lowest atom shifted up by one cell length -- **not** the distance
    to the ``c`` plane.  ``n_slab`` = number of slab (+ hydroxyl) atoms, i.e. the
    ligand is ``structure[n_slab:]``."""
    c = float(_cell_array(structure.cell)[axis, axis])
    slab_lo = float(structure.positions[:n_slab, axis].min())
    lig_hi = float(structure.positions[n_slab:, axis].max())
    gap = (slab_lo + c) - lig_hi
    if gap < min_gap:
        print(f"  [vacuum_fit] WARNING: ligand top at {lig_hi:.1f} Å is only {gap:.1f} Å "
              f"below the slab's periodic image (slab bottom {slab_lo:.1f} + c {c:.1f}); "
              f"want >= {min_gap}.  Enlarge the cell along the surface normal.")
    return {"cell_c": round(c, 3), "slab_bottom_z": round(slab_lo, 3),
            "ligand_top_z": round(lig_hi, 3), "vacuum_gap": round(gap, 3),
            "fits_vacuum": gap >= min_gap}


def _min_gap(a_pos: np.ndarray, b_pos: np.ndarray, cell) -> float:
    d = mic_delta(a_pos[:, None, :], b_pos[None, :, :], cell)
    return float(np.linalg.norm(d, axis=-1).min())


@dataclass(frozen=True)
class ContactDiagnostic:
    """Most limiting ligand--slab atom pair relative to its chemical floor."""

    ok: bool
    min_margin: float
    distance: float
    cutoff: float
    ligand_index: int
    slab_index: int
    ligand_symbol: str
    slab_symbol: str


def _contact_cutoff(ligand_symbol: str, slab_symbol: str,
                    default_min: float = DEFAULT_CONTACT_MIN) -> float:
    key = tuple(sorted((str(ligand_symbol), str(slab_symbol))))
    return float(CONTACT_MINIMA.get(key, default_min))


def _chemical_contact_diagnostic(
    ligand_positions: np.ndarray,
    ligand_symbols: Sequence[str],
    slab_positions: np.ndarray,
    slab_symbols: Sequence[str],
    cell,
    *,
    default_min: float = DEFAULT_CONTACT_MIN,
) -> ContactDiagnostic:
    """Evaluate every ligand--slab pair using species-aware rejection floors."""
    delta = mic_delta(
        np.asarray(ligand_positions)[:, None, :],
        np.asarray(slab_positions)[None, :, :],
        cell,
    )
    distances = np.linalg.norm(delta, axis=-1)
    cutoffs = np.array(
        [
            [_contact_cutoff(ls, ss, default_min) for ss in slab_symbols]
            for ls in ligand_symbols
        ],
        dtype=float,
    )
    margins = distances - cutoffs
    ligand_i, slab_i = np.unravel_index(int(np.argmin(margins)), margins.shape)
    margin = float(margins[ligand_i, slab_i])
    return ContactDiagnostic(
        ok=margin >= -1e-8,
        min_margin=margin,
        distance=float(distances[ligand_i, slab_i]),
        cutoff=float(cutoffs[ligand_i, slab_i]),
        ligand_index=int(ligand_i),
        slab_index=int(slab_i),
        ligand_symbol=str(ligand_symbols[ligand_i]),
        slab_symbol=str(slab_symbols[slab_i]),
    )


def chemical_contact_diagnostic(structure: Atoms, n_slab: int, *,
                                default_min: float = DEFAULT_CONTACT_MIN
                                ) -> ContactDiagnostic:
    """Public chemistry-aware contact audit for an assembled slab + ligand."""
    if not 0 < int(n_slab) < len(structure):
        raise ValueError("n_slab must split a nonempty slab from a nonempty ligand")
    symbols = structure.get_chemical_symbols()
    return _chemical_contact_diagnostic(
        structure.positions[n_slab:],
        symbols[n_slab:],
        structure.positions[:n_slab],
        symbols[:n_slab],
        structure.cell,
        default_min=default_min,
    )


def binding_oxygen(anchor: PhosphonateAnchor) -> int:
    """Choose the neutral phosphonate O used for initial molecular Ni binding.

    Prefer the terminal P=O oxygen.  Explicit deprotonated mono-/bidentate
    products are a separate chemical-state generator and must not be implied by
    silently deleting an acid proton here.
    """
    return int(anchor.eq_o_indices[0] if anchor.eq_o_indices else anchor.o_indices[0])


def place_ligand(built_surface: Atoms, oriented_mol: Atoms, anchor: PhosphonateAnchor,
                 xy: Sequence[float], *, ni_plane_z: float, oh_height: float = D_NI_OP,
                 anchor_o_index: int | None = None,
                 min_clearance: float = DEFAULT_CONTACT_MIN,
                 max_azimuth_adjust_deg: float = 180.0,
                 max_contact_tilt_deg: float = 40.0) -> tuple[Atoms, float]:
    """Dock an oriented molecule with one chosen anchor O directly over ``xy``.

    The selected O is placed ``oh_height`` Å above the exposed-Ni plane, so a
    target exposed Ni at ``xy`` starts with the requested Ni--O distance.  This
    replaces the old P-over-Ni placement, whose lateral P--O offset produced
    nominally bound cases with actual Ni--O distances of 2.3--3.7 Å.

    Steric clashes are resolved by a deterministic rigid-body orientation
    search about the *fixed binding O*, first through the full azimuth and then
    with modest contact tilts. Candidate structures are accepted using
    species-aware contact floors, so accidental 1.5 Å Ni--H or 1.3 Å Ni--O
    contacts cannot pass a generic overlap test. The Ni--O anchor is never
    destroyed by lifting the molecule. If no chemically clean bound
    orientation exists, generation stops.

    The molecule is brought into the cell as a rigid body (whole lattice
    vectors only) -- never ``wrap()``, which would tear it across the boundary.
    Returns ``(structure, contact_tilt_deg)``; slab constraint preserved."""
    mol = oriented_mol.copy()
    bind_o = binding_oxygen(anchor) if anchor_o_index is None else int(anchor_o_index)
    if bind_o not in anchor.o_indices:
        raise ValueError(f"anchor_o_index {bind_o} is not one of {anchor.o_indices}")
    mol.positions[:, :2] += np.asarray(xy) - mol.positions[bind_o, :2]
    mol.positions[:, 2] += (ni_plane_z + oh_height) - mol.positions[bind_o, 2]

    slab_pos = built_surface.positions
    slab_symbols = built_surface.get_chemical_symbols()
    mol_symbols = mol.get_chemical_symbols()
    pivot = mol.positions[bind_o].copy()
    base = mol.positions.copy()
    best_positions = base
    best_clearance = _min_gap(base, slab_pos, built_surface.cell)
    best_contact = _chemical_contact_diagnostic(
        base, mol_symbols, slab_pos, slab_symbols, built_surface.cell,
        default_min=min_clearance,
    )
    best_tilt = 0.0

    az_step = 15.0
    azimuths = [0.0]
    for angle in np.arange(az_step, max_azimuth_adjust_deg + 0.1, az_step):
        azimuths.extend([float(angle), -float(angle)])

    def consider(positions: np.ndarray, tilt: float) -> None:
        nonlocal best_positions, best_clearance, best_contact, best_tilt
        clearance = _min_gap(positions, slab_pos, built_surface.cell)
        contact = _chemical_contact_diagnostic(
            positions, mol_symbols, slab_pos, slab_symbols, built_surface.cell,
            default_min=min_clearance,
        )
        if (contact.min_margin, clearance) > (best_contact.min_margin, best_clearance):
            best_positions = positions
            best_clearance = clearance
            best_contact = contact
            best_tilt = tilt

    for azimuth in azimuths:
        rz = _rot_about_z(np.radians(azimuth))
        spun = (rz @ (base - pivot).T).T + pivot
        consider(spun, 0.0)
        if best_contact.ok:
            break

    if not best_contact.ok:
        # Tilt directions span the in-plane circle; every rotation is about the
        # binding O, so its requested Ni--O distance remains exactly fixed.
        for tilt in np.arange(5.0, max_contact_tilt_deg + 0.1, 5.0):
            for axis_azimuth in np.arange(0.0, 360.0, 30.0):
                phi = np.radians(axis_azimuth)
                axis = np.array([np.cos(phi), np.sin(phi), 0.0])
                rt = _rotmat_axis(axis, np.radians(tilt))
                for azimuth in azimuths:
                    rz = _rot_about_z(np.radians(azimuth))
                    candidate = (rt @ (rz @ (base - pivot).T)).T + pivot
                    consider(candidate, float(tilt))
            if best_contact.ok:
                break

    if not best_contact.ok:
        raise ValueError(
            "no chemically valid bound ligand orientation found with the binding O "
            f"fixed; limiting contact is ligand {best_contact.ligand_symbol}"
            f"[{best_contact.ligand_index}]--slab {best_contact.slab_symbol}"
            f"[{best_contact.slab_index}] at {best_contact.distance:.3f} Å "
            f"(minimum {best_contact.cutoff:.3f} Å, margin "
            f"{best_contact.min_margin:.3f} Å)"
        )
    mol.positions = best_positions

    C = _cell_array(built_surface.cell)
    pfrac = mol.positions[anchor.p_index] @ np.linalg.inv(C)
    mol.positions -= np.array([np.floor(pfrac[0]), np.floor(pfrac[1]), 0.0]) @ C

    return _extend(built_surface.copy(), mol), round(best_tilt, 2)


# ==========================================================================
# selective dynamics -- partial slab freezing
# ==========================================================================
def detect_layers(atoms: Atoms, *, axis: int = 2, layer_tol: float = 0.60) -> list[np.ndarray]:
    z = atoms.positions[:, axis]
    order = np.argsort(z)
    layers, current = [], [order[0]]
    for prev, idx in zip(order, order[1:], strict=False):
        if z[idx] - z[prev] < layer_tol:
            current.append(idx)
        else:
            layers.append(np.array(current))
            current = [idx]
    layers.append(np.array(current))
    return layers


def freeze_outside_window(atoms: Atoms, lo: float, hi: float, *, axis: int = 2,
                          only_symbols: Sequence[str] | None = None,
                          also_freeze: Iterable[int] | None = None,
                          report: bool = True) -> Atoms:
    """Fix every atom whose Cartesian ``axis`` coordinate is ``< lo`` or ``> hi``
    -- the direct equivalent of the ``atom.x``-window freezer.  ``hi`` above the
    cell -> bottom-only; lower it -> also pin the top surface."""
    from ase.constraints import FixAtoms

    out = atoms.copy()
    coord = out.positions[:, axis]
    mask = (coord < lo) | (coord > hi)
    if only_symbols is not None:
        mask &= np.isin(np.array(out.get_chemical_symbols()), list(only_symbols))
    if also_freeze is not None:
        mask[list(also_freeze)] = True
    out.set_constraint(FixAtoms(indices=np.where(mask)[0].tolist()))
    if report:
        free = coord[~mask]
        span = f"{free.min():.2f}-{free.max():.2f} Å" if len(free) else "(none!)"
        print(f"  freeze_outside_window(axis={axis}, {lo}-{hi}): "
              f"{int(mask.sum())}/{len(out)} fixed; mobile span {span}")
    return out


def freeze_bottom(atoms: Atoms, *, mode: str = "layers", value: float = 1.0,
                  axis: int = 2, only_symbols: Sequence[str] | None = ("Ni", "O"),
                  also_freeze: Iterable[int] | None = None, layer_tol: float = 0.60,
                  report: bool = True) -> Atoms:
    """Fix the bottom of the slab: ``mode`` = ``layers`` (bottom N detected
    layers) | ``thickness`` (Å band) | ``fraction`` (scaled coord) | ``count``."""
    from ase.constraints import FixAtoms

    out = atoms.copy()
    pos = out.positions[:, axis]
    if mode == "thickness":
        mask = pos <= pos.min() + value
    elif mode == "fraction":
        mask = out.get_scaled_positions()[:, axis] < value
    elif mode == "layers":
        layers = detect_layers(out, axis=axis, layer_tol=layer_tol)
        keep = np.concatenate(layers[: int(value)]) if int(value) else np.array([], int)
        mask = np.zeros(len(out), bool)
        mask[keep] = True
    elif mode == "count":
        mask = np.zeros(len(out), bool)
        mask[np.argsort(pos)[: int(value)]] = True
    else:
        raise ValueError(f"unknown freeze mode {mode!r}")
    if only_symbols is not None:
        mask &= np.isin(np.array(out.get_chemical_symbols()), list(only_symbols))
    if also_freeze is not None:
        mask[list(also_freeze)] = True
    frozen = np.where(mask)[0]
    out.set_constraint(FixAtoms(indices=frozen.tolist()))
    if report:
        sym = np.array(out.get_chemical_symbols())
        by = {s: int((sym[frozen] == s).sum()) for s in sorted(set(sym[frozen]))}
        tail = (f"free z {pos[~mask].min():.2f}-{pos[~mask].max():.2f} Å"
                if len(frozen) < len(out) else "ALL fixed (check value)")
        print(f"  freeze_bottom[{mode}={value}]: {len(frozen)}/{len(out)} fixed ({by}); {tail}")
    return out


def frozen_free_indices(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    frozen: set[int] = set()
    for c in atoms.constraints:
        idx = getattr(c, "index", getattr(c, "a", None))
        if idx is not None:
            frozen.update(int(i) for i in np.atleast_1d(idx))
    fr = np.array(sorted(frozen), dtype=int)
    fe = np.array([i for i in range(len(atoms)) if i not in frozen], dtype=int)
    return fr, fe


# ==========================================================================
# magnetism + runnable VASP inputs
# ==========================================================================
def magmom_from_incar(incar_path: str | Path, n_atoms: int, *, repeats: int = 1) -> np.ndarray:
    """Per-atom moment array from a template INCAR's ``MAGMOM = n*v n*v …`` line,
    tiled ``repeats`` times (for an in-plane supercell -- ``atoms.repeat`` keeps
    the original atom order within each image block).  Length is checked."""
    text = Path(incar_path).read_text()
    m = re.search(r"^\s*MAGMOM\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"no MAGMOM line in {incar_path}")
    vals: list[float] = []
    for tok in m.group(1).split():
        if "*" in tok:
            k, v = tok.split("*")
            vals += [float(v)] * int(k)
        else:
            vals.append(float(tok))
    base = n_atoms // repeats
    if len(vals) != base:
        raise ValueError(f"MAGMOM has {len(vals)} entries; the 1x1 slab has {base} atoms")
    return np.array(vals * repeats, dtype=float)


def assign_afm_ii_moments(atoms: Atoms, *, ni_o_cut: float = 2.7, mu: float = 2.0,
                          linear_dot: float = -0.85, check: bool = True,
                          a_nio=None) -> Atoms:
    """Set AFM-II moments in place by 2-colouring the **Ni-O-Ni 180° superexchange
    graph**: for every lattice O, each pair of its bonded Ni whose bond vectors
    are nearly antiparallel (``dot < linear_dot``) must have opposite spin.
    Purely angular -- frame- and lattice-spacing-independent, and robust to
    surface relaxation.  Non-Ni atoms get 0; survives ``repeat`` / decoration."""
    import collections

    from ase.neighborlist import neighbor_list

    sym = np.array(atoms.get_chemical_symbols())
    ni_set = set(int(x) for x in np.where(sym == "Ni")[0])
    if not ni_set:
        return atoms

    i, j, D = neighbor_list("ijD", atoms, {("O", "Ni"): ni_o_cut})
    by_o: dict[int, list] = collections.defaultdict(list)
    for a, b, d in zip(i, j, D, strict=True):
        if sym[a] == "O" and sym[b] == "Ni":
            by_o[int(a)].append((int(b), d / (np.linalg.norm(d) + 1e-12)))

    adj = collections.defaultdict(set)
    for bonds in by_o.values():
        for x in range(len(bonds)):
            for y in range(x + 1, len(bonds)):
                (na, va), (nb, vb) = bonds[x], bonds[y]
                if float(np.dot(va, vb)) < linear_dot:
                    adj[na].add(nb)
                    adj[nb].add(na)

    colour: dict[int, int] = {}
    frustrated = 0
    for start in sorted(ni_set):
        if start in colour:
            continue
        colour[start] = 0
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    q.append(v)
                elif colour[v] == colour[u]:
                    frustrated += 1
    m = np.zeros(len(atoms))
    for k, c in colour.items():
        m[k] = mu if c == 0 else -mu
    atoms.set_initial_magnetic_moments(m)
    if check:
        bad = sum(1 for u in adj for v in adj[u] if u < v and m[u] * m[v] > 0)
        net = float(m.sum())
        ok = bad == 0 and frustrated == 0
        print(f"  AFM-II: {int((m > 0).sum())} up / {int((m < 0).sum())} down, "
              f"net {net:+.1f} uB, {bad} parallel superexchange pairs  "
              f"[{'OK' if ok else 'CHECK'}]"
              + ("" if abs(net) < 1e-6 else "  (cell not spin-balanced -> set NUPDOWN)"))
    return atoms


def ensure_afm_moments(atoms: Atoms, *, template_incar: str | Path | None = None,
                       repeats: int = 1, mu: float = 2.0) -> str:
    """Ensure a spin-balanced AFM initialization and return its provenance.

    This helper is deliberately used by both the interactive and batch paths;
    previously the batch path reloaded the slab with zero moments and exported
    the template's unrelated 165-entry ``MAGMOM`` unchanged.
    """
    sym = np.array(atoms.get_chemical_symbols())
    ni = np.where(sym == "Ni")[0]
    if not len(ni):
        return "not-applicable"

    moments = atoms.get_initial_magnetic_moments()
    source = "input structure"
    if not np.any(moments):
        try:
            if template_incar is None:
                raise ValueError("no template INCAR supplied")
            moments = magmom_from_incar(template_incar, len(atoms), repeats=repeats)
            atoms.set_initial_magnetic_moments(moments)
            source = f"template INCAR {Path(template_incar)}"
        except Exception as exc:
            print(f"  moments: {exc} -> deriving AFM-II from geometry")
            assign_afm_ii_moments(atoms, mu=mu)
            moments = atoms.get_initial_magnetic_moments()
            source = "geometry-derived AFM-II"

    ni_mom = np.asarray(moments)[ni]
    n_up, n_down = int((ni_mom > 0).sum()), int((ni_mom < 0).sum())
    if n_up + n_down != len(ni):
        raise ValueError(f"{len(ni) - n_up - n_down} Ni atoms have zero initial moment")
    if n_up != n_down or abs(float(ni_mom.sum())) > 1e-6:
        raise ValueError(
            f"AFM initialization is not spin-balanced: {n_up} up / {n_down} down, "
            f"net {float(ni_mom.sum()):+.3f} uB"
        )
    return source


def order_by_element(atoms: Atoms) -> Atoms:
    """Stable sort atoms alphabetically by chemical symbol -- exactly the order
    ``ase.io.write(format='vasp', sort=True)`` uses.  Ni-up stays before Ni-down
    (stable within an element), and the ``FixAtoms`` constraint + initial
    magnetic moments are carried through.  Doing it here (then writing with
    ``sort=False``) keeps the POSCAR, MAGMOM, and LDAU tags in one agreed order
    and gives an external POTCAR generator an unambiguous species order.  This
    matters because ``LDAUL/LDAUU/LDAUJ`` and ``MAGMOM``
    are *per POSCAR species, in POSCAR order*."""
    sym = np.array(atoms.get_chemical_symbols())
    return atoms[np.argsort(sym, kind="stable")]


def _template_ueff(incar_text: str, l_for: str = "2") -> float | None:
    """Ueff for the (only) L=``l_for`` species in a template INCAR's LDAU lines."""
    ll = re.search(r"^\s*LDAUL\s*=\s*(.+)$", incar_text, re.MULTILINE)
    uu = re.search(r"^\s*LDAUU\s*=\s*(.+)$", incar_text, re.MULTILINE)
    if not (ll and uu):
        return None
    lt, ut = ll.group(1).split(), uu.group(1).split()
    for k, tok in enumerate(lt):
        if tok == l_for and k < len(ut):
            return float(ut[k])
    return None


def write_run_inputs(case_dir: str | Path, structure: Atoms, template_dir: str | Path, *,
                     system_name: str | None = None,
                     incar_overrides: dict | None = None) -> dict:
    """Make ``case_dir`` a runnable VASP optimization folder from a template run:

    * **KPOINTS**, ``runvasp.sh`` / ``run.slurm`` -- copied verbatim.
    * **INCAR** -- copied with ``MAGMOM`` rewritten for this structure's atom
      count/order (``compact_magmom``), ``SYSTEM`` set to the case name, and any
      ``incar_overrides`` applied (e.g. ``{"ISIF": "2"}`` for a decorated slab).
    POTCAR generation is intentionally out of scope.  The exported POSCAR is
    element ordered and the matching species order is recorded, allowing the
    project's standard POTCAR generator to remain the single source of truth.
    """
    import shutil

    tdir, case_dir = Path(template_dir), Path(case_dir)
    notes: dict = {}

    if not tdir.is_dir():
        print(f"  [write_run_inputs] WARNING: RUN_TEMPLATE {tdir} does not exist -- "
              "no INCAR / KPOINTS written. Set RUN_TEMPLATE in §1.")
        return {"error": f"template dir missing: {tdir}"}
    for want in ("INCAR", "KPOINTS"):
        if not (tdir / want).is_file():
            print(f"  [write_run_inputs] WARNING: RUN_TEMPLATE has no {want}.")

    for fname in ("KPOINTS", "runvasp.sh", "run.slurm", "run.sh"):
        if (tdir / fname).is_file():
            shutil.copy2(tdir / fname, case_dir / fname)

    # POSCAR species order (the structure is already element-ordered by export_case)
    elements = list(dict.fromkeys(structure.get_chemical_symbols()))

    if (tdir / "INCAR").is_file():
        incar_text = (tdir / "INCAR").read_text()
        mom = structure.get_initial_magnetic_moments()
        if "Ni" in elements and not np.any(mom):
            raise ValueError(
                "refusing to export a Ni-containing VASP run without explicit "
                "initial magnetic moments; call ensure_afm_moments() first"
            )
        magmom = compact_magmom(mom) if np.any(mom) else None
        # DFT+U rebuilt for THIS case's species order: Ni -> (L=2, U=Ueff), else (-1, 0)
        ueff = _template_ueff(incar_text)
        if ueff is not None:
            ldaul = {"LDAUL": " ".join("2" if e == "Ni" else "-1" for e in elements),
                     "LDAUU": " ".join(f"{ueff:g}" if e == "Ni" else "0.0" for e in elements),
                     "LDAUJ": " ".join("0.0" for _ in elements)}
        else:
            ldaul = {}
        ov = {k.upper(): str(v) for k, v in (incar_overrides or {}).items()}
        done = set()
        out = []
        for ln in incar_text.splitlines():
            key = ln.split("=", 1)[0].strip().upper() if "=" in ln else ""
            if key == "SYSTEM" and system_name:
                out.append(f"SYSTEM = {system_name}")
                done.add("SYSTEM")
            elif key == "MAGMOM" and magmom is not None:
                out.append(f"MAGMOM = {magmom}")
                done.add("MAGMOM")
            elif key in ldaul:
                out.append(f"{key} = {ldaul[key]}")
                done.add(key)
            elif key in ov:
                out.append(f"{key} = {ov[key]}")
                done.add(key)
            elif ln.strip().startswith("# Dudarev DFT+U: species order"):
                out.append(f"# Dudarev DFT+U: species order {' '.join(elements)}")
            elif ln.strip().startswith("# C18 H22 N1 |"):
                out.append(f"# Per-atom AFM initialization for {len(structure)} atoms")
            else:
                out.append(ln)
        for extra in (ldaul, ov):
            for key, val in extra.items():
                if key not in done:
                    out.append(f"{key} = {val}")
                    done.add(key)
        if magmom is not None and "MAGMOM" not in done:
            out.append(f"MAGMOM = {magmom}")
        (case_dir / "INCAR").write_text("\n".join(out) + "\n", encoding="utf-8")
        notes["incar"] = ("rewritten: MAGMOM, LDAUL/U/J (species order "
                          f"{elements}), SYSTEM" + (", " + ",".join(ov) if ov else ""))
    notes["species_order"] = elements
    notes["potcar"] = "not generated; use the project POTCAR generator with POSCAR species order"
    return notes


def dipole_center(atoms: Atoms) -> str:
    """Mass-centre DIPOL tag in direct coordinates for an ``IDIPOL = 3`` slab."""
    scaled = np.asarray(atoms.get_center_of_mass(scaled=True), dtype=float)
    return f"0.5 0.5 {scaled[2] % 1.0:.8f}"


def slab_incar_overrides(atoms: Atoms, *, decorated: bool) -> dict[str, str]:
    """Safe cell/dipole overrides for a vacuum-containing slab optimization."""
    overrides = {"ISIF": "2"}
    if decorated:
        overrides.update({"LDIPOL": ".TRUE.", "IDIPOL": "3", "DIPOL": dipole_center(atoms)})
    else:
        overrides["LDIPOL"] = ".FALSE."
    return overrides


# ==========================================================================
# naming + export
# ==========================================================================
MOTIF_TAG = {"capped": "capped", "dissoc": "dissoc"}
BASE_NAME = "NiO_m110_Big_U46"


def case_name(fraction: float, pattern_id: str, motif: str | None, *,
              ligand: str | None = None, anchor_position: str | None = None,
              base_name: str = BASE_NAME) -> str:
    """Leaf run-directory name.

    ``NiO_m110_Big_U46``                                  bare pristine (0 %)
    ``NiO_m110_Big_U46_Me4PACz``                          pristine + ligand (matches
                                                          the hand-built references)
    ``NiO_m110_Big_U46_OH50_clustered_capped``            hydroxylated
    ``NiO_m110_Big_U46_OH50_clustered_capped_Me4PACz_boundary``   + ligand
    """
    if fraction == 0:
        return base_name if not ligand else f"{base_name}_{ligand.replace('-', '')}"
    parts = [base_name, f"OH{int(round(fraction * 100))}", pattern_id]
    if motif:
        parts.append(MOTIF_TAG[motif])
    if ligand:
        parts.append(ligand.replace("-", ""))
        if anchor_position:
            parts.append(anchor_position)
    return "_".join(parts)


def case_subdir(fraction: float, pattern_id: str = "", motif: str | None = None) -> str:
    """One directory per **OH coverage** -- every case (all patterns, motifs, and
    their ligand-decorated variants) sits directly under it:

    ``generated/OH0/``   ``generated/OH25/``   ...   ``generated/OH100/``
    """
    return f"OH{int(round(fraction * 100))}"


def provenance_stamp() -> dict:
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform()}


def compact_magmom(moments) -> str:
    """VASP compact ``MAGMOM`` string for a moment array, e.g.
    ``60*2.0 60*-2.0 66*0.0`` -- run-length encoded in atom order (so it matches
    a ``sort_poscar=False`` POSCAR; added OH / ligand atoms trail as ``N*0.0``)."""
    m = np.round(np.asarray(moments, float), 3)
    out, i = [], 0
    while i < len(m):
        j = i
        while j < len(m) and m[j] == m[i]:
            j += 1
        out.append(f"{j - i}*{m[i]:g}")
        i = j
    return " ".join(out)


def export_case(out_root: str | Path, name: str, structure: Atoms, provenance: dict, *,
                subdir: str = "", run_template: str | Path | None = None,
                incar_overrides: dict | None = None,
                template_dir: str | Path | None = None,
                sort_poscar: str = "element") -> Path:
    """Write a **runnable VASP optimization folder**.

    The structure is written **element-ordered** (alphabetical, ``sort=True``
    order) but via an explicit stable sort here -- so the POSCAR, MAGMOM, and
    the per-species ``LDAUL/LDAUU/LDAUJ`` all agree on one order.  Ni-up stays
    before Ni-down.  POTCAR is deliberately not generated.  Pass
    ``sort_poscar="keep"`` to disable sorting.

    ``run_template`` -- a directory with INCAR / KPOINTS from a working
    run; ``write_run_inputs`` copies KPOINTS, rewrites INCAR (MAGMOM + LDAU tags
    for *this* case's species order, SYSTEM, and ``incar_overrides``).
    """
    import shutil

    if sort_poscar == "element":
        structure = order_by_element(structure)

    case_dir = Path(out_root) / subdir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    provenance = {**provenance, "run_dir": str(Path(subdir) / name).replace("\\", "/")}
    poscar = case_dir / "POSCAR"
    ase_write(poscar, structure, format="vasp", direct=True, sort=False)
    ase_write(case_dir / "structure.xyz", structure)

    if run_template is not None:
        provenance["run_inputs"] = write_run_inputs(
            case_dir, structure, run_template, system_name=name,
            incar_overrides=incar_overrides)
        template_dir = None

    fr, fe = frozen_free_indices(structure)
    provenance = {**provenance, "selective_dynamics": {
        "frozen_atom_count": int(len(fr)), "free_atom_count": int(len(fe)),
        "frozen_indices": fr.tolist()}}
    _mom = structure.get_initial_magnetic_moments()
    if np.any(_mom):
        provenance["magmom"] = compact_magmom(_mom)
    if template_dir:
        for fname in ("INCAR", "KPOINTS", "runvasp.sh", "run.slurm"):
            src = Path(template_dir) / fname
            if src.is_file():
                shutil.copy2(src, case_dir / fname)
    provenance = {**provenance, "poscar_sha256": sha256_file(poscar),
                  "n_atoms": len(structure), "formula": structure.get_chemical_formula()}
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return case_dir
