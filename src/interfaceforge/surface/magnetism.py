"""Magnetic initialization and convergence audits for decorated surfaces."""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any

import numpy as np


def compact_magmom(values) -> str:
    moments = np.asarray(values, dtype=float)
    if not len(moments):
        return ""
    pieces: list[str] = []
    start = 0
    while start < len(moments):
        end = start + 1
        while end < len(moments) and np.isclose(moments[end], moments[start]):
            end += 1
        pieces.append(f"{end - start}*{moments[start]:g}")
        start = end
    return " ".join(pieces)


def superexchange_graph(
    atoms,
    *,
    magnetic_species: str,
    bridge_species: str = "O",
    bond_cutoff: float = 2.7,
    linear_dot: float = -0.85,
) -> dict[int, set[int]]:
    """Build edges for nearly linear magnetic-bridge-magnetic paths."""
    from ase.neighborlist import neighbor_list

    symbols = np.asarray(atoms.get_chemical_symbols())
    i_values, j_values, vectors = neighbor_list("ijD", atoms, bond_cutoff)
    by_bridge: dict[int, list[tuple[int, np.ndarray]]] = collections.defaultdict(list)
    for i, j, vector in zip(i_values, j_values, vectors, strict=True):
        if symbols[i] == bridge_species and symbols[j] == magnetic_species:
            norm = float(np.linalg.norm(vector))
            if norm:
                by_bridge[int(i)].append((int(j), np.asarray(vector) / norm))

    magnetic = set(int(i) for i in np.where(symbols == magnetic_species)[0])
    adjacency: dict[int, set[int]] = {index: set() for index in magnetic}
    for bonds in by_bridge.values():
        for left in range(len(bonds)):
            for right in range(left + 1, len(bonds)):
                a, va = bonds[left]
                b, vb = bonds[right]
                if float(np.dot(va, vb)) < linear_dot:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
    return adjacency


def assign_superexchange_afm(
    atoms,
    *,
    magnetic_species: str,
    bridge_species: str = "O",
    moment: float = 2.0,
    bond_cutoff: float = 2.7,
    linear_dot: float = -0.85,
    require_balanced: bool = True,
) -> dict[str, Any]:
    """Two-colour a surface superexchange graph and attach initial moments."""
    adjacency = superexchange_graph(
        atoms,
        magnetic_species=magnetic_species,
        bridge_species=bridge_species,
        bond_cutoff=bond_cutoff,
        linear_dot=linear_dot,
    )
    if not adjacency:
        raise ValueError(f"no {magnetic_species} atoms found for AFM initialization")
    colour: dict[int, int] = {}
    frustrated: list[tuple[int, int]] = []
    for start in sorted(adjacency):
        if start in colour:
            continue
        colour[start] = 0
        queue = collections.deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                wanted = 1 - colour[node]
                if neighbor not in colour:
                    colour[neighbor] = wanted
                    queue.append(neighbor)
                elif colour[neighbor] != wanted:
                    frustrated.append(tuple(sorted((node, neighbor))))
    frustrated = sorted(set(frustrated))
    if frustrated:
        raise ValueError(f"superexchange graph is not bipartite ({len(frustrated)} frustrated edges)")

    moments = np.zeros(len(atoms), dtype=float)
    for index, value in colour.items():
        moments[index] = moment if value == 0 else -moment
    n_up = int(np.sum(moments > 0))
    n_down = int(np.sum(moments < 0))
    net = float(np.sum(moments))
    if require_balanced and (n_up != n_down or not np.isclose(net, 0.0)):
        raise ValueError(f"AFM graph is not spin balanced: {n_up} up / {n_down} down, net {net:+g}")
    atoms.set_initial_magnetic_moments(moments)
    edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return {
        "mode": "superexchange-bipartite",
        "magnetic_species": magnetic_species,
        "bridge_species": bridge_species,
        "moment": moment,
        "up": n_up,
        "down": n_down,
        "net_moment": net,
        "exchange_edges": edges,
        "magmom": compact_magmom(moments),
    }


def parse_outcar_magnetization(path: str | Path) -> np.ndarray | None:
    """Return per-ion total moments from the final ``magnetization (x)`` table."""
    source = Path(path)
    if not source.is_file():
        return None
    rows: list[float] = []
    current: list[float] | None = None
    pattern = re.compile(r"^\s*(\d+)\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+([-+0-9.Ee]+)\s*$")
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "magnetization (x)" in line.lower():
            current = []
            continue
        if current is None:
            continue
        match = pattern.match(line)
        if match:
            current.append(float(match.group(2)))
        elif current and line.strip().startswith("tot"):
            rows = current
            current = None
    return np.asarray(rows, dtype=float) if rows else None


def audit_magnetization(
    atoms,
    final_moments: np.ndarray | None,
    *,
    magnetic_species: str,
    collapse_threshold: float = 0.5,
    minimum_sign_fidelity: float = 0.90,
) -> dict[str, Any]:
    initial = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols())
    indices = np.where(symbols == magnetic_species)[0]
    if not len(indices) or not np.any(initial[indices]):
        return {"status": "NOT_APPLICABLE", "reason": "no initial magnetic ordering"}
    if final_moments is None:
        return {"status": "MISSING", "reason": "no OUTCAR magnetization table"}
    if len(final_moments) != len(atoms):
        return {
            "status": "INVALID",
            "reason": f"OUTCAR has {len(final_moments)} moments for {len(atoms)} atoms",
        }
    final = final_moments[indices]
    reference = initial[indices]
    active = np.abs(final) >= collapse_threshold
    sign_matches = np.sign(final[active]) == np.sign(reference[active])
    fidelity = float(np.mean(sign_matches)) if np.any(active) else 0.0
    collapsed = int(np.sum(~active))
    status = "PASS" if collapsed == 0 and fidelity >= minimum_sign_fidelity else "CHECK"
    return {
        "status": status,
        "magnetic_atoms": int(len(indices)),
        "collapsed": collapsed,
        "sign_fidelity": round(fidelity, 6),
        "final_net_moment": round(float(np.sum(final)), 6),
        "minimum_abs_moment": round(float(np.min(np.abs(final))), 6),
    }
