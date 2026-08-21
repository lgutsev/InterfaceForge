"""Optional adapter for RegFGW registry pre-screening, and a comparison
report checking whether it preserved the true low-energy registries.

RegFGW (https://github.com/YuxuanTang2002/RegFGW) uses a fused
Gromov-Wasserstein distance plus Bayesian optimization to cheaply rank
candidate interface registries (lateral stacking offsets) without relaxing
every one -- the same registry space InterMat's `displacement_interval`
already sweeps exhaustively (see intermat.py).

This module wraps RegFGW's documented `regfgw_coherent` CLI as a subprocess
rather than importing it as a library: no stable Python API is documented,
and its output schema has not been verified against a real run in this
environment. It does not parse RegFGW's output files -- inspect
`output_dir` after `run_regfgw_optimize` returns and adapt the resulting
ranked registry list into the small CSV `compare_registry_selection`
expects. The comparison math itself needs no such assumption and is fully
testable on its own.
"""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import DependencyError, SafetyError

ADAPTER_ID = "interfaceforge.regfgw"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def regfgw_status() -> dict[str, Any]:
    """Report optional dependency availability without importing regfgw."""

    installed = importlib.util.find_spec("regfgw") is not None
    executable_path = shutil.which("regfgw_coherent")
    return {
        "adapter": ADAPTER_ID,
        "available": installed and executable_path is not None,
        "regfgw_version": _package_version("regfgw"),
        "executable_found": executable_path is not None,
        "executable_path": executable_path,
        "install": "pip install 'interfaceforge[regfgw]'",
        "output_schema_verified": False,
        "note": (
            "RegFGW's output schema has not been verified against a real run in "
            "this repo. run_regfgw_optimize does not parse it; compare "
            "run_regfgw_optimize's output_dir by hand into the top-k CSV "
            "compare_registry_selection expects before trusting this path."
        ),
    }


def run_regfgw_optimize(
    substrate: str | Path,
    film: str | Path,
    output_dir: str | Path,
    *,
    embedding: str | Path | None = None,
    budget: int = 3,
    max_miller_idx: int | None = None,
    substrate_layers: int | None = None,
    film_layers: int | None = None,
    gap: float | None = None,
    vacuum: float | None = None,
    executable: str = "regfgw_coherent",
) -> dict[str, Any]:
    """Run ``regfgw_coherent --mode optimize`` as a subprocess.

    Returns the captured command/exit status/stdout/stderr rather than a
    parsed result -- see the module docstring for why. Raises SafetyError
    on a nonzero exit rather than silently returning a failed run.
    """

    if shutil.which(executable) is None:
        raise DependencyError(
            f"{executable} was not found on PATH. Install with: "
            "pip install 'interfaceforge[regfgw]'"
        )
    substrate_path = Path(substrate).resolve()
    film_path = Path(film).resolve()
    if not substrate_path.is_file():
        raise SafetyError(f"Substrate structure not found: {substrate_path}")
    if not film_path.is_file():
        raise SafetyError(f"Film structure not found: {film_path}")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "--mode", "optimize",
        "--substrate", str(substrate_path),
        "--film", str(film_path),
        "--out-dir", str(output_path),
        "--budget", str(int(budget)),
    ]
    if embedding is not None:
        embedding_path = Path(embedding).resolve()
        if not embedding_path.is_file():
            raise SafetyError(f"Embedding file not found: {embedding_path}")
        command += ["--embedding", str(embedding_path)]
    if max_miller_idx is not None:
        command += ["--max-miller-idx", str(int(max_miller_idx))]
    if substrate_layers is not None:
        command += ["--substrate-layers", str(int(substrate_layers))]
    if film_layers is not None:
        command += ["--film-layers", str(int(film_layers))]
    if gap is not None:
        command += ["--gap", str(float(gap))]
    if vacuum is not None:
        command += ["--vacuum", str(float(vacuum))]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "output_dir": str(output_path),
        "output_schema_verified": False,
    }
    if result.returncode != 0:
        raise SafetyError(
            f"{executable} exited with status {result.returncode}. "
            f"stderr (tail): {result.stderr[-2000:]}"
        )
    return payload


def compare_registry_selection(
    topk_source: str | Path,
    exhaustive_source: str | Path,
    output: str | Path,
    *,
    id_column: str = "registry_id",
    energy_column: str = "work_of_adhesion_ev_a2",
    lower_energy_is_better: bool = False,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Check whether a cheap top-k registry search preserved the true best ones.

    ``topk_source``: a CSV with at least ``id_column``, one row per registry
    a cheap screening method (RegFGW or otherwise) selected, in ranked
    order (first row = best pick). Not RegFGW-specific by design, since its
    own output format is unverified here -- build this CSV from whatever
    ``run_regfgw_optimize`` actually produced.

    ``exhaustive_source``: a CSV with ``id_column`` and ``energy_column``,
    one row per registry in the exhaustive grid (e.g. from InterMat's
    ``displacement_interval`` sweep), each already relaxed and audited (e.g.
    via ``iface vasp adhesion audit`` per registry) so ``energy_column``
    holds a real converged value.

    For each ``k`` in ``k_values``: recall@k is the fraction of the true
    best-``k`` exhaustive-grid ids also present in the proposed top-``k``;
    ``best_preserved`` is whether the single true best id was proposed at
    all; ``energy_regret`` is the gap between the best energy among
    proposed picks and the true best energy (0 = no regret). Whether "best"
    means lowest or highest ``energy_column`` is set by
    ``lower_energy_is_better`` -- for work of adhesion (higher = more
    stable interface) that is ``False``, the default.
    """

    topk_path = Path(topk_source).resolve()
    exhaustive_path = Path(exhaustive_source).resolve()
    output_path = Path(output).resolve()

    with topk_path.open(newline="", encoding="utf-8") as handle:
        topk_rows = list(csv.DictReader(handle))
    with exhaustive_path.open(newline="", encoding="utf-8") as handle:
        exhaustive_rows = list(csv.DictReader(handle))
    if not topk_rows:
        raise ValueError("No rows in top-k source CSV")
    if not exhaustive_rows:
        raise ValueError("No rows in exhaustive source CSV")
    if id_column not in topk_rows[0]:
        raise ValueError(f"Top-k CSV must have an {id_column!r} column")
    if id_column not in exhaustive_rows[0] or energy_column not in exhaustive_rows[0]:
        raise ValueError(f"Exhaustive CSV must have {id_column!r} and {energy_column!r} columns")

    proposed_ids = [row[id_column] for row in topk_rows]
    energy_by_id = {row[id_column]: float(row[energy_column]) for row in exhaustive_rows}
    ranked_ids = sorted(energy_by_id, key=lambda item: energy_by_id[item], reverse=not lower_energy_is_better)
    true_best_energy = energy_by_id[ranked_ids[0]]

    results: list[dict[str, Any]] = []
    for k in k_values:
        true_best_k = set(ranked_ids[:k])
        proposed_k = proposed_ids[:k]
        proposed_k_in_grid = [pid for pid in proposed_k if pid in energy_by_id]
        overlap = len(true_best_k & set(proposed_k))
        best_energy_among_proposed: float | None = None
        if proposed_k_in_grid:
            candidate_energies = [energy_by_id[pid] for pid in proposed_k_in_grid]
            best_energy_among_proposed = (
                min(candidate_energies) if lower_energy_is_better else max(candidate_energies)
            )
        energy_regret = (
            None
            if best_energy_among_proposed is None
            else abs(best_energy_among_proposed - true_best_energy)
        )
        results.append(
            {
                "k": k,
                "recall_at_k": overlap / k,
                "best_preserved": ranked_ids[0] in proposed_ids[:k],
                "proposed_ids_missing_from_grid": len(proposed_k) - len(proposed_k_in_grid),
                "energy_regret": energy_regret,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "topk_source": str(topk_path),
        "exhaustive_source": str(exhaustive_path),
        "output": str(output_path),
        "lower_energy_is_better": lower_energy_is_better,
        "true_best_id": ranked_ids[0],
        "true_best_energy": true_best_energy,
        "exhaustive_grid_size": len(ranked_ids),
        "proposed_count": len(proposed_ids),
        "results": results,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
