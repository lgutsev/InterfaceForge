from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from interfaceforge.deepmd_audit import summarize


def _write(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# reference prediction\n"
        + "".join(" ".join(str(value) for value in row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_summarize_weights_raw_errors_not_system_rmse(tmp_path: Path) -> None:
    systems = tmp_path / "test_systems.txt"
    systems.write_text("small\nlarge\n", encoding="utf-8")
    eval_root = tmp_path / "evaluation"

    # Energy errors are 1 and 3 eV/atom, so the combined RMSE is sqrt(5),
    # not the arithmetic mean of the two system RMSE values (2.0).
    _write(
        eval_root / "by_system/system_000/model_000_detail.e_peratom.out",
        [[0.0, 1.0]],
    )
    _write(
        eval_root / "by_system/system_001/model_000_detail.e_peratom.out",
        [[0.0, 3.0]],
    )
    for index in range(2):
        _write(
            eval_root / f"by_system/system_{index:03d}/model_000_detail.f.out",
            [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]],
        )

    payload = summarize(eval_root, systems, "dpa2", [11])
    metrics = payload["models"][0]["metrics"]
    assert metrics["energy"]["rmse"] == pytest.approx(5**0.5)
    assert metrics["force"]["rmse"] == pytest.approx((14 / 3) ** 0.5)
    assert metrics["virial"] == {"count": 0, "mae": None, "rmse": None}

    with (eval_root / "rmse_overall.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["energy_rmse_mev_per_atom"]) == pytest.approx(1000 * 5**0.5)
    assert json.loads((eval_root / "rmse_audit.json").read_text())["schema_version"] == 1
