"""Campaign-level VASP MLFF Bayesian force-error plots."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import find_runs
from .errors import DependencyError, SafetyError
from .vasp import parse_incar


@dataclass(frozen=True)
class BeefPoint:
    step: int
    time_fs: float
    beef_ev_a: float
    ml_ctifor_ev_a: float | None
    state: str


@dataclass(frozen=True)
class BeefSeries:
    run: Path
    relative_path: str
    potim_fs: float
    points: tuple[BeefPoint, ...]


def _finite_number(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_beef_series(run: str | Path, root: str | Path | None = None) -> BeefSeries:
    """Parse BEEF, ML_CTIFOR and STATUS state records from one VASP MLFF run."""

    run_path = Path(run).resolve()
    root_path = Path(root).resolve() if root is not None else run_path
    log_path = run_path / "ML_LOGFILE"
    if not log_path.is_file() or not log_path.stat().st_size:
        raise SafetyError(f"Missing or empty ML_LOGFILE: {log_path}")

    incar = parse_incar(run_path / "INCAR")
    try:
        potim = float(incar.get("POTIM", 1.0))
    except ValueError:
        potim = 1.0
    if not math.isfinite(potim) or potim <= 0:
        raise SafetyError(f"POTIM must be positive and finite in {run_path / 'INCAR'}")

    statuses: dict[int, str] = {}
    raw_points: list[tuple[int, float, float | None]] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        record = tokens[0].upper()
        if record == "STATUS" and len(tokens) >= 3:
            try:
                statuses[int(tokens[1])] = tokens[2].lower()
            except ValueError:
                continue
        elif record in {"BEEF", "BEFF"} and len(tokens) >= 4:
            try:
                step = int(tokens[1])
            except ValueError:
                continue
            beef = _finite_number(tokens[3])
            if beef is None:
                continue
            threshold = _finite_number(tokens[5]) if len(tokens) >= 6 else None
            raw_points.append((step, beef, threshold))

    if not raw_points:
        raise SafetyError(f"No usable BEEF records were found in {log_path}")
    try:
        relative = "." if run_path == root_path else run_path.relative_to(root_path).as_posix()
    except ValueError:
        relative = run_path.name
    points = tuple(
        BeefPoint(
            step=step,
            time_fs=step * potim,
            beef_ev_a=beef,
            ml_ctifor_ev_a=threshold,
            state=statuses.get(step, ""),
        )
        for step, beef, threshold in raw_points
    )
    return BeefSeries(run=run_path, relative_path=relative, potim_fs=potim, points=points)


def discover_beef_series(
    root: str | Path,
    *,
    include_archives: bool = False,
) -> list[BeefSeries]:
    """Discover and parse every usable MLFF BEEF series below a campaign root."""

    campaign_root = Path(root).resolve()
    series: list[BeefSeries] = []
    for run in find_runs(campaign_root, recursive=True, include_archives=include_archives):
        if not (run / "ML_LOGFILE").is_file():
            continue
        try:
            series.append(parse_beef_series(run, campaign_root))
        except SafetyError:
            continue
    if not series:
        raise SafetyError(f"No usable ML_LOGFILE BEEF records were found below {campaign_root}")
    return series


def _write_csv(path: Path, series: list[BeefSeries]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "run",
                "relative_path",
                "step",
                "time_fs",
                "bayesian_force_error_max_ev_a",
                "ml_ctifor_ev_a",
                "status_state",
            )
        )
        for item in series:
            for point in item.points:
                writer.writerow(
                    (
                        item.run.name,
                        item.relative_path,
                        point.step,
                        point.time_fs,
                        point.beef_ev_a,
                        "" if point.ml_ctifor_ev_a is None else point.ml_ctifor_ev_a,
                        point.state,
                    )
                )
                rows += 1
    return rows


def _matplotlib() -> Any:
    if "MPLCONFIGDIR" not in os.environ:
        cache_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "interfaceforge-matplotlib"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_root)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyError(
            "Bayesian-error plotting requires matplotlib; install InterfaceForge with '.[report]' or '.[all]'"
        ) from exc
    return plt


def _plot_series(axis: Any, item: BeefSeries, *, title: str) -> None:
    times = [point.time_fs for point in item.points]
    beef = [point.beef_ev_a for point in item.points]
    axis.plot(times, beef, color="#1f77b4", linewidth=1.7, label="Bayesian force error")
    threshold_points = [point for point in item.points if point.ml_ctifor_ev_a is not None]
    if threshold_points:
        axis.plot(
            [point.time_fs for point in threshold_points],
            [point.ml_ctifor_ev_a for point in threshold_points],
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label="ML_CTIFOR",
        )
    for state, color, marker, label in (
        ("learning", "#ff7f0e", "o", "Learning event"),
        ("critical", "#b2182b", "x", "Critical event"),
    ):
        selected = [point for point in item.points if point.state == state]
        if selected:
            axis.scatter(
                [point.time_fs for point in selected],
                [point.beef_ev_a for point in selected],
                color=color,
                marker=marker,
                s=28,
                linewidths=1.3,
                zorder=3,
                label=label,
            )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Time (fs)")
    axis.set_ylabel("Force error (eV/Å)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend(fontsize=7, loc="upper right")


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return cleaned.strip("._") or "run"


def plot_beef_campaign(
    root: str | Path,
    *,
    output: str | Path | None = None,
    data_output: str | Path | None = None,
    include_archives: bool = False,
    individual: bool = False,
    dpi: int = 160,
) -> dict[str, Any]:
    """Create a campaign BEEF plot, its CSV data and optional per-run plots."""

    campaign_root = Path(root).resolve()
    plot_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else campaign_root / "ML_BayesianErrorPlot_campaign.png"
    )
    csv_path = (
        Path(data_output).expanduser().resolve()
        if data_output is not None
        else plot_path.with_suffix(".csv")
    )
    if dpi < 72:
        raise SafetyError("--dpi must be at least 72")
    series = discover_beef_series(campaign_root, include_archives=include_archives)
    data_rows = _write_csv(csv_path, series)
    plt = _matplotlib()

    columns = 1 if len(series) == 1 else 2
    rows = math.ceil(len(series) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 3.9 * rows), squeeze=False)
    flat_axes = list(axes.flat)
    for axis, item in zip(flat_axes, series, strict=False):
        _plot_series(axis, item, title=item.relative_path)
    for axis in flat_axes[len(series) :]:
        axis.set_visible(False)
    figure.suptitle("VASP MLFF campaign Bayesian force errors", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    individual_outputs: list[str] = []
    if individual:
        individual_root = plot_path.parent / f"{plot_path.stem}_individual"
        individual_root.mkdir(parents=True, exist_ok=True)
        for item in series:
            figure, axes = plt.subplots(1, 1, figsize=(9, 5.5))
            _plot_series(axes, item, title=item.relative_path)
            figure.tight_layout()
            target = individual_root / f"{_safe_filename(item.relative_path)}.png"
            figure.savefig(target, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            individual_outputs.append(str(target))

    return {
        "root": str(campaign_root),
        "run_count": len(series),
        "point_count": data_rows,
        "include_archives": include_archives,
        "plot": str(plot_path),
        "data": str(csv_path),
        "individual_plots": individual_outputs,
    }
