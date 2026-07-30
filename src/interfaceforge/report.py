"""Build a self-contained campaign dashboard from InterfaceForge artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .config import Campaign
from .state import StateStore


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (json.JSONDecodeError, OSError):
        return None


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value if value is not None else ''))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _lcurve_final(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = [line.split() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header = lines[0]
    try:
        values = [float(value) for value in lines[-1]]
    except ValueError:
        return None
    return dict(zip(header, values, strict=False))


def build_report(campaign: Campaign, output: str | Path | None = None) -> dict[str, Any]:
    destination = Path(output).resolve() if output else campaign.root / "reports" / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = StateStore(campaign.root).load()
    plan = _load_json(campaign.root / ".interfaceforge" / "plan.json")
    audit = _load_json(campaign.root / "reports" / "audit" / "audit.json")
    dataset = _load_json(campaign.root / "datasets" / "canonical" / "manifest.json")
    mace = _load_json(campaign.root / "models" / "mace" / "training_manifest.json")
    deepmd = _load_json(campaign.root / "models" / "deepmd" / "ensemble_manifest.json")

    sections: list[str] = []
    if plan:
        sections.append(
            "<h2>Campaign plan</h2>"
            + _table(
                ["Engine", "System", "Stage", "Profile"],
                [
                    [task.get("engine"), task.get("system", "—"), task.get("stage"), task.get("profile")]
                    for task in plan.get("tasks", [])
                ],
            )
        )
    if audit:
        sections.append(
            "<h2>VASP and VASP-MLFF health</h2>"
            + _table(
                ["Run", "Mode", "Progress", "Health", "Next action"],
                [
                    [
                        row.get("relative_path"),
                        row.get("ml_mode"),
                        f"{row['progress_pct']:.1f}%" if row.get("progress_pct") is not None else "",
                        row.get("health"),
                        row.get("next_action"),
                    ]
                    for row in audit.get("runs", [])
                ],
            )
        )
    if dataset:
        sections.append(
            "<h2>Canonical dataset</h2>"
            + _table(
                ["Strategy", "Trajectories", "Train frames", "Validation frames", "Test frames", "Type map"],
                [[
                    dataset.get("strategy"),
                    dataset.get("trajectories"),
                    dataset.get("frame_counts", {}).get("train"),
                    dataset.get("frame_counts", {}).get("valid"),
                    dataset.get("frame_counts", {}).get("test"),
                    " ".join(dataset.get("type_map", [])),
                ]],
            )
        )
    model_rows: list[list[Any]] = []
    if mace:
        model_rows.append(["MACE", "two-stage", len(mace.get("stages", [])), "generated"])
    if deepmd:
        model_rows.append([
            "DeePMD",
            ", ".join(deepmd.get("architectures", [])),
            len(deepmd.get("models", [])),
            deepmd.get("backend"),
        ])
    if model_rows:
        sections.append(
            "<h2>Model campaigns</h2>"
            + _table(["Engine", "Architecture", "Runs", "Backend/status"], model_rows)
        )

    learning_rows: list[list[Any]] = []
    for path in sorted((campaign.root / "models").rglob("lcurve.out")):
        final = _lcurve_final(path)
        if final:
            learning_rows.append([str(path.relative_to(campaign.root)), final.get("step", ""), json.dumps(final)])
    if learning_rows:
        sections.append("<h2>Completed learning curves</h2>" + _table(["Run", "Step", "Final metrics"], learning_rows))

    event_rows = [
        [event.get("time"), event.get("action"), json.dumps(event.get("details", {}), sort_keys=True)]
        for event in state.get("events", [])[-30:]
    ]
    if event_rows:
        sections.append("<h2>Provenance log</h2>" + _table(["Time", "Action", "Details"], event_rows))

    style = """
    body{font-family:Inter,system-ui,sans-serif;margin:2rem auto;max-width:1400px;padding:0 1rem;color:#17202a}
    h1{margin-bottom:.25rem} h2{margin-top:2rem;border-bottom:2px solid #1f6f78;padding-bottom:.35rem}
    .sub{color:#58636d} table{border-collapse:collapse;width:100%;font-size:.9rem}
    th,td{border:1px solid #d8dee4;padding:.5rem;vertical-align:top} th{background:#eaf4f4;text-align:left}
    tr:nth-child(even){background:#f8fafb} code{background:#f1f3f5;padding:.1rem .25rem}
    """
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(campaign.name)} — InterfaceForge</title><style>{style}</style></head>
<body><h1>{html.escape(campaign.name)}</h1>
<div class="sub">{html.escape(campaign.description)} · generated by InterfaceForge</div>
{''.join(sections) if sections else '<p>No generated campaign artifacts were found yet.</p>'}
</body></html>
"""
    destination.write_text(document, encoding="utf-8")
    summary = {
        "campaign": campaign.name,
        "output": str(destination),
        "has_plan": bool(plan),
        "has_audit": bool(audit),
        "has_dataset": bool(dataset),
        "has_mace": bool(mace),
        "has_deepmd": bool(deepmd),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    StateStore(campaign.root).artifact("campaign_report", destination)
    return summary
