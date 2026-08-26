"""Cross-audit synchronized MACE and DeePMD leaf datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

SPLITS = ("train", "valid", "test")
COLORS = {"train": "#2563eb", "valid": "#f59e0b", "test": "#dc2626"}


def _read_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing collector manifest: {path}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            leaf = str(raw.get("relative_leaf", "")).strip()
            split = str(raw.get("split", "")).strip()
            if not leaf:
                continue
            key = (leaf, split)
            if key in rows:
                raise ValueError(f"Duplicate leaf/split in {path}: {leaf} [{split}]")
            rows[key] = {
                "relative_leaf": leaf,
                "split": split,
                "frames": int(raw.get("frames", 0) or 0),
                "frame_digest": str(raw.get("frame_digest", "")).strip(),
                "status": str(raw.get("status", "UNKNOWN")).strip(),
                "detail": str(raw.get("detail", "")).strip(),
            }
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def _branch(leaf: str) -> str:
    parts = Path(leaf).parts
    if parts and parts[0] == "interface" and len(parts) >= 4:
        return "/".join(parts[:4])
    if parts and parts[0] == "bulk" and len(parts) >= 2:
        return "/".join(parts[:2])
    return "/".join(parts[:-1] or parts) or "unknown"


def audit_leaf_manifests(
    mace_manifest: str | Path,
    deepmd_manifest: str | Path,
) -> dict[str, Any]:
    """Require identical leaf/split membership, frame counts, and frame indices."""
    mace = _read_manifest(Path(mace_manifest))
    deepmd = _read_manifest(Path(deepmd_manifest))
    memberships = sorted(set(mace) | set(deepmd))
    split_frames = {split: 0 for split in SPLITS}
    split_leaves = {split: 0 for split in SPLITS}
    branch_frames: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    for leaf, manifest_split in memberships:
        key = (leaf, manifest_split)
        m = mace.get(key)
        d = deepmd.get(key)
        issues: list[str] = []
        if m is None:
            issues.append("missing from MACE")
        if d is None:
            issues.append("missing from DeePMD")
        if m and d:
            if m["split"] != d["split"]:
                issues.append(
                    f"split mismatch: MACE={m['split']}, DeePMD={d['split']}"
                )
            if m["frames"] != d["frames"]:
                issues.append(
                    f"frame mismatch: MACE={m['frames']}, DeePMD={d['frames']}"
                )
            if (
                m["frame_digest"]
                and d["frame_digest"]
                and m["frame_digest"] != d["frame_digest"]
            ):
                issues.append("source-frame membership digest mismatch")
            for engine, row in (("MACE", m), ("DeePMD", d)):
                if row["status"] != "OK":
                    issues.append(
                        f"{engine} status={row['status']}: {row['detail']}"
                    )

        split = (m or d or {}).get("split", manifest_split)
        frames = int((m or d or {}).get("frames", 0))
        if not issues and split in split_frames:
            split_frames[split] += frames
            split_leaves[split] += 1
            branch_frames[_branch(leaf)][split] += frames
        if issues:
            problems.append(
                {
                    "relative_leaf": leaf,
                    "split": manifest_split,
                    "issues": issues,
                }
            )
        rows.append(
            {
                "relative_leaf": leaf,
                "branch": _branch(leaf),
                "split": manifest_split,
                "mace_frames": m["frames"] if m else 0,
                "deepmd_frames": d["frames"] if d else 0,
                "mace_frame_digest": m["frame_digest"] if m else "",
                "deepmd_frame_digest": d["frame_digest"] if d else "",
                "audit_status": "FAILED" if issues else "OK",
                "detail": "; ".join(issues),
            }
        )

    empty = [split for split, count in split_frames.items() if count == 0]
    if empty:
        problems.append(
            {
                "relative_leaf": "__dataset__",
                "issues": [f"empty splits: {', '.join(empty)}"],
            }
        )
    unique_leaves = {leaf for leaf, _ in memberships}
    return {
        "schema_version": 2,
        "status": "FAILED" if problems else "OK",
        "synchronized": not problems,
        "leaves": len(unique_leaves),
        "leaf_split_memberships": len(memberships),
        "split_frames": split_frames,
        "split_leaves": split_leaves,
        "branch_frames": {
            key: dict(value) for key, value in sorted(branch_frames.items())
        },
        "problems": problems,
        "rows": rows,
    }


def _write_svg(path: Path, report: dict[str, Any]) -> None:
    branches = list(report["branch_frames"])
    width, margin, chart_x, row_h, top = 1400, 54, 420, 30, 245
    chart_w = width - chart_x - margin - 50
    height = max(600, top + len(branches) * row_h + 110)
    totals = {
        branch: sum(report["branch_frames"][branch].get(split, 0) for split in SPLITS)
        for branch in branches
    }
    scale = max(totals.values(), default=1)
    total_frames = sum(report["split_frames"].values())
    status_color = "#15803d" if report["status"] == "OK" else "#b91c1c"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<style>text{font-family:Arial,sans-serif;fill:#172033}'
            '.title{font-size:28px;font-weight:700}.sub{font-size:14px;fill:#526071}'
            '.small{font-size:12px;fill:#526071}.value{font-size:24px;font-weight:700}</style>'
        ),
        f'<text x="{margin}" y="48" class="title">Synchronized leaf-dataset audit</text>',
        (
            f'<text x="{margin}" y="75" class="sub">MACE extxyz and DeePMD NPY '
            "membership, split, and frame agreement</text>"
        ),
        f'<rect x="1210" y="28" width="130" height="44" rx="12" fill="{status_color}"/>',
        (
            '<text x="1275" y="57" text-anchor="middle" '
            'style="font-size:18px;font-weight:700;fill:white">'
            f'{report["status"]}</text>'
        ),
    ]
    cards = [
        ("Leaves", report["leaves"], "#0f766e"),
        ("Frames", total_frames, "#4338ca"),
        ("Train", report["split_frames"]["train"], COLORS["train"]),
        ("Validation", report["split_frames"]["valid"], COLORS["valid"]),
        ("Test", report["split_frames"]["test"], COLORS["test"]),
    ]
    card_w = 244
    for index, (label, value, color) in enumerate(cards):
        x = margin + index * (card_w + 18)
        svg.extend(
            [
                f'<rect x="{x}" y="100" width="{card_w}" height="98" rx="14" fill="white" stroke="#dbe3ed"/>',
                f'<rect x="{x}" y="100" width="7" height="98" rx="3" fill="{color}"/>',
                f'<text x="{x+22}" y="132" class="small">{escape(label)}</text>',
                f'<text x="{x+22}" y="171" class="value">{value:,}</text>',
            ]
        )
    svg.append(f'<text x="{margin}" y="{top-20}" style="font-size:18px;font-weight:700">Frames by source branch</text>')
    for index, branch in enumerate(branches):
        y = top + index * row_h
        svg.append(
            f'<text x="{chart_x-14}" y="{y+17}" text-anchor="end" '
            f'style="font-size:13px">{escape(branch)}</text>'
        )
        x = chart_x
        for split in SPLITS:
            value = report["branch_frames"][branch].get(split, 0)
            bar_width = chart_w * value / scale
            if bar_width:
                svg.append(
                    f'<rect x="{x:.1f}" y="{y+3}" width="{bar_width:.1f}" '
                    f'height="20" rx="3" fill="{COLORS[split]}"/>'
                )
            x += bar_width
        svg.append(f'<text x="{chart_x+chart_w+8}" y="{y+17}" class="small">{totals[branch]:,}</text>')
    legend_y = top + len(branches) * row_h + 30
    x = chart_x
    for split in SPLITS:
        svg.append(f'<rect x="{x}" y="{legend_y}" width="16" height="16" rx="3" fill="{COLORS[split]}"/>')
        svg.append(f'<text x="{x+23}" y="{legend_y+13}" class="small">{split}</text>')
        x += 105
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_leaf_audit(report: dict[str, Any], output: str | Path) -> dict[str, str]:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": root / "leaf_dataset_audit.json",
        "csv": root / "leaf_dataset_audit.csv",
        "markdown": root / "leaf_dataset_audit.md",
        "svg": root / "leaf_dataset_audit.svg",
    }
    paths["json"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = [
        "# Synchronized leaf-dataset audit",
        "",
        f"**Status:** {report['status']}",
        "",
        "| Split | Leaves | Frames |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(f"| {split} | {report['split_leaves'][split]} | {report['split_frames'][split]} |")
    if report["problems"]:
        lines.extend(["", "## Problems", ""])
        for problem in report["problems"]:
            lines.append(f"- `{problem['relative_leaf']}`: {'; '.join(problem['issues'])}")
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_svg(paths["svg"], report)
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-manifest", required=True)
    parser.add_argument("--deepmd-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_leaf_manifests(args.mace_manifest, args.deepmd_manifest)
    report["outputs"] = write_leaf_audit(report, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
