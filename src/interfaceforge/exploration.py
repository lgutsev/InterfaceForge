"""Generate explicit exploration matrices for active-learning campaigns."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from .config import Campaign
from .state import StateStore


def generate_exploration(campaign: Campaign, output: str | Path | None = None) -> dict[str, Any]:
    """Expand configured thermodynamic conditions into deterministic tasks."""

    settings = campaign.exploration
    temperatures = [float(value) for value in settings.get("temperatures", [300, 450])]
    strains = [float(value) for value in settings.get("strains", [0.0])]
    replicas = int(settings.get("replicas", 1))
    if replicas < 1:
        raise ValueError("exploration.replicas must be positive")
    tasks: list[dict[str, Any]] = []
    for system, temperature, strain, replica in itertools.product(
        campaign.systems, temperatures, strains, range(replicas)
    ):
        task_id = len(tasks)
        tasks.append(
            {
                "task_id": task_id,
                "system": system.id,
                "kind": system.kind,
                "temperature_k": temperature,
                "strain": strain,
                "replica": replica,
                "seed": 20260730 + task_id * 7919,
                "status": "planned",
            }
        )
    root = Path(output).resolve() if output else campaign.root / "runs" / "exploration"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    payload = {
        "schema_version": 1,
        "campaign": campaign.name,
        "task_count": len(tasks),
        "temperatures_k": temperatures,
        "strains": strains,
        "replicas": replicas,
        "tasks": tasks,
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    state = StateStore(campaign.root)
    state.event("explore", tasks=len(tasks), output=str(root))
    state.artifact("exploration_manifest", manifest)
    return payload
