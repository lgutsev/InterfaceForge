"""Pre-registered, code-checked agent assignments.

An assignment is a directory holding ``assignment.yaml`` (the machine-readable
spec, see ``schemas/assignment.schema.json``). The workflow is:

1. ``iface assignment render <assignment.yaml>`` writes ``ASSIGNMENT.md`` in the
   PI handoff format. Rendering is deterministic: the same YAML always gives
   byte-identical Markdown.
2. ``iface assignment freeze <dir>`` pre-registers the spec by hash in
   ``frozen.json`` and appends a ``frozen`` event to the directory's
   :class:`~interfaceforge.state.StateStore` ledger. It refuses a second freeze.
   Later amendments go only into ``NOTES.md`` as dated entries.
3. The agent returns ``report.json`` (``schemas/assignment_report.schema.json``).
4. ``iface assignment validate <dir> --report report.json`` checks the report by
   code: hash pinning, one allowed verdict per task, sourced values, unchanged
   target git state, and write permissions.

Only judgement stays in prose; anything a script can check is an ``iface audit``
command referenced from an ``audit`` task.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import yaml

from .. import __version__
from ..errors import ConfigurationError, SafetyError
from ..schema_check import schema_errors
from ..state import StateStore, sha256_file, utc_now

ASSIGNMENT_FILE = "assignment.yaml"
RENDERED_FILE = "ASSIGNMENT.md"
FROZEN_FILE = "frozen.json"
NOTES_FILE = "NOTES.md"
ASSIGNMENT_SCHEMA = "assignment.schema.json"
REPORT_SCHEMA = "assignment_report.schema.json"

KIND_MEANING = {
    "audit": "audit: run the command; its output decides the verdict",
    "diag_prepare": "diagnostic proposal: prepare and dry-run only; never --execute without PI approval",
    "judgement": "judgement: written assessment with labelled evidence",
}
COST_MEANING = {
    "from_records": "price only from recorded timings/k-point counts; show the arithmetic",
    "not_inferable": "not inferable from records; write 'not inferable' and do not invent hours",
}
DEFAULT_EVIDENCE_LABELS = {
    "E": "established from a file",
    "P": "recorded/probable",
    "H": "hypothesis",
    "C": "concern",
    "S": "suggestion",
}


# --------------------------------------------------------------------------- spec


def load_assignment(path: str | Path) -> dict[str, Any]:
    """Load and schema-check ``assignment.yaml``; raise ConfigurationError on violations."""

    source = Path(path)
    try:
        spec = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(spec, dict):
        raise ConfigurationError(f"{source}: top level must be a mapping")
    problems = schema_errors(spec, ASSIGNMENT_SCHEMA)
    ids = [task.get("id") for task in spec.get("tasks", []) if isinstance(task, dict)]
    duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1 and task_id is not None})
    if duplicates:
        problems.append(f"/tasks: duplicate task id(s) {duplicates}")
    for task in spec.get("tasks", []):
        if isinstance(task, dict) and isinstance(task.get("verdicts"), list):
            verdicts = task["verdicts"]
            if len(set(verdicts)) != len(verdicts):
                problems.append(f"/tasks/{task.get('id')}: duplicate verdicts in vocabulary")
    if problems:
        raise ConfigurationError(f"{source} does not conform to {ASSIGNMENT_SCHEMA}:\n  " + "\n  ".join(problems))
    return spec


def _numbered(items: list[str], start: int = 1) -> list[str]:
    return [f"{index}. {item}" for index, item in enumerate(items, start)]


def _block(text: str) -> list[str]:
    return [line.rstrip() for line in str(text).strip("\n").splitlines()]


def hard_rules(spec: dict[str, Any]) -> list[str]:
    """Hard rules generated from ``permissions`` plus any extra ``rules``."""

    permissions = spec["permissions"]
    rules = ["Write only to paths matching: " + "; ".join(permissions["write"]) + ". Everything else is read-only."]
    if permissions["read"]:
        rules.append("Read only: " + "; ".join(permissions["read"]) + ".")
    if permissions["git_allowed"]:
        rules.append(
            "Git: only " + ", ".join(permissions["git_allowed"]) + ". No other git subcommand; never change refs, "
            "the index or the working tree."
        )
    else:
        rules.append("Git: none.")
    for location in permissions["forbid"]:
        rules.append(f"Never create, edit, move or delete anything in: {location}")
    rules.extend(spec.get("rules", []))
    rules.append(
        "Every number in the report must come from a command you ran or a file you read, cited in its `source`. "
        "Unchecked items go to `not_assessed`."
    )
    rules.append(
        "Return report.json matching assignment_report.schema.json and run "
        "`iface assignment validate <assignment dir> --report report.json` before claiming done."
    )
    return rules


def _render_task(number: int, task: dict[str, Any]) -> list[str]:
    title = f" — {task['title']}" if task.get("title") else ""
    lines = [f"TASK {number} [{task['id']}]{title} ({KIND_MEANING[task['kind']]})"]
    lines += _block(task["description"])
    if task.get("hypothesis"):
        lines.append(f"- Hypothesis [H]: {task['hypothesis']}")
    if task.get("command"):
        lines.append(f"- Command: `{task['command']}`")
    criterion = task.get("criterion")
    if criterion:
        lines.append(
            f"- Pre-registered criterion: {criterion['metric']} [{criterion['unit']}], "
            f"threshold {criterion['threshold']} — {criterion['rule']}"
        )
    if task.get("cost"):
        lines.append(f"- Cost: {COST_MEANING[task['cost']]}")
    lines.append("- Verdict (exactly one): " + " | ".join(task["verdicts"]))
    if task["outputs"]:
        lines.append("- Outputs: " + "; ".join(task["outputs"]))
    return lines


def render_assignment(spec: dict[str, Any]) -> str:
    """Render the spec to handoff-format Markdown. Pure and deterministic."""

    title = spec.get("title", spec["id"])
    target = spec["target"]
    labels = spec.get("evidence_labels") or DEFAULT_EVIDENCE_LABELS
    lines = [f"# ASSIGNMENT {spec['id']} — {title}", ""]
    if spec.get("prepared"):
        lines += [spec["prepared"], ""]
    lines.append(f"Template: {spec['template']}")
    also = ", ".join(target["also_read_commits"]) or "none"
    lines.append(f"Target: {target['repo']} @ {target['commit']} (also read commits: {also})")
    lines.append("Evidence labels: " + " · ".join(f"[{key}] {value}" for key, value in labels.items()))
    if spec.get("context"):
        lines += ["", "## Context", ""] + _block(spec["context"])
    hypotheses = [task for task in spec["tasks"] if task.get("hypothesis")]
    if hypotheses:
        lines += ["", "## Hypotheses under test", "", "| Task | Kind | Hypothesis [H] |", "|---|---|---|"]
        for task in hypotheses:
            text = " ".join(task["hypothesis"].split()).replace("|", "\\|")
            lines.append(f"| {task['id']} | {task['kind']} | {text} |")

    lines += ["", "=====", f"TASK: {title}", f"TARGET REPO: {target['repo']} @ {target['commit']}"]
    lines += ["", "HARD RULES"] + _numbered(hard_rules(spec))
    for number, task in enumerate(spec["tasks"], 1):
        lines += [""] + _render_task(number, task)

    deliverables = list(spec.get("deliverables", []))
    for task in spec["tasks"]:
        deliverables += [f"[{task['id']}] {output}" for output in task["outputs"]]
    deliverables.append("report.json matching assignment_report.schema.json")
    lines += ["", "=====", "DELIVERABLES"] + _numbered(deliverables)
    lines += ["", "DONE WHEN"] + [f"- {item}" for item in spec["done_when"]]
    lines += ["", "REPORT BACK"] + [f"- {item}" for item in spec["report_back"]]
    lines += ["====="]
    return "\n".join(lines) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_to_file(source: str | Path, output: str | Path | None = None, *, force: bool = False) -> dict[str, Any]:
    source = Path(source)
    spec = load_assignment(source)
    data = render_assignment(spec).encode("utf-8")
    target = Path(output) if output else source.parent / RENDERED_FILE
    digest = _sha256_bytes(data)
    if target.exists():
        if target.read_bytes() == data:
            return {"assignment_id": spec["id"], "output": str(target), "sha256": digest, "changed": False}
        if (target.parent / FROZEN_FILE).exists():
            raise SafetyError(
                f"{target.parent / FROZEN_FILE} exists; a frozen assignment is never re-rendered. "
                f"Record amendments in {NOTES_FILE}."
            )
        if not force:
            raise SafetyError(f"{target} exists and differs; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"assignment_id": spec["id"], "output": str(target), "sha256": digest, "changed": True}


# --------------------------------------------------------------------------- templates


def template_text(name: str) -> str:
    """Raw text of a bundled template, e.g. ``due_diligence_v1``."""

    resource = resources.files("interfaceforge.assignments").joinpath("templates", name, ASSIGNMENT_FILE)
    if not resource.is_file():
        raise ConfigurationError(f"Unknown assignment template {name!r}")
    return resource.read_text(encoding="utf-8")


def template_placeholders(name: str) -> list[str]:
    """Placeholder names in a template, in first-seen order."""

    seen: list[str] = []
    for match in Template.pattern.finditer(template_text(name)):
        key = match.group("named") or match.group("braced")
        if key and key not in seen:
            seen.append(key)
    return seen


def init_assignment(
    directory: str | Path, template: str, values: dict[str, str], *, force: bool = False
) -> dict[str, Any]:
    """Write ``assignment.yaml`` from a template with ``string.Template`` substitution."""

    target = Path(directory) / ASSIGNMENT_FILE
    if target.exists() and not force:
        raise SafetyError(f"{target} exists; pass --force to overwrite")
    if (Path(directory) / FROZEN_FILE).exists():
        raise SafetyError(f"{Path(directory) / FROZEN_FILE} exists; a frozen assignment is never re-initialised")
    text = Template(template_text(template)).safe_substitute(values)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    remaining = [key for key in template_placeholders(template) if key not in values]
    return {"output": str(target), "template": template, "unfilled_placeholders": remaining}


# --------------------------------------------------------------------------- freeze


def freeze_assignment(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    frozen_path = root / FROZEN_FILE
    if frozen_path.exists():
        raise SafetyError(f"{frozen_path} already exists; an assignment is frozen once. Amend via {NOTES_FILE}.")
    spec_path = root / ASSIGNMENT_FILE
    rendered_path = root / RENDERED_FILE
    spec = load_assignment(spec_path)
    if not rendered_path.is_file():
        raise SafetyError(f"{rendered_path} is missing; run 'iface assignment render {spec_path}' first")
    if rendered_path.read_bytes() != render_assignment(spec).encode("utf-8"):
        raise SafetyError(f"{rendered_path} does not match a fresh render of {spec_path}; re-render before freezing")
    payload = {
        "assignment_id": spec["id"],
        "template": spec["template"],
        "assignment_yaml_sha256": sha256_file(spec_path),
        "assignment_md_sha256": sha256_file(rendered_path),
        "frozen_at_utc": utc_now(),
        "tool": "interfaceforge",
        "tool_version": __version__,
    }
    with frozen_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    notes = root / NOTES_FILE
    if not notes.exists():
        notes.write_text(
            f"# NOTES — {spec['id']}\n\n"
            "Dated amendments only, one per entry: `## YYYY-MM-DD — <amendment>`.\n"
            "The criteria in assignment.yaml are frozen (see frozen.json); amendments never rewrite them.\n",
            encoding="utf-8",
            newline="\n",
        )
    StateStore(root).event(
        "frozen",
        assignment_id=spec["id"],
        assignment_yaml_sha256=payload["assignment_yaml_sha256"],
        assignment_md_sha256=payload["assignment_md_sha256"],
        frozen_json_sha256=sha256_file(frozen_path),
    )
    return {**payload, "frozen_json": str(frozen_path)}


# --------------------------------------------------------------------------- git snapshot


def git_snapshot(repo: str | Path) -> dict[str, str]:
    """Read-only HEAD + porcelain hash, in the form the report schema expects.

    ``porcelain_sha256`` hashes the ``--porcelain=v1`` lines *without* the
    ``## branch`` header, each terminated by ``\\n``; the header is returned
    separately because ``[behind N]`` can change when another process fetches.
    """

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            raise ConfigurationError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}")
        return result.stdout

    head = run("rev-parse", "HEAD").strip()
    lines = run("status", "--porcelain=v1", "-b").splitlines()
    header = lines[0] if lines and lines[0].startswith("## ") else ""
    body = [line for line in lines if not line.startswith("## ")]
    porcelain = "".join(line + "\n" for line in body)
    return {
        "head": head,
        "porcelain_sha256": _sha256_bytes(porcelain.encode("utf-8")),
        "branch_header": header,
    }


# --------------------------------------------------------------------------- validate


def _norm(path: str) -> str:
    text = str(path).replace("\\", "/")
    return text.lower() if os.name == "nt" or (len(text) > 1 and text[1] == ":") else text


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(_norm(path), _norm(pattern))


def _forbidden(path: str, location: str) -> bool:
    if _has_glob(location):
        return _matches(path, location)
    candidate, prefix = _norm(path).rstrip("/"), _norm(location).rstrip("/")
    return candidate == prefix or candidate.startswith(prefix + "/")


def validate_report(directory: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Check an agent report against a frozen assignment. Never raises for an invalid report."""

    root = Path(directory)
    spec = load_assignment(root / ASSIGNMENT_FILE)
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{report_path}: not valid JSON: {exc}") from exc
    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)

    schema_problems = schema_errors(report, REPORT_SCHEMA)
    failures += [f"schema: {problem}" for problem in schema_problems]
    if not isinstance(report, dict):
        return {"valid": False, "assignment_id": spec["id"], "failures": failures, "tasks": {}}

    # Pre-registration: the report must quote the frozen spec hash, and nothing may have changed since.
    frozen_path = root / FROZEN_FILE
    frozen: dict[str, Any] = {}
    if not frozen_path.is_file():
        fail(f"frozen: {frozen_path} is missing; the assignment was never frozen")
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if report.get("frozen_sha256") != frozen.get("assignment_yaml_sha256"):
            fail(
                f"frozen: report frozen_sha256 {report.get('frozen_sha256')} != frozen.json "
                f"assignment_yaml_sha256 {frozen.get('assignment_yaml_sha256')}"
            )
        if sha256_file(root / ASSIGNMENT_FILE) != frozen.get("assignment_yaml_sha256"):
            fail(f"frozen: {ASSIGNMENT_FILE} was edited after freezing")
        rendered = root / RENDERED_FILE
        if not rendered.is_file() or sha256_file(rendered) != frozen.get("assignment_md_sha256"):
            fail(f"frozen: {RENDERED_FILE} is missing or was edited after freezing")
    if report.get("assignment_id") != spec["id"]:
        fail(f"assignment_id: report says {report.get('assignment_id')!r}, spec is {spec['id']!r}")

    # One verdict per task, from that task's vocabulary.
    reported = [task for task in report.get("tasks", []) if isinstance(task, dict)]
    task_status: dict[str, dict[str, Any]] = {}
    spec_ids = [task["id"] for task in spec["tasks"]]
    for task in spec["tasks"]:
        entries = [entry for entry in reported if entry.get("id") == task["id"]]
        status: dict[str, Any] = {"verdict": None, "status": "ok"}
        if not entries:
            status["status"] = "missing"
            fail(f"task {task['id']}: no verdict reported")
        elif len(entries) > 1:
            status["status"] = "duplicate"
            fail(f"task {task['id']}: {len(entries)} entries; exactly one verdict is allowed")
        else:
            verdict = entries[0].get("verdict")
            status["verdict"] = verdict
            if verdict not in task["verdicts"]:
                status["status"] = "bad_verdict"
                fail(f"task {task['id']}: verdict {verdict!r} not in {task['verdicts']}")
        task_status[task["id"]] = status
    for entry in reported:
        if entry.get("id") not in spec_ids:
            fail(f"task {entry.get('id')!r}: not a task of this assignment")

    # Every value carries a source.
    for entry in reported:
        for f_index, finding in enumerate(entry.get("findings", []) or []):
            for v_index, value in enumerate((finding or {}).get("values", []) or []):
                if not str((value or {}).get("source") or "").strip():
                    fail(f"task {entry.get('id')}: findings[{f_index}].values[{v_index}] has an empty source")
                    if entry.get("id") in task_status and task_status[entry["id"]]["status"] == "ok":
                        task_status[entry["id"]]["status"] = "unsourced_value"

    # Target repository untouched.
    git_state = report.get("git_state") or {}
    before, after = git_state.get("before") or {}, git_state.get("after") or {}
    if before and after:
        if before.get("head") != after.get("head"):
            fail(f"git_state: HEAD moved {before.get('head')} -> {after.get('head')}")
        if before.get("porcelain_sha256") != after.get("porcelain_sha256"):
            fail("git_state: working-tree porcelain hash changed between before and after")
        if before.get("branch_header") != after.get("branch_header"):
            explained = any("branch_header" in str(item) for item in report.get("deviations") or [])
            if not explained:
                fail("git_state: branch_header changed and no deviation mentions 'branch_header'")

    # Writes stayed inside permissions.
    permissions = spec["permissions"]
    for item in report.get("files_written", []) or []:
        path = str((item or {}).get("path", ""))
        resolved = Path(path) if Path(path).is_absolute() else root / path
        if not any(_matches(str(resolved), pattern) or _matches(path, pattern) for pattern in permissions["write"]):
            fail(f"files_written: {path} matches no permissions.write glob")
        for location in permissions["forbid"]:
            if _forbidden(str(resolved), location):
                fail(f"files_written: {path} is inside forbidden location {location}")
        if not resolved.is_file():
            fail(f"files_written: {path} does not exist")
        elif sha256_file(resolved) != (item or {}).get("sha256"):
            fail(f"files_written: {path} sha256 does not match the file on disk")

    for field in ("not_assessed", "deviations"):
        if field not in report:
            fail(f"{field}: field must be present (it may be empty)")

    return {
        "valid": not failures,
        "assignment_id": spec["id"],
        "report": str(report_path),
        "frozen_sha256": frozen.get("assignment_yaml_sha256"),
        "tasks": task_status,
        "failures": failures,
    }


# --------------------------------------------------------------------------- CLI


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_render(args: argparse.Namespace) -> int:
    _print(render_to_file(args.assignment, args.output, force=args.force))
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    _print(freeze_assignment(args.directory))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = Path(args.report)
    if not report.is_file():
        raise FileNotFoundError(f"report not found: {report}")
    payload = validate_report(args.directory, report)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    return 0


def cmd_git_snapshot(args: argparse.Namespace) -> int:
    _print(git_snapshot(args.repo))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    values: dict[str, str] = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigurationError(f"--set expects KEY=VALUE, got {item!r}")
        values[key] = value
    _print(init_assignment(args.directory, args.template, values, force=args.force))
    return 0


def register_commands(commands: Any) -> None:
    """Attach init/render/freeze/validate/git-snapshot to an ``iface assignment`` subparsers object."""

    init = commands.add_parser("init", help="Write assignment.yaml from a bundled template (string.Template)")
    init.add_argument("directory", help="Assignment directory to create")
    init.add_argument("--template", default="due_diligence_v1", help="Bundled template (default due_diligence_v1)")
    init.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Fill a ${KEY} placeholder")
    init.add_argument("--force", action="store_true", help="Overwrite an existing assignment.yaml")
    init.set_defaults(func=cmd_init)

    render = commands.add_parser("render", help="Render assignment.yaml to ASSIGNMENT.md (deterministic)")
    render.add_argument("assignment", help="Path to assignment.yaml")
    render.add_argument("-o", "--output", help="Output Markdown path (default: ASSIGNMENT.md next to the YAML)")
    render.add_argument("--force", action="store_true", help="Overwrite a differing ASSIGNMENT.md (never once frozen)")
    render.set_defaults(func=cmd_render)

    freeze = commands.add_parser("freeze", help="Pre-register the assignment by hash in frozen.json (once)")
    freeze.add_argument("directory", help="Assignment directory holding assignment.yaml and ASSIGNMENT.md")
    freeze.set_defaults(func=cmd_freeze)

    validate = commands.add_parser(
        "validate", help="Check an agent report against the frozen assignment; exit 0 whether valid or not"
    )
    validate.add_argument("directory", help="Assignment directory")
    validate.add_argument("--report", required=True, help="report.json returned by the agent")
    validate.add_argument("-o", "--output", help="Also write the verdict JSON to this file")
    validate.set_defaults(func=cmd_validate)

    snapshot = commands.add_parser(
        "git-snapshot", help="Read-only HEAD/porcelain hash/branch header for a report's git_state"
    )
    snapshot.add_argument("repo", help="Git working tree to snapshot (read-only)")
    snapshot.set_defaults(func=cmd_git_snapshot)
