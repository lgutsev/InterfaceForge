"""Bundled literature reference profiles and their import into a campaign.

A *reference profile* is a small YAML file describing one published paper's
reported interface quantities (work of adhesion, surface energy, ...) together
with the DFT protocol that produced them, so a campaign can automatically check
a computed value against the literature.

Profiles bundled with InterfaceForge live in ``interfaceforge/references/``. A
campaign opts in::

    validation:
      reference_profiles: [sharifi2026]

``load_campaign`` calls :func:`resolve_reference_profiles` to expand every named
profile into ``validation.references`` entries (one per quantity). Hand-written
``validation.references`` entries are still honoured and win over a profile
entry with the same ``(key, quantity)``.

``iface reference list`` / ``iface reference show <name>`` inspect what is
available and exactly what a profile expands to.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

_SUFFIX = ".yaml"


def _bundled_root() -> Any:
    return resources.files("interfaceforge").joinpath("references")


def list_reference_profiles() -> list[str]:
    """Names of the reference profiles bundled with InterfaceForge."""

    root = _bundled_root()
    names: list[str] = []
    for entry in root.iterdir():
        if entry.name.endswith(_SUFFIX):
            names.append(entry.name[: -len(_SUFFIX)])
    return sorted(names)


def _read_yaml(text: str, where: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in reference profile {where}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Reference profile {where} must be a mapping")
    return data


def load_reference_profile(name_or_path: str | Path) -> dict[str, Any]:
    """Load a bundled profile by name, or a profile file by path.

    Validation here is deliberately shallow: it checks the profile envelope
    (``schema_version``, ``key``, a non-empty ``references`` list). The per-entry
    ``quantity``/``values`` shape is validated once, together with any
    hand-written entries, when the campaign is loaded.
    """

    candidate = Path(name_or_path)
    if candidate.suffix and candidate.exists():
        text = candidate.read_text(encoding="utf-8")
        where = str(candidate)
    else:
        name = str(name_or_path)
        if name.endswith(_SUFFIX):
            name = name[: -len(_SUFFIX)]
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ConfigurationError(f"Invalid reference profile name {name_or_path!r}")
        resource = _bundled_root().joinpath(f"{name}{_SUFFIX}")
        if not resource.is_file():
            available = ", ".join(list_reference_profiles()) or "(none)"
            raise ConfigurationError(
                f"Unknown reference profile {name!r}; bundled profiles: {available}"
            )
        text = resource.read_text(encoding="utf-8")
        where = f"{name} (bundled)"

    data = _read_yaml(text, where)
    if int(data.get("schema_version", 0)) != 1:
        raise ConfigurationError(f"Reference profile {where} needs schema_version: 1")
    key = str(data.get("key", "")).strip()
    if not key:
        raise ConfigurationError(f"Reference profile {where} needs a non-empty key")
    entries = data.get("references")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(
            f"Reference profile {where}.references must be a non-empty list"
        )
    data["key"] = key
    return data


def expand_reference_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a profile into ``validation.references`` entries (one per quantity).

    ``key``, ``doi``, ``citation`` and ``method`` are copied onto every entry so
    each is self-describing once merged into the campaign.
    """

    key = str(profile["key"]).strip()
    shared: dict[str, Any] = {"key": key}
    for field in ("doi", "citation"):
        if profile.get(field) is not None:
            shared[field] = str(profile[field]).strip()
    if profile.get("method") is not None:
        if not isinstance(profile["method"], dict):
            raise ConfigurationError(f"Reference profile {key}.method must be a mapping")
        shared["method"] = dict(profile["method"])

    out: list[dict[str, Any]] = []
    for index, item in enumerate(profile["references"]):
        if not isinstance(item, dict):
            raise ConfigurationError(
                f"Reference profile {key}.references[{index}] must be a mapping"
            )
        out.append({**shared, **item})
    return out


def resolve_reference_profiles(names: list[str]) -> list[dict[str, Any]]:
    """Expand a list of profile names/paths into ``validation.references`` entries."""

    resolved: list[dict[str, Any]] = []
    for name in names:
        resolved.extend(expand_reference_profile(load_reference_profile(name)))
    return resolved


# --- activation: adding a profile to validation.reference_profiles in place -----
#
# The campaign file is hand-authored and usually carries comments, so a
# safe_load/safe_dump round trip is not acceptable. Instead the token is spliced
# into the text with the minimum possible edit, and the result is refused unless
# it still parses and still passes load_campaign(). Anything the splice cannot do
# unambiguously (an inline `validation: {...}`, a non-list `reference_profiles`)
# raises with a "edit by hand" message rather than risk corrupting the file.


def _top_key_line(lines: list[str], key: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:(.*)$")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            return index
    return None


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _child_indent(lines: list[str], parent_index: int) -> str:
    parent = _line_indent(lines[parent_index])
    for line in lines[parent_index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = _line_indent(line)
        return " " * indent if indent > parent else "  "
    return "  "


def _inline_value(line: str) -> str:
    body = line.split(":", 1)[1] if ":" in line else ""
    return body.split("#", 1)[0].strip()


def _splice_reference_profile(text: str, parsed: dict[str, Any], token: str) -> str:
    lines = text.splitlines(keepends=True)
    newline = "\n"
    if lines and lines[-1].endswith("\r\n"):
        newline = "\r\n"

    validation = parsed.get("validation")
    if validation is not None and not isinstance(validation, dict):
        raise ConfigurationError(
            "campaign 'validation' is not a mapping; add reference_profiles by hand"
        )

    if "validation" not in parsed:
        prefix = "" if not text or text.endswith(("\n", "\r\n")) else newline
        block = (
            f"{prefix}{newline}validation:{newline}"
            f"  reference_profiles:{newline}    - {token}{newline}"
        )
        return text + block

    val_index = _top_key_line(lines, "validation")
    if val_index is None:
        raise ConfigurationError(
            "could not locate the 'validation:' line to edit; add reference_profiles by hand"
        )
    if _inline_value(lines[val_index]):
        raise ConfigurationError(
            "'validation' is written inline; add reference_profiles by hand"
        )

    child = _child_indent(lines, val_index)
    profiles = validation.get("reference_profiles") if isinstance(validation, dict) else None

    if profiles is None:
        insertion = (
            f"{child}reference_profiles:{newline}{child}  - {token}{newline}"
        )
        lines.insert(val_index + 1, insertion)
        return "".join(lines)

    if not isinstance(profiles, list):
        raise ConfigurationError(
            "'validation.reference_profiles' is not a list; edit by hand"
        )

    rp_pattern = re.compile(rf"^{child}reference_profiles\s*:(.*)$")
    rp_index = next(
        (i for i in range(val_index + 1, len(lines)) if rp_pattern.match(lines[i])),
        None,
    )
    if rp_index is None:
        raise ConfigurationError(
            "could not locate the 'reference_profiles:' line; edit by hand"
        )

    remainder = rp_pattern.match(lines[rp_index]).group(1).split("#", 1)[0].strip()
    if remainder.startswith("["):
        closing = remainder.rfind("]")
        if closing == -1:
            raise ConfigurationError(
                "'reference_profiles' spans multiple lines; edit by hand"
            )
        inside = remainder[1:closing].strip()
        items = f"{inside}, {token}" if inside else token
        line = lines[rp_index]
        head = line[: line.index("reference_profiles")]
        trailing = line[line.index(remainder) + closing + 1 :]
        lines[rp_index] = f"{head}reference_profiles: [{items}]{trailing}"
        return "".join(lines)

    # Block list (or an empty value): match the existing items' own indent.
    item_pattern = re.compile(r"^(\s*)-\s")
    first_item_indent: str | None = None
    last_item = rp_index
    for i in range(rp_index + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = item_pattern.match(lines[i])
        if match and len(match.group(1)) >= len(child):
            if first_item_indent is None:
                first_item_indent = match.group(1)
            if match.group(1) == first_item_indent:
                last_item = i
                continue
        break
    if first_item_indent is None:
        if remainder:
            raise ConfigurationError(
                "'reference_profiles' has an unexpected inline value; edit by hand"
            )
        lines.insert(rp_index + 1, f"{child}  - {token}{newline}")
        return "".join(lines)
    lines.insert(last_item + 1, f"{first_item_indent}- {token}{newline}")
    return "".join(lines)


def _verify_campaign(campaign_path: Path, text: str) -> None:
    from .config import load_campaign

    probe = campaign_path.parent / f".{campaign_path.name}.ifactivate"
    probe.write_text(text, encoding="utf-8")
    try:
        load_campaign(probe)
    finally:
        probe.unlink(missing_ok=True)


def activate_reference_profile(
    campaign_path: str | Path, name: str, *, write: bool = False
) -> dict[str, Any]:
    """Add ``name`` to ``validation.reference_profiles`` in a campaign file.

    Returns a report. With ``write=False`` (the default) the campaign is not
    touched and ``resulting_text`` carries the proposed file; with
    ``write=True`` the edit is applied atomically once it is confirmed to still
    parse and still load.
    """

    path = Path(campaign_path).expanduser()
    if not path.is_file():
        raise ConfigurationError(f"Campaign file does not exist: {path}")

    profile = load_reference_profile(name)  # fail early on a bad name/path
    token = str(name)

    original = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(original)
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ConfigurationError(f"{path} is not a YAML mapping")

    validation = parsed.get("validation")
    existing: list[str] = []
    if isinstance(validation, dict) and isinstance(validation.get("reference_profiles"), list):
        existing = [str(item) for item in validation["reference_profiles"]]
    already_active = token in existing

    updated = original if already_active else _splice_reference_profile(original, parsed, token)
    changed = updated != original
    if changed:
        _verify_campaign(path, updated)
    if write and changed:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(path)

    return {
        "campaign": str(path),
        "profile": token,
        "already_active": already_active,
        "changed": changed,
        "written": bool(write and changed),
        "reference_profiles": existing + ([] if already_active else [token]),
        "expands_to": sorted({entry["quantity"] for entry in expand_reference_profile(profile)}),
        "resulting_text": None if (write and changed) else updated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iface reference", description=__doc__)
    sub = parser.add_subparsers(dest="reference_command", required=True)
    sub.add_parser("list", help="List the bundled reference profiles")
    show = sub.add_parser(
        "show",
        help="Print a profile and the validation.references entries it expands to",
    )
    show.add_argument("name", help="Bundled profile name, or a path to a profile YAML file")
    activate = sub.add_parser(
        "activate",
        help="Add a profile to validation.reference_profiles in a campaign file",
    )
    activate.add_argument("name", help="Bundled profile name, or a path to a profile YAML file")
    activate.add_argument("-c", "--campaign", default="campaign.yaml")
    activate.add_argument(
        "--write",
        action="store_true",
        help="Apply the edit (default: print the resulting file and change nothing)",
    )
    args = parser.parse_args(argv)

    if args.reference_command == "list":
        for name in list_reference_profiles():
            print(name)
        return 0
    if args.reference_command == "activate":
        print(json.dumps(activate_reference_profile(args.campaign, args.name, write=args.write), indent=2))
        return 0
    profile = load_reference_profile(args.name)
    print(
        json.dumps(
            {"profile": profile, "validation_references": expand_reference_profile(profile)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
