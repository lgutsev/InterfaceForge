from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from interfaceforge.errors import SafetyError
from interfaceforge.rescue_prepare import prepare_rescue, write_preparation


def fixture(root: Path) -> tuple[Path, Path, Path]:
    mapped = root / "map.yaml"
    policy = root / "policy.yaml"
    policy.write_text("schema_version: 1\nreviews: {}\n")
    mapped.write_text(yaml.safe_dump({"sources": [
        {"target": "interface/300K/Real", "source": "${REMOTE_ONLY}/interface"},
        {"target": "bulk/TiO-Ideal", "source": "${REMOTE_ONLY}/bulk", "enabled": False}],
        "collection": {"stride": 5, "balance_frames_per_leaf": False, "include_virial": True}}))
    source_id = "mapped/interface/300K/Real/N_Term/example"
    rows = [{"source_id": source_id, "role": "train", "status": "REVIEW", "fingerprint": "a" * 64,
             "run": "/remote/interface", "files": {"OUTCAR": {"sha256": "b" * 64}},
             "outcar": {"frames": 11}, "labels": {"frames": 11}, "issues": []},
            {"source_id": "mapped/bulk/TiO-Ideal/.", "role": "train", "status": "QUARANTINED",
             "fingerprint": "c" * 64, "run": "/remote/bulk", "files": {"OUTCAR": {"sha256": "d" * 64}},
             "outcar": {"frames": 11}, "labels": {"frames": 11}, "issues": [], "reason": "bad MD"},
            {"source_id": "a2_reference/frame_0", "role": "reference", "status": "REVIEW",
             "fingerprint": "e" * 64, "run": "/remote/reference", "files": {}, "issues": []}]
    report = root / "source_audit.json"
    report.write_text(json.dumps({"schema_version": 1, "labels_scanned": True, "root_errors": [],
                                 "policy_sha256": sha256(policy.read_bytes()).hexdigest(),
                                 "mapped_config": {"sha256": sha256(mapped.read_bytes()).hexdigest()},
                                 "status": "BLOCKED", "rows": rows, "duplicate_outputs": []}))
    (root / "frames").mkdir()
    history = [{"source_frame": i, "ionic_step": i + 1, "scf_steps": 5,
                "final_rms_c": None, "last_rms_c": float(i)} for i in range(11)]
    history[9]["last_rms_c"] = 999.0  # Worst unretained frame must not be selected.
    evidence = root / "frames" / (sha256(source_id.encode()).hexdigest()[:16] + ".scf.json")
    evidence.write_text(json.dumps(history))
    return report, mapped, policy


def ledger(root: Path, report: Path) -> Path:
    path = root / "historical.json"
    path.write_text(json.dumps({"schema_version": 1, "report_sha256": sha256(report.read_bytes()).hexdigest(),
                               "frames": [{"source_id": "mapped/interface/300K/Real/N_Term/example",
                                           "source_frame": i, "outcar_sha256": "b" * 64, "split": split}
                                          for i, split in zip([0, 5, 10], ["train", "valid", "test"], strict=True)]}))
    return path


def test_offline_selection_and_quarantine(tmp_path):
    paths = fixture(tmp_path)
    plan = prepare_rescue(*paths)
    assert plan["counts"]["candidate_sources"] == 1
    assert plan["counts"]["candidate_frames"] == 3
    assert plan["counts"]["quarantined_records"] == 1
    assert [(p["kind"], p["source_frame"]) for p in plan["probes"]] == [("typical", 5), ("worst", 10)]
    assert all(p["residual_kind"] == "last_printed_rms_c" for p in plan["probes"])
    assert plan["historical_membership"]["status"] == "MISSING"
    assert plan["training_ready"] is False
    assert plan["raw_sources_rechecked"] is False
    assert prepare_rescue(*paths) == plan
    output = tmp_path / "prepared"
    write_preparation(plan, output)
    assert (output / "source_inventory.csv").is_file()
    assert (output / "probe_selection.csv").is_file()
    with pytest.raises(FileExistsError):
        write_preparation(plan, output)


@pytest.mark.parametrize("change", ["map", "policy", "root_error", "logs_only", "duplicate_source"])
def test_snapshot_safety(tmp_path, change):
    report, mapped, policy = fixture(tmp_path)
    if change == "map":
        mapped.write_text(mapped.read_text() + "# altered\n")
    elif change == "policy":
        policy.write_text(policy.read_text() + "# altered\n")
    else:
        data = json.loads(report.read_text())
        if change == "root_error":
            data["root_errors"] = ["missing required root"]
        elif change == "logs_only":
            data["labels_scanned"] = False
        else:
            data["rows"].append(data["rows"][0])
        report.write_text(json.dumps(data))
    with pytest.raises(SafetyError):
        prepare_rescue(report, mapped, policy)


def test_reenabled_quarantine_fails_even_when_map_hash_is_updated(tmp_path):
    report, mapped, policy = fixture(tmp_path)
    config = yaml.safe_load(mapped.read_text())
    config["sources"][1]["enabled"] = True
    mapped.write_text(yaml.safe_dump(config))
    data = json.loads(report.read_text())
    data["mapped_config"]["sha256"] = sha256(mapped.read_bytes()).hexdigest()
    report.write_text(json.dumps(data))
    with pytest.raises(SafetyError, match="Quarantined source is enabled"):
        prepare_rescue(report, mapped, policy)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered"])
def test_scf_evidence_integrity(tmp_path, change):
    paths = fixture(tmp_path)
    evidence = next((tmp_path / "frames").glob("*.scf.json"))
    if change == "missing":
        evidence.unlink()
    else:
        history = json.loads(evidence.read_text())
        if change == "duplicate":
            history[1]["source_frame"] = 0
        else:
            history.reverse()
        evidence.write_text(json.dumps(history))
    with pytest.raises(SafetyError):
        prepare_rescue(*paths)


def test_missing_residual_holds_all_probe_selection(tmp_path):
    paths = fixture(tmp_path)
    evidence = next((tmp_path / "frames").glob("*.scf.json"))
    history = json.loads(evidence.read_text())
    history[5]["last_rms_c"] = None
    evidence.write_text(json.dumps(history))
    plan = prepare_rescue(*paths)
    assert plan["probe_holds"]
    assert plan["probes"] == []
    assert plan["training_ready"] is False


def test_historical_membership_preserved_and_exclusions_counted(tmp_path):
    paths = fixture(tmp_path)
    membership = ledger(tmp_path, paths[0])
    data = json.loads(membership.read_text())
    data["frames"].append({"source_id": "mapped/bulk/TiO-Ideal/.", "source_frame": 0,
                           "outcar_sha256": "d" * 64, "split": "train"})
    membership.write_text(json.dumps(data))
    plan = prepare_rescue(*paths, membership)
    assert plan["historical_membership"]["active_split_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert plan["historical_membership"]["excluded_historical_frames"] == 1
    assert plan["training_ready"] is False


@pytest.mark.parametrize("change", ["hash", "duplicate", "missing", "reference", "stride", "bounds", "split"])
def test_historical_membership_rejects_leakage_or_wrong_identity(tmp_path, change):
    paths = fixture(tmp_path)
    membership = ledger(tmp_path, paths[0])
    data = json.loads(membership.read_text())
    if change == "hash":
        data["frames"][0]["outcar_sha256"] = "f" * 64
    elif change == "duplicate":
        data["frames"].append({**data["frames"][0], "split": "test"})
    elif change == "missing":
        data["frames"].pop()
    elif change == "reference":
        data["frames"][0]["source_id"] = "a2_reference/frame_0"
    elif change == "stride":
        data["frames"][0]["source_frame"] = 1
    elif change == "bounds":
        data["frames"][0]["source_frame"] = 15
    else:
        data["frames"][0]["split"] = "holdout"
    membership.write_text(json.dumps(data))
    with pytest.raises(SafetyError):
        prepare_rescue(*paths, membership)


@pytest.mark.parametrize("residual", [-1.0, float("nan"), float("inf"), True])
def test_invalid_residuals_hold_selection(tmp_path, residual):
    paths = fixture(tmp_path)
    evidence = next((tmp_path / "frames").glob("*.scf.json"))
    history = json.loads(evidence.read_text())
    history[5]["last_rms_c"] = residual
    evidence.write_text(json.dumps(history))
    plan = prepare_rescue(*paths)
    assert plan["probes"] == []
    assert plan["probe_holds"]


def test_floating_source_index_rejected(tmp_path):
    paths = fixture(tmp_path)
    evidence = next((tmp_path / "frames").glob("*.scf.json"))
    history = json.loads(evidence.read_text())
    history[0]["source_frame"] = 0.0
    evidence.write_text(json.dumps(history))
    with pytest.raises(SafetyError):
        prepare_rescue(*paths)


def test_membership_wrong_report_rejected(tmp_path):
    paths = fixture(tmp_path)
    membership = ledger(tmp_path, paths[0])
    data = json.loads(membership.read_text())
    data["report_sha256"] = "0" * 64
    membership.write_text(json.dumps(data))
    with pytest.raises(SafetyError, match="bind this audit report hash"):
        prepare_rescue(*paths, membership)
