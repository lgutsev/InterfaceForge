from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from interfaceforge.errors import SafetyError
from interfaceforge.leaf_collect import collect_leaf_dataset
from interfaceforge.mapped_collect import discover_mapped_leaves, load_mapped_config, run_mapped_collection
from interfaceforge.source_audit import (
    audit_sources,
    parse_oszicar,
    require_source_admission,
    scan_labels,
    scan_outcar,
    source_snapshot,
)


def fixture(tmp_path: Path, *, role: str = "train") -> tuple[Path, Path]:
    run = tmp_path / "source" / "run"
    run.mkdir(parents=True)
    (run / "INCAR").write_text("ENCUT=520\nTEBEG=300\nTEEND=300\nNELM=60\n")
    (run / "KPOINTS").write_text("Gamma\n0\nG\n1 1 1\n")
    (run / "POSCAR").write_text("fixture geometry")
    (run / "POTCAR").write_text("fixture identity only")
    (run / "OSZICAR").write_text(
        "DAV: 1 -1.0 -.1 -.1 10 .01 .00001\nDAV: 2 -1.0 -.000001 -.000001 10 .00001\n1 T= 300 F= -1.0 E0= -1.0\n"
    )
    (run / "OUTCAR").write_text(
        "vasp.6.6.1\n ENCUT = 520; NELM = 60; NKPTS = 1; NIONS = 2\n"
        "POSITION TOTAL-FORCE\n General timing and accounting informations for this job:\n"
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump({"schema_version": 1, "roots": [{"id": "bulk", "path": str(run.parent), "role": role}]})
    )
    return run, policy


def labels_mock(*_args, **_kwargs):
    return {"frames": 1, "symbols": ["Ti", "O"]}, []


def reviewed_report(tmp_path: Path, *, role: str = "train") -> tuple[Path, Path, Path]:
    run, policy = fixture(tmp_path, role=role)
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        report = audit_sources(policy, tmp_path / "first")
        row = report["rows"][0]
        config = yaml.safe_load(policy.read_text())
        config["reviews"] = {
            row["source_id"]: {
                "decision": "accept",
                "fingerprint": row["fingerprint"],
                "reviewer": "test",
                "evidence": "paired k-convergence test fixture",
                "acknowledged_codes": [i["code"] for i in row["issues"]],
            }
        }
        policy.write_text(yaml.safe_dump(config))
        report = audit_sources(policy, tmp_path / "reviewed")
    assert report["rows"][0]["status"] == "ACCEPTED"
    return run, policy, tmp_path / "reviewed" / "source_audit.json"


def test_false_scf_convergence_detected_with_missing_last_residual(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text(
        "RMM: 40 -1 -.1 -.1 10 .01 .00001\n"
        "RMM: 41 4 .1 .1 10 .01 .032\n"
        "RMM: 42 4 -.000001 -.000001 10 .00001\n"
        "1 F= 4 E0= 4\n"
    )
    frames, issues = parse_oszicar(path)
    assert frames[0]["last_rms_c"] == 0.032
    assert frames[0]["residual_jumps"] == 1
    assert {i["code"] for i in issues} >= {"SCF_RESIDUAL", "SCF_RESIDUAL_JUMP"}


def test_nelm_and_unfinished_scf(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text("DAV: 60 -1 .1 .1 10 .01 .01\n1 F= -1 E0= -1\nDAV: 1 -1 .1 .1 10 .01 .01\n")
    _, issues = parse_oszicar(path)
    assert {i["code"] for i in issues} >= {"SCF_LIMIT", "UNFINISHED_SCF"}


def test_nan_does_not_silently_pass(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text("DAV: 1 nan .1 .1 10 .01 .01\n1 F= nan E0= -1\n")
    _, issues = parse_oszicar(path)
    assert any(i["hard"] for i in issues)


def test_oszicar_signed_columns_can_touch_and_algorithm_colon_can_be_spaced(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text(
        "CG :  1   -.13238703E+04-.132E+04-.934E+02  56  .28E+02\n"
        "CG :  2   -.13391360D+04-.152D+02-.982D+01  82  .54D+01 .00001\n"
        "1 F= -.13391360E+04 E0= -.13391360E+04\n"
    )
    frames, issues = parse_oszicar(path)
    assert frames[0]["scf_steps"] == 2 and frames[0]["last_rms_c"] == 0.00001
    assert not any(i["hard"] for i in issues)


@pytest.mark.parametrize("payload", ["-1 .1 .1 10 nan", "-1 .1 .1 10 *****", "-1 .1 .1 10 .01 garbage"])
def test_bad_scf_columns_remain_hard_and_report_original_line(tmp_path, payload):
    path = tmp_path / "OSZICAR"
    path.write_text(f"DAV: 1 {payload}\n1 F= -1 E0= -1\n")
    _, issues = parse_oszicar(path)
    malformed = next(i for i in issues if i["code"] == "MALFORMED_SCF")
    assert malformed["hard"] and payload in malformed["detail"]


def test_outcar_logical_lreal_echo_is_review_but_false_is_real_mismatch(tmp_path):
    from interfaceforge.source_audit import equivalent

    assert not equivalent("Auto", ".TRUE.", "LREAL")
    run, policy = fixture(tmp_path)
    (run / "INCAR").write_text((run / "INCAR").read_text() + "LREAL=Auto\n")
    original = (run / "OUTCAR").read_text()
    (run / "OUTCAR").write_text(original + "LREAL = T\n")
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        row = audit_sources(policy, tmp_path / "logical")["rows"][0]
    assert row["status"] == "REVIEW"
    assert any(i["code"] == "LREAL_MODE_UNVERIFIED" for i in row["issues"])
    (run / "OUTCAR").write_text(original + "LREAL = F\n")
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        row = audit_sources(policy, tmp_path / "false")["rows"][0]
    assert row["status"] == "FAILED"
    assert any(i["code"] == "EXECUTED_INPUT_MISMATCH" for i in row["issues"])


def test_completed_run_still_requires_scientific_review(tmp_path):
    _, policy = fixture(tmp_path)
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        report = audit_sources(policy, tmp_path / "audit")
    assert report["rows"][0]["status"] == "REVIEW"
    assert report["status"] == "BLOCKED"


def test_log_only_cannot_qualify_even_with_review(tmp_path):
    run, policy = fixture(tmp_path)
    report = audit_sources(policy, tmp_path / "audit", labels=False)
    assert report["rows"][0]["status"] == "FAILED"
    with pytest.raises(SafetyError, match="logs-only"):
        require_source_admission(tmp_path / "audit/source_audit.json", [run / "OUTCAR"])


def test_known_quarantine_is_inventory_not_raw_deletion(tmp_path):
    run, policy = fixture(tmp_path)
    config = yaml.safe_load(policy.read_text())
    config["quarantine"] = [{"pattern": "run", "reason": "bad trajectory"}]
    policy.write_text(yaml.safe_dump(config))
    report = audit_sources(policy, tmp_path / "audit")
    assert report["rows"][0]["status"] == "QUARANTINED"
    assert report["rows"][0]["files"]["OUTCAR"]["sha256"]
    assert (run / "OUTCAR").exists()
    with pytest.raises(SafetyError, match="not accepted"):
        require_source_admission(tmp_path / "audit/source_audit.json", [run / "OUTCAR"])


def test_acceptance_bound_to_content_and_policy(tmp_path):
    run, policy, report = reviewed_report(tmp_path)
    assert require_source_admission(report, [run / "OUTCAR"])["admitted_sources"] == ["bulk/run"]
    (run / "OSZICAR").write_text("changed")
    with pytest.raises(SafetyError, match="changed since"):
        require_source_admission(report, [run / "OUTCAR"])
    policy.write_text(policy.read_text() + "\n# policy change\n")
    with pytest.raises(SafetyError, match="policy changed"):
        require_source_admission(report, [run / "OUTCAR"])


def test_holdout_and_reference_roles_never_admitted(tmp_path):
    run, _, report = reviewed_report(tmp_path, role="reference")
    with pytest.raises(SafetyError, match="not accepted"):
        require_source_admission(report, [run / "OUTCAR"])


def test_rejected_source_does_not_delete_existing_export(tmp_path):
    run, policy = fixture(tmp_path)
    audit_sources(policy, tmp_path / "audit", labels=False)
    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "old.extxyz"
    sentinel.write_text("preserve")
    with pytest.raises(SafetyError):
        collect_leaf_dataset(
            run.parent, output, engine="mace", force=True, source_audit=tmp_path / "audit/source_audit.json"
        )
    assert sentinel.read_text() == "preserve"


def test_input_edit_invalidates_review_on_rerun(tmp_path):
    run, policy, _ = reviewed_report(tmp_path)
    (run / "POSCAR").write_text("different geometry")
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        report = audit_sources(policy, tmp_path / "changed")
    assert report["rows"][0]["status"] == "REVIEW"


def test_outcar_gzip_and_truncated_detection(tmp_path):
    path = tmp_path / "OUTCAR.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("vasp.6.6.1\nNKPTS = 1\nPOSITION TOTAL-FORCE\n")
    result = scan_outcar(path)
    assert result["frames"] == 1 and not result["completed"]
    assert result["nkpts"] == 1


def test_all_frame_labels_and_short_contact(tmp_path):
    atoms = Atoms("TiO", positions=[[0, 0, 0], [0.2, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, free_energy=-1.0, forces=np.zeros((2, 3)), stress=np.zeros(6)
    )
    with patch("ase.io.iread", return_value=iter([atoms, atoms])):
        summary, issues = scan_labels(tmp_path / "OUTCAR", tmp_path / "frames.jsonl", {"contact_stride": 1})
    assert summary["frames"] == 2
    assert len([i for i in issues if i["code"] == "SHORT_CONTACT"]) == 2


def test_missing_root_blocks_admission(tmp_path):
    run, policy = fixture(tmp_path)
    config = yaml.safe_load(policy.read_text())
    config["roots"].append({"id": "missing", "path": str(tmp_path / "absent"), "role": "train"})
    policy.write_text(yaml.safe_dump(config))
    audit_sources(policy, tmp_path / "audit", labels=False)
    with pytest.raises(SafetyError, match="incomplete"):
        require_source_admission(tmp_path / "audit/source_audit.json", [run / "OUTCAR"])


def test_mapped_gate_rejects_before_staging(tmp_path):
    run, policy = fixture(tmp_path)
    audit_sources(policy, tmp_path / "audit", labels=False)
    config = tmp_path / "mapped.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_root": str(tmp_path / "campaign"),
                "source_audit": str(tmp_path / "audit/source_audit.json"),
                "sources": [{"source": str(run.parent), "target": "bulk"}],
            }
        )
    )
    with pytest.raises(SafetyError):
        run_mapped_collection(config, execute=True)
    assert not (tmp_path / "campaign").exists()


def test_a2_role_manifest_rejects_holdout_and_multiframe(tmp_path):
    run, _ = fixture(tmp_path)
    roles = run.parent / "roles.json"
    roles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "run": {
                        "role": "holdout",
                        "frames": 1,
                        "outcar_sha256": source_snapshot(run, run.parent)[0]["OUTCAR"]["sha256"],
                    }
                },
            }
        )
    )
    config = tmp_path / "mapped.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_root": str(tmp_path / "campaign"),
                "sources": [{"source": str(run.parent), "target": "a2/train", "train_role_manifest": str(roles)}],
            }
        )
    )
    with pytest.raises(SafetyError, match="train role"):
        discover_mapped_leaves(load_mapped_config(config))
    data = json.loads(roles.read_text())
    data["sources"]["run"]["role"] = "train"
    roles.write_text(json.dumps(data))
    assert len(discover_mapped_leaves(load_mapped_config(config))) == 1
    (run / "OUTCAR").write_text((run / "OUTCAR").read_text() + "POSITION TOTAL-FORCE\n")
    with pytest.raises(SafetyError, match="exactly one"):
        discover_mapped_leaves(load_mapped_config(config))


def test_inventory_includes_unmapped_run_without_double_count(tmp_path):
    run, policy = fixture(tmp_path)
    mapped = tmp_path / "map.yaml"
    mapped.write_text(yaml.safe_dump({"sources": [{"source": str(run), "target": "bulk/run"}]}))
    config = yaml.safe_load(policy.read_text())
    config["mapped_config"] = str(mapped)
    policy.write_text(yaml.safe_dump(config))
    report = audit_sources(policy, tmp_path / "audit", labels=False)
    assert len(report["rows"]) == 1
    assert report["mapped_config"]["sha256"]


def test_relative_sources_resolve_from_map_directory(tmp_path):
    run, policy = fixture(tmp_path)
    maps = tmp_path / "maps"
    maps.mkdir()
    mapped = maps / "map.yaml"
    mapped.write_text(yaml.safe_dump({"sources": [{"source": "../source", "target": "bulk"}]}))
    policy.write_text(yaml.safe_dump({"schema_version": 1, "mapped_config": "maps/map.yaml"}))
    report = audit_sources(policy, tmp_path / "audit", labels=False)
    assert not report["root_errors"]
    assert report["rows"][0]["run"] == str(run)


def test_quarantine_matches_source_root_and_disabled_map(tmp_path):
    run, policy = fixture(tmp_path)
    config = yaml.safe_load(policy.read_text())
    config["roots"][0]["path"] = str(run)
    config["quarantine"] = [{"pattern": "run", "reason": "quarantine root itself"}]
    policy.write_text(yaml.safe_dump(config))
    assert audit_sources(policy, tmp_path / "first", labels=False)["rows"][0]["status"] == "QUARANTINED"
    mapped = tmp_path / "mapped.yaml"
    mapped.write_text(
        yaml.safe_dump(
            {"sources": [{"source": str(run), "target": "bulk", "enabled": False, "reason": "disabled in map"}]}
        )
    )
    policy.write_text(yaml.safe_dump({"schema_version": 1, "mapped_config": str(mapped)}))
    row = audit_sources(policy, tmp_path / "second", labels=False)["rows"][0]
    assert row["status"] == "QUARANTINED" and row["reason"] == "disabled in map"


def test_policy_edit_during_scan_invalidates_report(tmp_path):
    _, policy = fixture(tmp_path)

    def changing_policy(*args, **kwargs):
        policy.write_text(policy.read_text() + "\n# changed during scan\n")
        return labels_mock()

    with patch("interfaceforge.source_audit.scan_labels", changing_policy):
        with pytest.raises(SafetyError, match="changed during scanning"):
            audit_sources(policy, tmp_path / "audit")
    assert not (tmp_path / "audit/source_audit.json").exists()


def test_source_edit_during_export_prevents_admission_manifest(tmp_path):
    run, _, report = reviewed_report(tmp_path)
    atoms = Atoms("TiO", positions=[[0, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    from interfaceforge.leaf_collect import _write_mace_frames

    def changing_source(*args, **kwargs):
        _write_mace_frames(*args, **kwargs)
        (run / "OSZICAR").write_text("edited during export")

    with patch("ase.io.iread", return_value=iter([atoms])):
        with patch("interfaceforge.leaf_collect._write_mace_frames", changing_source):
            with pytest.raises(SafetyError, match="changed since audit"):
                collect_leaf_dataset(run.parent, tmp_path / "export", engine="mace", source_audit=report)
    assert not (tmp_path / "export/leaf_manifest.json").exists()


def test_stale_staging_cannot_resurrect_quarantined_source(tmp_path):
    run, _ = fixture(tmp_path)
    campaign = tmp_path / "campaign"
    stale = campaign / "reference_runs/old_quarantined"
    stale.mkdir(parents=True)
    (stale / "OUTCAR").write_text("old")
    config = tmp_path / "map.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_root": str(campaign),
                "initialize_campaign": False,
                "sources": [{"source": str(run.parent), "target": "bulk"}],
            }
        )
    )
    with patch("interfaceforge.mapped_collect.stage_mapped_leaves", return_value={}):
        with pytest.raises(SafetyError, match="unmapped OUTCAR"):
            run_mapped_collection(config, execute=True, collect=True)
    assert not (campaign / "datasets").exists()


def test_vasp_header_abbreviations_are_not_false_input_mismatches(tmp_path):
    from interfaceforge.source_audit import equivalent

    assert equivalent("Accurate", "accura", "PREC")
    assert equivalent("Auto", "A", "LREAL")
    assert equivalent(".FALSE.", "F", "LREAL")
    assert not equivalent("Med", "accura", "PREC")
    run, policy = fixture(tmp_path)
    (run / "INCAR").write_text((run / "INCAR").read_text() + "PREC=Accurate\nLREAL=Auto\n")
    (run / "OUTCAR").write_text((run / "OUTCAR").read_text() + " PREC = accura; LREAL = A\n")
    with patch("interfaceforge.source_audit.scan_labels", labels_mock):
        report = audit_sources(policy, tmp_path / "audit")
    assert not any(i["code"] == "EXECUTED_INPUT_MISMATCH" for i in report["rows"][0]["issues"])


def test_read_real_outcar_with_ase_and_audit(tmp_path):
    import ase

    source = Path(ase.__file__).parent / "test/testdata/vasp/OUTCAR_example_1"
    if not source.is_file():
        pytest.skip("ASE installation does not include VASP test data")
    metadata = scan_outcar(source)
    summary, issues = scan_labels(source, tmp_path / "real_labels.jsonl", {})
    assert metadata["completed"] and metadata["frames"] == summary["frames"] == 1
    assert not any(i["hard"] for i in issues)
    record = json.loads((tmp_path / "real_labels.jsonl").read_text())
    assert record["natoms"] == 18 and record["free_energy_ev"] is not None


def test_qualified_real_outcar_exports_matching_energy_forces_and_virial(tmp_path):
    import shutil

    import ase
    from ase.io import read

    source = Path(ase.__file__).parent / "test/testdata/vasp/OUTCAR_example_1"
    if not source.is_file():
        pytest.skip("ASE installation does not include VASP test data")
    run, policy = fixture(tmp_path)
    shutil.copyfile(source, run / "OUTCAR")
    reference = read(str(source), format="vasp-out")
    reference.write(str(run / "POSCAR"), format="vasp")
    executed = scan_outcar(source)["executed"]
    (run / "INCAR").write_text("\n".join(f"{key}={value}" for key, value in executed.items()))
    (run / "OSZICAR").write_text("DAV: 1 -1 -.001 -.001 10 .001 .00001\n1 F= -1 E0= -1\n")
    first = audit_sources(policy, tmp_path / "first")
    row = first["rows"][0]
    config = yaml.safe_load(policy.read_text())
    config["reviews"] = {
        row["source_id"]: {
            "decision": "accept",
            "fingerprint": row["fingerprint"],
            "reviewer": "test",
            "evidence": "export regression fixture; not physical qualification",
            "acknowledged_codes": [item["code"] for item in row["issues"]],
        }
    }
    policy.write_text(yaml.safe_dump(config))
    assert audit_sources(policy, tmp_path / "reviewed")["rows"][0]["status"] == "ACCEPTED"
    for engine in ("mace", "deepmd"):
        result = collect_leaf_dataset(
            run.parent,
            tmp_path / engine,
            engine=engine,
            include_virial=True,
            split_mode="random-frame",
            ratios=[0.8, 0.1, 0.1],
            source_audit=tmp_path / "reviewed/source_audit.json",
        )
        assert result["frame_counts"] == {"train": 1, "valid": 0, "test": 0}
        assert result["failed_leaves"] == 0 and result["source_admission"]
    mace = read(str(tmp_path / "mace/train.extxyz"))
    deepmd = tmp_path / "deepmd/train/run/set.000"
    expected_virial = -reference.get_volume() * reference.get_stress(voigt=False)
    assert mace.info["REF_energy"] == pytest.approx(reference.get_potential_energy())
    np.testing.assert_allclose(mace.arrays["REF_forces"], reference.get_forces(apply_constraint=False))
    np.testing.assert_allclose(mace.info["REF_virial"].reshape(3, 3), expected_virial)
    np.testing.assert_allclose(np.load(deepmd / "energy.npy"), [[reference.get_potential_energy()]])
    np.testing.assert_allclose(np.load(deepmd / "force.npy").reshape(18, 3), mace.arrays["REF_forces"])
    np.testing.assert_allclose(np.load(deepmd / "virial.npy").reshape(3, 3), expected_virial)


def test_unbalanced_a2_does_not_disable_cross_format_membership_checks(tmp_path):
    import csv

    from interfaceforge.leaf_audit import audit_leaf_manifests

    rows = []
    for split in ("train", "valid", "test"):
        rows.extend(
            [
                {
                    "relative_leaf": "bulk/md",
                    "split": split,
                    "frames": 10,
                    "sampled_frames": 30,
                    "status": "OK",
                    "frame_digest": "md",
                },
                {
                    "relative_leaf": "a2/train/frame",
                    "split": split,
                    "frames": 1,
                    "sampled_frames": 1,
                    "status": "OK",
                    "frame_digest": "a2",
                },
            ]
        )
    for name in ("mace.csv", "deepmd.csv"):
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = audit_leaf_manifests(tmp_path / "mace.csv", tmp_path / "deepmd.csv", require_balanced_frames=False)
    assert report["status"] == "OK"
    rows[0]["frame_digest"] = "wrong"
    with (tmp_path / "deepmd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = audit_leaf_manifests(tmp_path / "mace.csv", tmp_path / "deepmd.csv", require_balanced_frames=False)
    assert report["status"] == "FAILED"
