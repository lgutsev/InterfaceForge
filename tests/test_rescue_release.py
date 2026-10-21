from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read, write

from interfaceforge.errors import SafetyError
from interfaceforge.rescue_release import SOURCE_STRUCTURE, assemble, freeze, load_policy, plan, source_rule
from interfaceforge.vasp_provenance import sha256_file

RUNS = {"bulk_A": "bulk/A", "bulk_B": "bulk/B", "bulk_X": "bulk/X"}


def _frame(run: str, index: int, split: str, *, energy: float | None = None, shift: float = 0.0) -> Atoms:
    rng = np.random.default_rng(1000 * len(run) + index)
    atoms = Atoms("TiN", positions=[[0.1, 0.2, 0.3], [2.1 + 0.01 * index, 2.0, 2.0]], cell=np.eye(3) * 4.2, pbc=True)
    atoms.positions[0, 0] += shift
    atoms.info.update(
        REF_energy=-10.0 - 0.01 * index if energy is None else energy,
        REF_virial=rng.normal(size=9),
        source_run=run,
        source_frame=index,
        split=split,
        IF_leaf=RUNS[run],
        source_path=f"/remote/{run}/OUTCAR",
    )
    atoms.arrays["REF_forces"] = rng.normal(size=(2, 3))
    atoms.arrays["move_mask"] = np.ones(2, dtype=int)
    return atoms


def deck(tmp_path: Path) -> Path:
    root = tmp_path / "deck"
    root.mkdir()
    splits = {"train": [0, 1, 2, 3], "valid": [4], "test": [5]}
    for split, indices in splits.items():
        frames = [_frame(run, i, split) for run in RUNS for i in indices]
        write(root / f"{split}.extxyz", frames, format="extxyz")
    return root


def relabel(path: Path, frames: list[Atoms]) -> str:
    write(path, frames, format="extxyz")
    return sha256_file(path)


def policy(tmp_path: Path, replacement: Path | None, *, b_action: str = "replace_subset", **extra) -> Path:
    sources = {
        "bulk_A": {"action": "keep"},
        "bulk_B": {"action": b_action, "unreplaced": "hold", "reason": "Gamma-only labels"},
        "bulk_X": {"action": "exclude", "reason": "sampling quarantine"},
    }
    if b_action != "replace_subset":
        sources["bulk_B"].pop("unreplaced")
    data = {"schema_version": 1, "sources": sources, "replacements": []}
    if replacement is not None:
        data["replacements"].append({"id": "relabel_b", "path": replacement.name, "sha256": sha256_file(replacement)})
    data.update(extra)
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def frozen(tmp_path: Path) -> tuple[Path, Path]:
    root = deck(tmp_path)
    report = freeze(root, tmp_path / "freeze")
    assert report["errors"] == []
    return root, tmp_path / "freeze" / "split_ledger.tsv"


def test_freeze_records_every_frame_and_split(tmp_path):
    root = deck(tmp_path)
    report = freeze(root, tmp_path / "freeze")
    assert report["frames"] == 18
    assert report["split_counts"] == {"train": 12, "valid": 3, "test": 3}
    assert report["per_run"]["bulk_B"] == {"train": 4, "valid": 1, "test": 1}
    assert report["geometries_in_more_than_one_split"] == 0


def test_freeze_flags_a_frame_listed_in_two_splits(tmp_path):
    root = deck(tmp_path)
    frames = read(root / "valid.extxyz", index=":")
    frames.append(_frame("bulk_A", 0, "valid"))
    write(root / "valid.extxyz", frames, format="extxyz")
    report = freeze(root, tmp_path / "freeze")
    assert any("bulk_A frame 0 in both train and valid" in e for e in report["errors"])


def test_partial_replacement_keeps_historical_splits_and_replaces_all_channels(tmp_path):
    root, ledger = frozen(tmp_path)
    new = []
    for index, split in ((1, "train"), (4, "valid")):
        atoms = _frame("bulk_B", index, split, energy=-99.0 - index)
        atoms.arrays["REF_forces"] = np.full((2, 3), 0.5)
        atoms.info["REF_virial"] = np.arange(9.0)
        atoms.info["split"] = "test"  # a relabel's own split field is ignored: the ledger decides
        new.append(atoms)
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, new)
    report = assemble(ledger, policy(tmp_path, replacement), root, tmp_path / "release")
    assert report["status_counts"] == {"kept": 6, "replaced": 2, "held": 4, "excluded": 6}
    assert report["split_counts"] == {"train": 5, "valid": 2, "test": 1}
    assert report["replaced_by_split"] == {"train": 1, "valid": 1}
    valid = read(tmp_path / "release" / "valid.extxyz", index=":")
    b = next(a for a in valid if a.info["source_run"] == "bulk_B")
    assert b.info["REF_energy"] == pytest.approx(-103.0)
    assert np.allclose(b.arrays["REF_forces"], 0.5)
    assert np.allclose(b.info["REF_virial"], np.arange(9.0))
    assert b.info["label_source"] == "relabel_b"
    virial = np.load(tmp_path / "release" / "deepmd" / "valid" / "all" / "bulk" / "B" / "set.000" / "virial.npy")
    assert np.allclose(virial[0], np.arange(9.0))
    assert not (tmp_path / "release" / "deepmd" / "train" / "all" / "bulk" / "X").exists()


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: [_frame("bulk_B", 1, "train"), _frame("bulk_B", 1, "train")], "Duplicate replacement"),
        (lambda: [_frame("bulk_A", 1, "train")], "whose rule 'bulk_A' is keep"),
        (lambda: [_frame("bulk_X", 1, "train")], "whose rule 'bulk_X' is exclude"),
        (lambda: [_frame("bulk_B", 99, "train")], "absent from the frozen ledger"),
        (lambda: [_frame("bulk_B", 1, "train", shift=0.01)], "positions differ"),
    ],
)
def test_wrong_replacements_fail(tmp_path, make, message):
    root, ledger = frozen(tmp_path)
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, make())
    with pytest.raises(SafetyError, match=message):
        assemble(ledger, policy(tmp_path, replacement), root, tmp_path / "release")


def test_replace_all_requires_every_frame(tmp_path):
    root, ledger = frozen(tmp_path)
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, [_frame("bulk_B", i, "train") for i in range(5)])
    with pytest.raises(SafetyError, match="replace_all sources lack replacements: {'bulk_B': 1}"):
        assemble(ledger, policy(tmp_path, replacement, b_action="replace_all"), root, tmp_path / "release")


def test_unlisted_source_and_changed_replacement_fail(tmp_path):
    root, ledger = frozen(tmp_path)
    path = policy(tmp_path, None)
    data = yaml.safe_load(path.read_text())
    del data["sources"]["bulk_A"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SafetyError, match="No release-policy rule for source bulk_A"):
        assemble(ledger, path, root, tmp_path / "release")
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, [_frame("bulk_B", 1, "train")])
    path = policy(tmp_path, replacement)
    write(replacement, [_frame("bulk_B", 2, "train")], format="extxyz")
    with pytest.raises(SafetyError, match="Replacement relabel_b changed"):
        assemble(ledger, path, root, tmp_path / "release2")


def test_deck_changed_after_freeze_fails(tmp_path):
    root, ledger = frozen(tmp_path)
    frames = read(root / "train.extxyz", index=":")
    frames[0].info["REF_energy"] += 1e-3
    write(root / "train.extxyz", frames, format="extxyz")
    with pytest.raises(SafetyError, match="changed since the ledger was frozen"):
        assemble(ledger, policy(tmp_path, None), root, tmp_path / "release")


def test_release_directory_is_never_overwritten(tmp_path):
    root, ledger = frozen(tmp_path)
    out = tmp_path / "release"
    out.mkdir()
    (out / "old").write_text("x")
    with pytest.raises(SafetyError, match="never overwritten"):
        assemble(ledger, policy(tmp_path, None), root, out)


def test_fresh_recollection_is_matched_by_source_frame_not_position(tmp_path):
    root, ledger = frozen(tmp_path)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for split in ("train", "valid", "test"):
        frames = [a for a in read(root / f"{split}.extxyz", index=":") if a.info["source_run"] != "bulk_X"]
        write(fresh / f"{split}.extxyz", frames[::-1], format="extxyz")  # excluded source absent, order changed
    report = assemble(ledger, policy(tmp_path, None, b_action="keep"), fresh, tmp_path / "release")
    assert report["status_counts"] == {"kept": 12, "excluded": 6}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda frames: frames[1:], "active ledger frames are missing"),
        (lambda frames: frames + [_frame("bulk_A", 4, "train")], "is in train; the frozen ledger says valid"),
        (lambda frames: frames + [_frame("bulk_A", 77, "train")], "is not in the frozen ledger"),
    ],
)
def test_recollection_membership_must_equal_the_ledger(tmp_path, edit, message):
    root, ledger = frozen(tmp_path)
    frames = read(root / "train.extxyz", index=":")
    write(root / "train.extxyz", edit(frames), format="extxyz")
    with pytest.raises(SafetyError, match=message):
        assemble(ledger, policy(tmp_path, None, b_action="keep"), root, tmp_path / "release")


# --------------------------------------------------------------------------- label masks, added frames, leakage


def _edit(path: Path, change) -> Path:
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _strain(index: int, role: str, *, shift: float = 0.5) -> Atoms:
    atoms = _frame("bulk_A", index, "train", shift=shift + 0.01 * index)
    for key in ("source_run", "source_frame", "split", "IF_leaf"):
        del atoms.info[key]
    atoms.info.update(case_id=f"strain_{index}", role=role)
    return atoms


def test_label_masks_leave_the_term_out_of_extxyz_and_deepmd(tmp_path):
    root, ledger = frozen(tmp_path)
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, [_frame("bulk_B", 1, "train", energy=-50.0)])

    def masks(data):
        data["sources"]["bulk_A"]["label_use"] = "energy_forces"
        data["sources"]["bulk_B"]["label_use"] = "forces_virial"

    report = assemble(ledger, _edit(policy(tmp_path, replacement), masks), root, tmp_path / "release")
    assert report["label_use_by_split"]["train"] == {"energy_forces": 4, "forces_virial": 1}
    deepmd = tmp_path / "release" / "deepmd"
    a = deepmd / "train" / "energy_forces" / "bulk" / "A" / "set.000"
    b = deepmd / "train" / "forces_virial" / "bulk" / "B" / "set.000"
    assert (a / "energy.npy").is_file() and (a / "force.npy").is_file() and not (a / "virial.npy").exists()
    assert (b / "virial.npy").is_file() and (b / "force.npy").is_file() and not (b / "energy.npy").exists()
    assert not (deepmd / "train" / "all").exists()
    meta = json.loads((b.parent / "release_labels.json").read_text())
    assert meta["absent_files"] == ["energy.npy"] and meta["trained_terms"] == ["forces", "virial"]
    systems = json.loads((deepmd / "systems.json").read_text())
    assert ("train", "forces_virial", 1) in {(s["split"], s["label_use"], s["frames"]) for s in systems}
    train = read(tmp_path / "release" / "train.extxyz", index=":")
    a_frame = next(x for x in train if x.info["source_run"] == "bulk_A")
    b_frame = next(x for x in train if x.info["source_run"] == "bulk_B")
    assert "REF_virial" not in a_frame.info and a_frame.info["label_use"] == "energy_forces"
    assert "REF_energy" not in b_frame.info and b_frame.info["label_use"] == "forces_virial"
    header = (tmp_path / "release" / "release_ledger.tsv").read_text().splitlines()[0].split("\t")
    assert "label_use" in header


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["sources"]["bulk_A"].update(label_use="virial_only"), "label_use must be one of"),
        (lambda d: d["replacements"][0].update(label_use="energy_forces"), "label_use belongs to the source rule"),
        (lambda d: d["replacements"][0].update(manifest="m.json"), "needs its manifest_sha256"),
        (
            lambda d: d.update(added_frames=[{"id": "s", "path": "x", "sha256": "0", "split": "train"}]),
            "needs id, path, sha256, split, label_use and reason",
        ),
        (
            lambda d: d.update(
                added_frames=[
                    {"id": "s", "path": "x", "sha256": "0", "split": "train2", "label_use": "all", "reason": "r"}
                ]
            ),
            "split must be one of",
        ),
        (
            lambda d: d.update(leakage_references=[{"id": "r", "path": "x", "sha256": "0", "forbid": ["holdout"]}]),
            "forbid must list splits",
        ),
        (
            lambda d: d.update(
                leakage_references=[{"id": "relabel_b", "path": "x", "sha256": "0", "forbid": ["train"]}]
            ),
            "ids must be unique",
        ),
    ],
)
def test_policy_rejects_bad_masks_and_sections(tmp_path, change, message):
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, [_frame("bulk_B", 1, "train")])
    with pytest.raises(SafetyError, match=message):
        load_policy(_edit(policy(tmp_path, replacement), change))


def _strain_policy(tmp_path: Path, frames: list[Atoms], *, holdouts: bool = True, **extra) -> Path:
    path = tmp_path / "strain.extxyz"
    sha = relabel(path, frames)
    common = {"path": path.name, "sha256": sha, "label_use": "forces_virial"}
    entries = [
        {
            "id": "strain_train",
            **common,
            "select": {"role": "train"},
            "split": "train",
            "reason": "decision 9",
            "leaf": "strain/demo",
        },
    ]
    if holdouts:
        entries.append(
            {
                "id": "strain_holdout",
                **common,
                "select": {"role": "holdout"},
                "split": "holdout",
                "reason": "never trained",
            }
        )
    return policy(tmp_path, None, b_action="keep", added_frames=entries, **extra)


def test_added_frames_train_and_holdouts_never_reach_deepmd(tmp_path):
    root, ledger = frozen(tmp_path)
    frames = [_strain(0, "train"), _strain(1, "train"), _strain(2, "holdout")]
    report = assemble(ledger, _strain_policy(tmp_path, frames), root, tmp_path / "release")
    assert report["added_by_split"] == {"train": 2, "holdout": 1}
    assert report["split_counts"]["train"] == 10
    system = tmp_path / "release" / "deepmd" / "train" / "forces_virial" / "strain" / "demo" / "NTi_2"
    assert (system / "set.000" / "virial.npy").is_file() and not (system / "set.000" / "energy.npy").exists()
    assert np.load(system / "set.000" / "force.npy").shape == (2, 6)
    assert not list((tmp_path / "release" / "deepmd").glob("holdout*"))
    holdout = read(tmp_path / "release" / "holdout.extxyz", index=":")
    assert [a.info["case_id"] for a in holdout] == ["strain_2"]
    assert "REF_energy" in holdout[0].info and "REF_virial" in holdout[0].info  # holdouts keep every label
    train = read(tmp_path / "release" / "train.extxyz", index=":")
    added = [a for a in train if str(a.info["source_run"]).startswith("added:")]
    assert len(added) == 2 and all("REF_energy" not in a.info for a in added)
    for split in ("valid", "test"):
        assert all(a.info["split"] != "holdout" for a in read(tmp_path / "release" / f"{split}.extxyz", index=":"))


def test_holdout_geometry_in_any_release_split_fails(tmp_path):
    root, ledger = frozen(tmp_path)
    leaked = _frame("bulk_A", 5, "test")  # a deck test frame listed as a holdout
    leaked.info.update(case_id="leak", role="holdout")
    with pytest.raises(SafetyError, match=r"strain_holdout\[1\] leak is bulk_A frame 5 in test"):
        assemble(ledger, _strain_policy(tmp_path, [_strain(0, "train"), leaked]), root, tmp_path / "release")


def test_holdout_matched_within_tolerance_not_only_by_digest(tmp_path):
    root, ledger = frozen(tmp_path)
    near = _frame("bulk_A", 2, "train")
    near.positions[1, 2] += 4e-4  # different digest, same structure
    near.info.update(case_id="near", role="holdout")
    path = _strain_policy(tmp_path, [_strain(0, "train"), near])
    plan(ledger, path)  # digest level only: the ledger holds no positions
    with pytest.raises(SafetyError, match="near is bulk_A frame 2 in train"):
        plan(ledger, path, root)
    with pytest.raises(SafetyError, match="near is bulk_A frame 2 in train"):
        assemble(ledger, path, root, tmp_path / "release")


def test_leakage_reference_respects_its_forbidden_splits(tmp_path):
    root, ledger = frozen(tmp_path)
    ref = tmp_path / "refs.extxyz"
    sha = relabel(ref, [_frame("bulk_A", 5, "test")])
    allowed = [{"id": "ti_term_test", "path": ref.name, "sha256": sha, "forbid": ["train", "valid"]}]
    report = assemble(
        ledger, policy(tmp_path, None, b_action="keep", leakage_references=allowed), root, tmp_path / "release"
    )
    assert report["leakage_references"] == {"ti_term_test": 1}
    forbidden = [{"id": "pkg14", "path": ref.name, "sha256": sha, "forbid": ["train", "valid", "test"]}]
    with pytest.raises(SafetyError, match=r"pkg14\[0\]  is bulk_A frame 5 in test"):
        plan(ledger, policy(tmp_path, None, b_action="keep", leakage_references=forbidden))


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (lambda: [_strain(0, "train"), _strain(0, "train")], r"repeats a \['train'\] frame"),
        (lambda: [_frame("bulk_A", 1, "train")], r"repeats a \['train'\] frame"),
    ],
)
def test_added_frames_must_be_new_geometries(tmp_path, frames, message):
    root, ledger = frozen(tmp_path)
    made = frames()
    for atoms in made:
        atoms.info["role"] = "train"
    with pytest.raises(SafetyError, match=message):
        plan(ledger, _strain_policy(tmp_path, made, holdouts=False))


def test_added_frames_count_overlap_and_hash_are_checked(tmp_path):
    root, ledger = frozen(tmp_path)
    frames = [_strain(0, "train"), _strain(1, "holdout")]
    path = _strain_policy(tmp_path, frames)
    with pytest.raises(SafetyError, match="1 frames selected, the policy expects 2"):
        plan(ledger, _edit(path, lambda d: d["added_frames"][0].update(expected_frames=2)))
    path = _strain_policy(tmp_path, frames)
    with pytest.raises(SafetyError, match="is selected by strain_train and strain_holdout"):
        plan(ledger, _edit(path, lambda d: d["added_frames"][1].pop("select")))
    path = _strain_policy(tmp_path, frames)
    write(tmp_path / "strain.extxyz", frames[:1], format="extxyz")
    with pytest.raises(SafetyError, match="Added frames strain_train changed"):
        plan(ledger, path)


def _collector(tmp_path: Path, keys: list[tuple[int, str]], *, manifest_split: str | None = None) -> Path:
    """Collector-style relabels: no source_run, source_frame is the ionic step, identity via the manifest."""
    frames, manifest = [], {"frames": []}
    for index, split in keys:
        atoms = _frame("bulk_B", index, split, energy=-70.0 - index)
        for key in ("source_run", "split", "IF_leaf"):
            del atoms.info[key]
        atoms.info.update(source_frame=0, case_id=f"r300_b_f{index:04d}", role=split)
        frames.append(atoms)
        manifest["frames"].append(
            {
                "case_id": atoms.info["case_id"],
                "source_structure": f"bulk_B source_frame {index} (split {manifest_split or split})",
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    sha_manifest = sha256_file(tmp_path / "manifest.json")
    for atoms in frames:
        atoms.info["manifest_sha256"] = sha_manifest
    replacement = tmp_path / "relabel_b.extxyz"
    relabel(replacement, frames)
    return _edit(
        policy(tmp_path, replacement),
        lambda d: d["replacements"][0].update(manifest="manifest.json", manifest_sha256=sha_manifest),
    )


def test_collector_relabels_map_to_deck_frames_through_the_manifest(tmp_path):
    root, ledger = frozen(tmp_path)
    path = _collector(tmp_path, [(1, "train"), (4, "valid")])
    assert plan(ledger, path)["replaced_by_source"] == {"bulk_B": 2}
    assemble(ledger, path, root, tmp_path / "release")
    valid = read(tmp_path / "release" / "valid.extxyz", index=":")
    assert next(a for a in valid if a.info["source_run"] == "bulk_B").info["REF_energy"] == pytest.approx(-74.0)


def test_manifest_split_must_match_the_frozen_ledger(tmp_path):
    root, ledger = frozen(tmp_path)
    with pytest.raises(SafetyError, match="selected as test, the frozen ledger says train"):
        plan(ledger, _collector(tmp_path, [(1, "train")], manifest_split="test"))


def test_changed_manifest_fails(tmp_path):
    root, ledger = frozen(tmp_path)
    path = _collector(tmp_path, [(1, "train")])
    (tmp_path / "manifest.json").write_text((tmp_path / "manifest.json").read_text() + "\n")
    with pytest.raises(SafetyError, match="manifest changed"):
        plan(ledger, path)


def test_manifest_regex_reads_both_collector_formats():
    deck_style = "D:\\Traj, source_run interface_300K_Ideal_Ti_Term_X_O_x0.75, source_frame 2060 (split test)"
    assert SOURCE_STRUCTURE.search(deck_style).groups() == ("interface_300K_Ideal_Ti_Term_X_O_x0.75", "2060", "test")
    assert SOURCE_STRUCTURE.search("bulk_SiN-Bulk_300K source_frame 5 (split train)").groups() == (
        "bulk_SiN-Bulk_300K",
        "5",
        "train",
    )


def test_plan_writes_nothing_and_counts_label_use(tmp_path):
    root, ledger = frozen(tmp_path)
    path = _strain_policy(tmp_path, [_strain(0, "train"), _strain(1, "holdout")])
    before = sorted(p.name for p in tmp_path.iterdir())
    summary = plan(ledger, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert summary["status_counts"] == {"kept": 12, "excluded": 6, "added": 1, "holdout": 1}
    assert summary["label_use_by_split"]["train"] == {"all": 8, "forces_virial": 1}
    assert summary["added_by_entry"] == {"added:strain_holdout": {"holdout": 1}, "added:strain_train": {"train": 1}}


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "source-rescue"


@pytest.mark.parametrize(
    ("run", "action", "label_use"),
    [
        ("bulk_SiN-Bulk_300K", "replace_subset", "all"),
        ("bulk_TiN-Bulk_300K", "replace_subset", "energy_forces"),
        ("bulk_TiN-Bulk_450K", "exclude", "all"),
        ("bulk_TiO-Bulk-Ideal_300K", "exclude", "all"),
        ("bulk_TiO-Bulk-Real_450K", "replace_subset", "all"),
        ("interface_300K_Ideal_Ti_Term_SiN-TiN-Ti-term", "keep", "energy_forces"),
        ("interface_450K_Ideal_Ti_Term_SiN-TiN-Ti-term_O_x0.25", "keep", "energy_forces"),
        ("interface_450K_Ideal_Ti_Term_SiN-TiN-Ti-term", "keep", "energy_forces"),  # HOLD lifted 2026-10-21
        ("interface_300K_Real_Ti_Term_SiN-TiN-Ti-term_O_x1.0", "keep", "all"),
        ("interface_450K_Ideal_N_Term_SiN_TiN_N-term_O_x0.5", "keep", "all"),
    ],
)
def test_shipped_policy_encodes_the_recorded_decisions(run, action, label_use):
    rule = source_rule(load_policy(EXAMPLES / "release_policy.yaml"), run)[1]
    assert (rule["action"], rule.get("label_use", "all")) == (action, label_use)


def test_shipped_policy_lists_the_strain_frames_and_holdouts():
    shipped = load_policy(EXAMPLES / "release_policy.yaml")
    added = {a["id"]: (a["split"], a["label_use"], a["expected_frames"]) for a in shipped["added_frames"]}
    assert added == {
        "pkg15_strain": ("train", "forces_virial", 24),
        "SD006_strain_train": ("train", "forces_virial", 20),
        "SD006_strain_holdout": ("holdout", "forces_virial", 16),
    }
    assert sum(r["expected_frames"] for r in shipped["replacements"]) == 300
    assert {r["id"]: r["forbid"] for r in shipped["leakage_references"]} == {
        "pkg14_references": ["train", "valid", "test"],
        "SD015_ideal_ti_term_test": ["train", "valid"],
    }
    load_policy(EXAMPLES / "release_policy_cleanup_diagnostic.yaml")
