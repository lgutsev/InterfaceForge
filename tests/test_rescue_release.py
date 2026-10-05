from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read, write

from interfaceforge.errors import SafetyError
from interfaceforge.rescue_release import assemble, freeze
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
    virial = np.load(tmp_path / "release" / "deepmd" / "valid" / "bulk" / "B" / "set.000" / "virial.npy")
    assert np.allclose(virial[0], np.arange(9.0))
    assert not (tmp_path / "release" / "deepmd" / "train" / "bulk" / "X").exists()


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
