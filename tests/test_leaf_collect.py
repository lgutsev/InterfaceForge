from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interfaceforge.leaf_collect import (
    LeafFrame,
    LeafSource,
    _write_deepmd_system,
    assign_heritage_groups,
    assign_random_frame_splits,
    balance_leaf_frames,
    discover_leaf_outcars,
)


class LeafDiscoveryTests(unittest.TestCase):
    def test_only_terminal_directories_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "material" / "termination" / "400K"
            parent.mkdir(parents=True)
            (parent / "OUTCAR").write_text("parent", encoding="utf-8")
            for replica in ("run_01", "run_02"):
                leaf = parent / replica
                leaf.mkdir()
                (leaf / "OUTCAR").write_text("", encoding="utf-8")

            sources = discover_leaf_outcars(root, heritage_depth=2)

            self.assertEqual({source.leaf.name for source in sources}, {"run_01", "run_02"})
            self.assertTrue(
                all(source.heritage_parts == ("termination", "400K") for source in sources)
            )
            self.assertTrue(
                all(source.heritage_parent == "material/termination/400K" for source in sources)
            )

    def test_same_folder_labels_in_different_branches_do_not_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for chemistry in ("A", "B"):
                leaf = root / chemistry / "termination" / "400K" / "run_01"
                leaf.mkdir(parents=True)
                (leaf / "OUTCAR").write_text("", encoding="utf-8")

            sources = discover_leaf_outcars(root, heritage_depth=2)

            self.assertEqual(len({source.heritage_key for source in sources}), 2)
            self.assertEqual(
                {source.heritage_parent for source in sources},
                {"A/termination/400K", "B/termination/400K"},
            )

    def test_backup_and_x_branches_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for branch in ("good", "backup_old", "Xdisabled"):
                leaf = root / branch / "run_01"
                leaf.mkdir(parents=True)
                (leaf / "OUTCAR").write_text("", encoding="utf-8")

            sources = discover_leaf_outcars(root)

            self.assertEqual([source.relative_leaf.as_posix() for source in sources], ["good/run_01"])


class HeritageSplitTests(unittest.TestCase):
    def test_sibling_leaves_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for system in ("A", "B", "C", "D", "E"):
                for replica in ("r1", "r2"):
                    leaf = root / system / "400K" / replica
                    leaf.mkdir(parents=True)
                    (leaf / "OUTCAR").write_text("", encoding="utf-8")

            sources = discover_leaf_outcars(root, heritage_depth=2)
            assignment = assign_heritage_groups(sources, (0.6, 0.2, 0.2), seed=9)

            seen: dict[str, set[str]] = {}
            for source in sources:
                seen.setdefault(source.heritage_key, set()).add(assignment[source.outcar])
            self.assertTrue(all(len(splits) == 1 for splits in seen.values()))
            self.assertEqual(set(assignment.values()), {"train", "valid", "test"})



class RandomFrameSplitTests(unittest.TestCase):
    def test_every_leaf_is_deterministically_stratified(self) -> None:
        frames = [
            LeafFrame(
                source_index=index,
                atoms=None,
                energy=float(index),
                forces=np.empty((0, 3)),
                move_mask=np.empty(0, dtype=np.int8),
                virial=None,
            )
            for index in range(10)
        ]

        first = assign_random_frame_splits(
            frames, (0.8, 0.1, 0.1), seed=17, leaf_key="interface/A"
        )
        second = assign_random_frame_splits(
            frames, (0.8, 0.1, 0.1), seed=17, leaf_key="interface/A"
        )

        self.assertEqual(
            {name: len(items) for name, items in first.items()},
            {"train": 8, "valid": 1, "test": 1},
        )
        self.assertEqual(
            {
                name: [frame.source_index for frame in items]
                for name, items in first.items()
            },
            {
                name: [frame.source_index for frame in items]
                for name, items in second.items()
            },
        )
        self.assertEqual(
            sorted(
                frame.source_index
                for items in first.values()
                for frame in items
            ),
            list(range(10)),
        )

    def test_balancing_deterministically_subsamples_longer_leaves(self) -> None:
        frames = [
            LeafFrame(
                source_index=index,
                atoms=None,
                energy=float(index),
                forces=np.empty((0, 3)),
                move_mask=np.empty(0, dtype=np.int8),
                virial=None,
            )
            for index in range(20)
        ]
        first = balance_leaf_frames(frames, 8, seed=17, leaf_key="interface/A")
        second = balance_leaf_frames(frames, 8, seed=17, leaf_key="interface/A")

        self.assertEqual(len(first), 8)
        self.assertEqual(
            [frame.source_index for frame in first],
            [frame.source_index for frame in second],
        )
        self.assertNotEqual([frame.source_index for frame in first], list(range(8)))


class _Cell:
    def __init__(self) -> None:
        self.array = np.eye(3) * 5.0


class _Atoms:
    def __init__(self) -> None:
        self.positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.cell = _Cell()

    def __len__(self) -> int:
        return 2

    def get_chemical_symbols(self) -> list[str]:
        return ["H", "H"]


class DeepMDContextTests(unittest.TestCase):
    def test_deepmd_system_physically_preserves_full_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = root / "chemistry" / "termination" / "450K" / "replica_01"
            leaf.mkdir(parents=True)
            outcar = leaf / "OUTCAR"
            outcar.write_text("", encoding="utf-8")
            source = LeafSource(
                outcar=outcar,
                leaf=leaf,
                relative_leaf=Path("chemistry/termination/450K/replica_01"),
                run_id="chemistry__termination__450K__replica_01",
                heritage_parts=("termination", "450K"),
                heritage_parent="chemistry/termination/450K",
                heritage_key="chemistry_termination_450K",
            )
            frame = LeafFrame(
                source_index=7,
                atoms=_Atoms(),
                energy=-2.0,
                forces=np.zeros((2, 3)),
                move_mask=np.ones(2, dtype=np.int8),
                virial=None,
            )

            system = _write_deepmd_system(
                root / "dataset" / "train", source, "train", [frame], ["H"]
            )

            self.assertEqual(
                system,
                root
                / "dataset"
                / "train"
                / "chemistry"
                / "termination"
                / "450K"
                / "replica_01",
            )
            self.assertTrue((system / "set.000" / "coord.npy").is_file())
            context = json.loads((system / "heritage.json").read_text(encoding="utf-8"))
            self.assertEqual(context["heritage_parent"], "chemistry/termination/450K")
            self.assertEqual(context["heritage_context"], ["termination", "450K"])
            self.assertEqual(
                context["source_root_relative_leaf"],
                "chemistry/termination/450K/replica_01",
            )


if __name__ == "__main__":
    unittest.main()
