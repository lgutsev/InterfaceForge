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
    discover_leaf_outcars,
)


class LeafDiscoveryTests(unittest.TestCase):
    def test_only_terminal_directories_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "system" / "400K"
            parent.mkdir(parents=True)
            (parent / "OUTCAR").write_text("parent", encoding="utf-8")
            leaf_a = parent / "run_01"
            leaf_b = parent / "run_02"
            leaf_a.mkdir()
            leaf_b.mkdir()
            (leaf_a / "OUTCAR").write_text("a", encoding="utf-8")
            (leaf_b / "OUTCAR").write_text("b", encoding="utf-8")

            sources = discover_leaf_outcars(root, heritage_depth=2)

            self.assertEqual({source.leaf.name for source in sources}, {"run_01", "run_02"})
            self.assertTrue(all(source.heritage_parts == ("system", "400K") for source in sources))

    def test_depth_uses_immediate_ancestors_not_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = root / "chemistry" / "termination" / "450K" / "replica_03"
            leaf.mkdir(parents=True)
            (leaf / "OUTCAR").write_text("", encoding="utf-8")

            [source] = discover_leaf_outcars(root, heritage_depth=2)

            self.assertEqual(source.heritage_parts, ("termination", "450K"))
            self.assertNotIn("replica_03", source.heritage_key)


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

            seen: dict[str, str] = {}
            for source in sources:
                split = assignment[source.outcar]
                previous = seen.setdefault(source.heritage_key, split)
                self.assertEqual(previous, split)
            self.assertEqual(set(seen.values()), {"train", "valid", "test"})


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
    def test_deepmd_system_physically_preserves_heritage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = root / "chemistry" / "450K" / "replica_01"
            leaf.mkdir(parents=True)
            outcar = leaf / "OUTCAR"
            outcar.write_text("", encoding="utf-8")
            source = LeafSource(
                outcar=outcar,
                leaf=leaf,
                relative_leaf=Path("chemistry/450K/replica_01"),
                run_id="chemistry__450K__replica_01",
                heritage_parts=("chemistry", "450K"),
                heritage_key="chemistry__450K",
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
                root / "dataset" / "train" / "chemistry" / "450K" / "replica_01",
            )
            self.assertTrue((system / "set.000" / "coord.npy").is_file())
            context = json.loads((system / "heritage.json").read_text(encoding="utf-8"))
            self.assertEqual(context["heritage_parts"], ["chemistry", "450K"])
            self.assertEqual(
                context["source_root_relative_leaf"], "chemistry/450K/replica_01"
            )


if __name__ == "__main__":
    unittest.main()
