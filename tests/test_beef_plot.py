from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from interfaceforge.beef import discover_beef_series, parse_beef_series, plot_beef_campaign
from interfaceforge.cli import build_parser

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def write_run(path: Path, *, scale: float = 1.0) -> None:
    path.mkdir(parents=True)
    (path / "INCAR").write_text("ML_MODE=train\nPOTIM=0.5\n", encoding="utf-8")
    (path / "ML_LOGFILE").write_text(
        "# synthetic fixture\n"
        "STATUS 1 accurate 1 F T 1 1\n"
        "BEEF 1 0.0 0.010 0.0 0.020\n"
        "STATUS 2 learning 3 T T 0 0\n"
        f"BEEF 2 0.0 {0.030 * scale:.6f} 0.0 0.020\n"
        "STATUS 3 critical 4 T T 0 0\n"
        f"BEEF 3 0.0 {0.050 * scale:.6f} 0.0 0.025\n",
        encoding="utf-8",
    )


class BeefPlotTests(unittest.TestCase):
    def test_parser_recovers_time_threshold_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            write_run(run)

            series = parse_beef_series(run)

            self.assertEqual(len(series.points), 3)
            self.assertEqual(series.points[-1].time_fs, 1.5)
            self.assertEqual(series.points[-1].ml_ctifor_ev_a, 0.025)
            self.assertEqual(series.points[-1].state, "critical")

    def test_campaign_discovery_excludes_archives_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active"
            archived = root / "active" / "expand_archive_20260728"
            write_run(active)
            write_run(archived, scale=2.0)

            self.assertEqual([item.run for item in discover_beef_series(root)], [active])
            self.assertEqual(len(discover_beef_series(root, include_archives=True)), 2)

    def test_cli_parses_campaign_plot_options(self) -> None:
        args = build_parser().parse_args(
            ["vasp", "beef-plot", "runs", "--individual", "--include-archives", "--dpi", "200"]
        )
        self.assertEqual(args.root, "runs")
        self.assertTrue(args.individual)
        self.assertTrue(args.include_archives)
        self.assertEqual(args.dpi, 200)

    @unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib is required for plot rendering")
    def test_campaign_plot_writes_png_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root / "run_a")
            write_run(root / "run_b", scale=1.5)

            result = plot_beef_campaign(root, individual=True)

            self.assertEqual(result["run_count"], 2)
            self.assertEqual(result["point_count"], 6)
            self.assertTrue(Path(result["plot"]).is_file())
            self.assertTrue(Path(result["data"]).is_file())
            self.assertEqual(len(result["individual_plots"]), 2)
            with Path(result["data"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["status_state"], "critical")


if __name__ == "__main__":
    unittest.main()
