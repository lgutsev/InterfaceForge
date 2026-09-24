"""Contract tests for the neural_paw_dft worker script.

The real worker runs in a real subprocess against a *stub* ``neural_paw_dft``
(and minimal ``pymatgen``) that mirrors the upstream API used by the worker:
``Pipeline(cfg).predict(...)``, ``assemble.build_chgcar(...)``,
``mp_potcar_map.MP_POTCAR_BY_Z`` and ``models.resolve_weights``.  The stub
records every call so the tests can assert the worker never calls
``Pipeline.build`` (which would rewrite INCAR/POTCAR/KPOINTS), disables CHGNet
when the INCAR MAGMOM is authoritative, and swaps to the charge-only model.

``NeuralPawRealIntegrationTest`` additionally exercises the *real* package when
``IFACE_NDI_PYTHON`` points at an interpreter that has it (skipped otherwise).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from density_init_fixtures import NIO_INCAR, NIO_MAGMOM, write_run

from interfaceforge.density_init import InferenceError, NeuralPawInitializer, initialize_density
from interfaceforge.errors import DependencyError

STUB_FILES = {
    "pymatgen/__init__.py": "",
    "pymatgen/core.py": '''
        from types import SimpleNamespace
        _Z = {"H": 1, "O": 8, "Ti": 22, "Ni": 28, "Sr": 38}

        class Element:
            def __init__(self, symbol):
                self.symbol = symbol
                self.Z = _Z[symbol]

        class Structure:
            def __init__(self, species, text):
                self.sites = [SimpleNamespace(specie=Element(s)) for s in species]
                self.text = text
                self.volume = 36.0
            def __len__(self):
                return len(self.sites)
            def __iter__(self):
                return iter(self.sites)
            @classmethod
            def from_file(cls, path):
                text = open(path).read()
                lines = text.splitlines()
                symbols, counts = lines[5].split(), [int(c) for c in lines[6].split()]
                species = [s for s, c in zip(symbols, counts) for _ in range(c)]
                return cls(species, text)
    ''',
    "neural_paw_dft/__init__.py": '__version__ = "0.1.0-stub"\n',
    "neural_paw_dft/_log.py": '''
        import json, os
        def log(event, **data):
            with open(os.environ["NDI_STUB_LOG"], "a") as handle:
                handle.write(json.dumps({"event": event, **data}) + "\\n")
    ''',
    "neural_paw_dft/augnet/__init__.py": "",
    "neural_paw_dft/augnet/mp_potcar_map.py": '''
        MP_POTCAR_BY_Z = {
            8: {"element": "O", "variant": "O", "l_channels": (0, 0, 1, 1), "n": 33},
            22: {"element": "Ti", "variant": "Ti_pv", "l_channels": (1, 1, 2, 2, 0, 0), "n": 138},
            28: {"element": "Ni", "variant": "Ni_pv", "l_channels": (1, 1, 2, 2, 0, 0), "n": 138},
            38: {"element": "Sr", "variant": "Sr_sv", "l_channels": (0, 0, 1, 1, 2, 2), "n": 138},
        }
    ''',
    "neural_paw_dft/models.py": '''
        from pathlib import Path
        REGISTRY = {"electrafi_total": "t", "electrafi_spin_constrained": "s",
                    "augnet_total_full": "a", "augnet_spin_full": "m"}
        def resolve_weights(name, root=None):
            path = Path(root or ".") / (name + ".safetensors")
            path.write_text(name)
            return path
    ''',
    "neural_paw_dft/pipeline/__init__.py": '''
        from .config import PipelineConfig, load_config
        from .pipeline import Pipeline
    ''',
    "neural_paw_dft/pipeline/config.py": '''
        from dataclasses import dataclass, field

        @dataclass
        class ElectrafiConfig:
            checkpoint: str = "electrafi_spin_constrained"
            spin: bool = True

        @dataclass
        class AugnetConfig:
            total_checkpoint: str = "augnet_total_full"
            mag_checkpoint: str = "augnet_spin_full"

        @dataclass
        class ChgnetConfig:
            enabled: bool = True
            model: str = None

        @dataclass
        class PipelineConfig:
            device: str = "auto"
            grid_dims: tuple = None
            weights_dir: str = None
            electrafi: ElectrafiConfig = field(default_factory=ElectrafiConfig)
            augnet: AugnetConfig = field(default_factory=AugnetConfig)
            chgnet: ChgnetConfig = field(default_factory=ChgnetConfig)

        def load_config(path):
            return PipelineConfig()
    ''',
    "neural_paw_dft/pipeline/pipeline.py": '''
        import os
        from dataclasses import asdict
        from types import SimpleNamespace
        import numpy as np
        from .._log import log

        class Pipeline:
            def __init__(self, cfg):
                self.cfg = cfg
                self.device = "cpu"
                log("init", cfg=asdict(cfg))
                if os.environ.get("NDI_STUB_FAIL"):
                    raise RuntimeError(os.environ["NDI_STUB_FAIL"])
            def electrafi(self, n):
                log("electrafi", checkpoint=self.cfg.electrafi.checkpoint, spin=self.cfg.electrafi.spin)
            def augnet(self, channel):
                log("augnet", channel=channel)
            def chgnet(self):
                log("chgnet")
            def site_moments(self, structure):
                if not self.cfg.chgnet.enabled:
                    return None
                log("chgnet_moments")
                return np.array([0.7] * len(structure))
            def build(self, *args, **kwargs):
                log("build")
                raise AssertionError("worker must not call Pipeline.build")
            def predict(self, inp, grid_dims=None, nelect=None, lmaxmix=None, site_moments=None):
                if site_moments is None:
                    site_moments = self.site_moments(inp)
                log("predict", grid=list(grid_dims), nelect=nelect, lmaxmix=lmaxmix,
                    site_moments=None if site_moments is None else [float(x) for x in site_moments])
                spin = self.cfg.electrafi.spin
                return SimpleNamespace(
                    grid_dims=tuple(grid_dims), nelect=nelect, lmaxmix=lmaxmix,
                    rho_total=np.full(grid_dims, nelect / 36.0),
                    rho_spin=np.zeros(grid_dims) if spin else None,
                    aug_total={1: np.zeros(3)}, aug_diff={1: np.zeros(3)} if spin else None,
                    site_moments=site_moments,
                    m_total=None if site_moments is None else float(np.sum(site_moments)),
                )
    ''',
    "neural_paw_dft/pipeline/assemble.py": '''
        from .._log import log
        def build_chgcar(structure, rho_total, rho_spin, aug_total, aug_diff, site_moments, out_path, nelect=None):
            log("build_chgcar", spin=rho_spin is not None, nelect=nelect)
            grid = " ".join(f"{g:5d}" for g in rho_total.shape)
            with open(out_path, "w") as handle:
                handle.write(structure.text.rstrip("\\n") + "\\n\\n" + grid + "\\n 1.0\\n")
            return out_path
    ''',
}


def _write_stub(root: Path) -> None:
    for relative, body in STUB_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")


class WorkerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        stub = self.tmp / "stub"
        _write_stub(stub)
        self.log = self.tmp / "stub_calls.jsonl"
        env = {
            "PYTHONPATH": os.pathsep.join([str(stub), os.environ.get("PYTHONPATH", "")]),
            "NDI_STUB_LOG": str(self.log),
        }
        self._env = mock.patch.dict(os.environ, env)
        self._env.start()
        self.backend = NeuralPawInitializer(python=sys.executable, weights_dir=str(self.tmp))

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_probe_reports_stub_package(self) -> None:
        info = self.backend.probe()
        self.assertTrue(info["available"], info)
        self.assertEqual(info["version"], "0.1.0-stub")

    def test_afm_incar_worker_uses_charge_only_model_without_chgnet(self) -> None:
        run = write_run(self.tmp / "run")
        result = initialize_density(run, backend=self.backend, grid=(20, 20, 24))
        self.assertEqual(result["status"], "PROMOTED")
        events = self.events()
        names = [event["event"] for event in events]
        self.assertNotIn("build", names)
        self.assertNotIn("chgnet", names)
        self.assertNotIn("chgnet_moments", names)
        init = next(event for event in events if event["event"] == "init")
        self.assertEqual(init["cfg"]["electrafi"], {"checkpoint": "electrafi_total", "spin": False})
        self.assertFalse(init["cfg"]["chgnet"]["enabled"])
        self.assertIsNone(init["cfg"]["augnet"]["mag_checkpoint"])
        predict = next(event for event in events if event["event"] == "predict")
        self.assertEqual(predict["grid"], [20, 20, 24])
        self.assertEqual(predict["nelect"], 44.0)
        self.assertEqual(predict["lmaxmix"], 4)
        self.assertIsNone(predict["site_moments"])
        self.assertIn(f"MAGMOM = {NIO_MAGMOM}", (run / "INCAR").read_text())
        self.assertEqual(result["backend_info"]["models"]["electrafi"]["name"], "electrafi_total")
        self.assertEqual(len(result["backend_info"]["models"]["electrafi"]["sha256"]), 64)
        self.assertIn("model_load_s", result["timing"])
        self.assertIn("inference_s", result["timing"])

    def test_sign_uniform_incar_moments_are_passed_to_predict(self) -> None:
        run = write_run(self.tmp / "run", incar=NIO_INCAR.replace(NIO_MAGMOM, "2*1.7 2*0"))
        result = initialize_density(run, backend=self.backend, grid=(20, 20, 24))
        predict = next(event for event in self.events() if event["event"] == "predict")
        self.assertEqual(predict["site_moments"], [1.7, 1.7, 0.0, 0.0])
        self.assertTrue(result["magnetism"]["spin_channel_written"])
        self.assertNotIn("chgnet_moments", [event["event"] for event in self.events()])

    def test_projector_mismatch_potcar_is_refused(self) -> None:
        run = write_run(self.tmp / "run", potcar_symbols=("Ni", "O"))
        with self.assertRaisesRegex(InferenceError, "augmentation schema differs"):
            initialize_density(run, backend=self.backend, grid=(20, 20, 24), force=True)
        self.assertFalse((run / "CHGCAR").exists())

    def test_worker_exception_is_reported_and_rolled_back(self) -> None:
        run = write_run(self.tmp / "run")
        with mock.patch.dict(os.environ, {"NDI_STUB_FAIL": "pykeops compilation failed"}):
            with self.assertRaisesRegex(InferenceError, "pykeops compilation failed"):
                initialize_density(run, backend=self.backend, grid=(20, 20, 24))
        self.assertFalse((run / "CHGCAR").exists())
        self.assertIn("pykeops", (run / "density_init.log").read_text() + (run / "density_init.json").read_text())


class WorkerWithoutPackageTests(unittest.TestCase):
    def test_probe_without_package(self) -> None:
        try:
            import neural_paw_dft  # noqa: F401

            self.skipTest("neural_paw_dft is installed")
        except ImportError:
            pass
        backend = NeuralPawInitializer(python=sys.executable)
        self.assertFalse(backend.probe()["available"])
        with self.assertRaises(DependencyError):
            backend.require_available()

    def test_missing_interpreter(self) -> None:
        backend = NeuralPawInitializer(python="/nonexistent/python")
        info = backend.probe()
        self.assertFalse(info["available"])


@unittest.skipUnless(os.environ.get("IFACE_NDI_PYTHON"), "set IFACE_NDI_PYTHON to run real neural_paw_dft inference")
class NeuralPawRealIntegrationTest(unittest.TestCase):
    """Real inference on a small NiO cell (needs the package, weights and a POTCAR-free fixture).

    This exercises the upstream models end to end but does not run VASP; see
    docs/density-init.md for the VASP validation protocol.
    """

    def test_real_charge_only_seed_for_afm_nio(self) -> None:
        backend = NeuralPawInitializer(device=os.environ.get("IFACE_NDI_DEVICE", "cpu"))
        if not backend.probe().get("available"):
            self.skipTest("IFACE_NDI_PYTHON cannot import neural_paw_dft")
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "run")
            result = initialize_density(run, backend=backend, grid=(24, 24, 24))
            self.assertEqual(result["status"], "PROMOTED")
            self.assertFalse(result["magnetism"]["spin_channel_written"])
            self.assertIn(f"MAGMOM = {NIO_MAGMOM}", (run / "INCAR").read_text())


if __name__ == "__main__":
    unittest.main()
