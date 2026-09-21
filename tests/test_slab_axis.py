"""Regression tests for axis-aware LOCPOT analysis."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from interfaceforge.slab_alignment import (
    read_locpot, analyze_profile, ionic_center_fraction, write_dipole_preview,
    load_alignment_config, analyze_slab_alignment,
)


class AxisTests(unittest.TestCase):
    def make_locpot(self, root, axis):
        i = 'xyz'.index(axis)
        cell = np.eye(3) * 10
        cell[i, i] = 40
        coords = np.full((3, 3), .5)
        coords[:, i] = [.30, .60, .65]
        header = 'Synthetic\n1\n' + '\n'.join(' '.join(map(str, row)) for row in cell)
        header += '\nPb I H\n1 1 1\nDirect\n' + '\n'.join(' '.join(map(str, row)) for row in coords) + '\n'
        grid = [2, 3, 4]
        grid[i] = 80
        pos = np.arange(80) * .5
        shifted = (pos - 39) % 40
        profile = np.where(shifted < 13, 4.8, np.where(shifted > 27, 5.2, 2.0))
        shape = [1, 1, 1]
        shape[2-i] = 80
        data = np.broadcast_to(profile.reshape(shape), tuple(grid[::-1]))
        path = root / 'LOCPOT'
        path.write_text(header + '\n' + ' '.join(map(str, grid)) + '\n' + ' '.join(map(str, data.ravel())))
        return path, profile

    def test_axes_equivalent(self):
        centers = []
        for axis in 'xyz':
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as tmp:
                path, expected = self.make_locpot(Path(tmp), axis)
                structure, grid, actual = read_locpot(path, axis)
                np.testing.assert_allclose(actual, expected)
                profile, _, _ = analyze_profile(structure, grid, actual)
                self.assertAlmostEqual(profile.high.plateau_eV, 5.2)
                self.assertEqual(profile.high.side, f'high-{axis}')
                centers.append(ionic_center_fraction(structure)[0])
        np.testing.assert_allclose(centers, centers[0])

    def test_config_and_preview(self):
        for axis in 'xyz':
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / 'config.json'
                config.write_text(json.dumps({'axis': axis}))
                self.assertEqual(load_alignment_config(config)['side'], f'high-{axis}')
                (root/'INCAR').write_text('DIPOL = 0.1 0.2 0.3\nIDIPOL = 3\n')
                output = write_dipole_preview(root, .45, axis).read_text()
                self.assertIn(f'IDIPOL = {"xyz".index(axis)+1}', output)
                values = [.1, .2, .3]
                values['xyz'.index(axis)] = .45
                self.assertIn('DIPOL  = ' + ' '.join(f'{x:.6f}' for x in values), output)
                config.write_text(json.dumps({'axis': axis, 'side': 'invalid'}))
                with self.assertRaises(Exception):
                    load_alignment_config(config)

    def test_wrong_recorded_axis_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calc = root/'slab'
            calc.mkdir()
            self.make_locpot(calc, 'x')
            (calc/'INCAR').write_text('IDIPOL = 1\nLDIPOL = .TRUE.\n')
            (calc/'OUTCAR').write_text('IDIPOL = 3\n E-fermi : 1.0\n')
            config = root/'config.json'
            config.write_text(json.dumps({'axis':'x', 'side':'high-x', 'references':[{'prefix':'slab','reference':'slab'}]}))
            with patch('interfaceforge.slab_alignment._plot_profile'), patch('interfaceforge.slab_alignment._plot_workfunction_profile'):
                result = analyze_slab_alignment(root, config='config.json')
            row = result['rows'][0]
            self.assertEqual(row['flatness_status'], 'FAILED_DIPOLE_AXIS')
            self.assertEqual(row['dipole_axis_status'], 'MISMATCH')
            self.assertIn('IDIPOL = 1', (calc/'INCAR.dipole_fix').read_text())
            self.assertTrue((calc/'locpot.dat').read_text().startswith('# shifted_x_A'))
