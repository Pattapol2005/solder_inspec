import json
import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pin_calibration import save_calibration, read_calibrations


class CalibrationFilesTests(unittest.TestCase):
    def test_local_copy_is_used_even_when_the_original_checkout_still_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / 'original' / 'calibrations'
            copied = Path(folder) / 'copied' / 'calibrations'
            image = np.random.default_rng(9).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            profile = save_calibration(original, 'U1', image, [[150, 50], [150, 160], [470, 50]])
            active = original / 'U1' / 'active.json'
            active.write_text(json.dumps(profile), encoding='utf-8')
            shutil.copytree(original, copied)
            pins, _, errors = read_calibrations(copied)
            self.assertEqual(errors, [])
            self.assertTrue(pins['U1'].path.is_relative_to(copied.resolve()))

    def test_missing_revision_is_not_substituted_with_another_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.random.default_rng(10).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            save_calibration(root, 'U1', image, [[150, 50], [150, 160], [470, 50]])
            active = root / 'U1' / 'active.json'
            data = json.loads(active.read_text('utf-8'))
            data['image'] = r'Z:\old-laptop\calibrations\U1\missing-revision\reference.png'
            active.write_text(json.dumps(data), encoding='utf-8')
            pins, _, errors = read_calibrations(root)
            self.assertEqual(pins, {})
            self.assertIn('missing-revision', errors[0])

    def test_changed_image_does_not_reuse_points_for_the_old_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.random.default_rng(11).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            profile = save_calibration(root, 'U1', image, [[150, 50], [150, 160], [470, 50]])
            with Path(profile['image']).open('ab') as handle:
                handle.write(b'changed')
            pins, _, errors = read_calibrations(root)
            self.assertEqual(pins, {})
            self.assertIn('ไม่ตรงกับจุด', errors[0])

    def test_legacy_reference_uses_this_checkout_after_download_from_github(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'calibrations'
            image = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            save_calibration(root, 'U1', image, [[150, 50], [150, 160], [470, 50]])
            active = root / 'U1' / 'active.json'
            data = json.loads(active.read_text('utf-8'))
            revision = next(path.name for path in active.parent.iterdir() if path.is_dir())
            data['image'] = rf'Z:\old-laptop\solder_inspec\calibrations\U1\{revision}\reference.png'
            active.write_text(json.dumps(data), encoding='utf-8')
            original = active.read_bytes()
            pins, _, errors = read_calibrations(root)
            self.assertEqual(errors, [])
            self.assertEqual(pins['U1'].path, (root / 'U1' / revision / 'reference.png').resolve())
            self.assertEqual(active.read_bytes(), original)

    def test_new_profiles_store_relative_reference_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.random.default_rng(8).integers(0, 255, (480, 640, 3), dtype=np.uint8)
            save_calibration(root, 'U1', image, [[150, 50], [150, 160], [470, 50]])
            active = root / 'U1' / 'active.json'
            data = json.loads(active.read_text('utf-8'))
            self.assertFalse(Path(data['image']).is_absolute())
            self.assertTrue((active.parent / data['image']).is_file())
            history = active.parent / Path(data['image']).parent / 'calibration.json'
            historical = json.loads(history.read_text('utf-8'))
            self.assertEqual(historical['image'], 'reference.png')


if __name__ == '__main__':
    unittest.main()
