import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inspection import Detection, decision, read_image, save_report, validate_names,visible_detections,detection_from_quad


class InspectionTests(unittest.TestCase):
    def test_missing_hides_contained_quality_boxes_even_with_lower_confidence(self):
        missing=Detection(3,.32,(0,0,100,100))
        inside=[Detection(kind,.99,(10,10,40,40)) for kind in (0,1,2)]
        outside=Detection(0,.90,(110,10,140,40))
        self.assertEqual(visible_detections([*inside,missing,outside]),[missing,outside])

    def test_rotated_missing_uses_polygon_instead_of_its_axis_aligned_bounds(self):
        missing=detection_from_quad([[100,20],[180,100],[100,180],[20,100]],3,.5)
        inside=Detection(0,.99,(80,80,120,120))
        outside=Detection(0,.99,(22,22,40,40))
        self.assertEqual(visible_detections([inside,missing,outside]),[missing,outside])

    def test_nested_missing_keeps_only_outer_region_and_duplicates_keep_stronger_one(self):
        outer=Detection(3,.40,(0,0,100,100))
        inner=Detection(3,.95,(10,10,30,30))
        stronger=Detection(3,.80,(0,0,100,100))
        self.assertEqual(visible_detections([outer,inner,stronger]),[stronger])

    def test_qc_never_passes_empty_or_incomplete(self):
        good = Detection(0, .9, (1, 2, 20, 30))
        self.assertEqual(decision([], 2, .5)[0], 'REVIEW')
        self.assertEqual(decision([good], 0, .5)[0], 'REVIEW')
        self.assertEqual(decision([good], 2, .5)[0], 'REVIEW')
        self.assertEqual(decision([good], 1, .95)[0], 'REVIEW')
        self.assertEqual(decision([good], 1, .5)[0], 'PASS')
        self.assertEqual(decision([good, good], 1, .5)[0], 'REVIEW')

    def test_defects_override_coverage(self):
        for class_id in (1, 2, 3):
            self.assertEqual(decision([Detection(class_id, .8, (0, 0, 10, 10))], 0, .5)[0], 'FAIL')

    def test_model_contract(self):
        validate_names({0: 'good', 1: 'bridge', 2: 'excess', 3: 'missing'})
        self.assertEqual(validate_names({0: 'bridge', 1: 'excess', 2: 'good', 3: 'missing'}),
                         {0: 1, 1: 2, 2: 0, 3: 3})
        with self.assertRaises(ValueError):
            validate_names({0: 'person'})

    def test_report_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = np.zeros((100, 100, 3), np.uint8)
            detection = Detection(1, .8, (0, 0, 30, 30))
            folder = save_report(temp, frame, frame, [detection], 'FAIL', 'ข้อบกพร่อง',
                                 dict(region='SOP16', expected=16, threshold=.5, model='test.pt'))
            data = json.loads((folder / 'result.json').read_text('utf-8'))
            self.assertEqual(data['detections'][0]['class_id'], 1)
            with (Path(temp) / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['status'], 'FAIL')
            self.assertEqual(rows[0]['count'], '1')
            self.assertTrue((folder / 'raw.png').is_file())


if __name__ == '__main__':
    unittest.main()
