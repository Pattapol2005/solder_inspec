from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np

from inspection import read_image
from positioning import PositionLocator,nominal_label_center
from reference_views import align_view,save_reference_view

REFERENCE=Path(__file__).resolve().parents[1]/'assets'/'smd-board-reference.png'


class ReferenceViewTests(unittest.TestCase):
    def view(self,top,names):
        image=read_image(REFERENCE)
        frame=cv2.resize(image[top:top+140,:195],None,fx=3,fy=3)
        markers={name:((nominal_label_center(name)-[0,top])*3).tolist() for name in names}
        return frame,markers

    def test_microscope_view_can_be_registered_without_full_board_or_training(self):
        frame,markers=self.view(55,['R1','R11','R2','R12'])
        with tempfile.TemporaryDirectory() as folder:
            path=save_reference_view(Path(folder)/'board.png',frame,markers)
            locator=PositionLocator(path)
            registration=locator.locate(frame)
            self.assertIsNotNone(registration)
            self.assertEqual(set(locator.operator_anchors),set(markers))
            self.assertIn('R1',[t.component for t in locator.targets(frame,registration,{})])

    def test_more_views_keep_previous_anchors_and_do_not_trust_a_mismatched_manifest(self):
        frame,markers=self.view(55,['R1','R11','R2','R12'])
        second,more=self.view(130,['R3','R13','R4','R14'])
        with tempfile.TemporaryDirectory() as folder:
            path=save_reference_view(Path(folder)/'board.png',frame,markers)
            save_reference_view(path,second,more)
            locator=PositionLocator(path)
            self.assertEqual(set(locator.operator_anchors),set(markers)|set(more))
            metadata=json.loads(path.with_suffix('.json').read_text('utf-8'))
            self.assertEqual(len(metadata['views']),2)
            metadata['image_sha256']='wrong'
            path.with_suffix('.json').write_text(json.dumps(metadata),'utf-8')
            self.assertEqual(PositionLocator(path).operator_anchors,{})

    def test_calibration_rejects_collinear_outside_or_inconsistent_markers(self):
        frame,markers=self.view(55,['R1','R11','R2','R12'])
        for points in ({'R1':[20,20],'R2':[40,40],'R3':[60,60]},
                       {'R1':[-1,30],'R11':[200,30],'R2':[100,180]},
                       {'R1':[10,10],'R11':[11,11]},
                       {'R1':[10,10],'Q99':[200,30],'R2':[100,180]}):
            with self.assertRaises(ValueError):align_view(frame,points)


if __name__=='__main__':unittest.main()
