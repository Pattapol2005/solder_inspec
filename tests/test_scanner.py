from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from board import BoardChecklist
from calibration import normalize_reference
from inspection import Detection, read_image
from scanner import BoardLocator, Registration, StableCapture, assess_pads


REFERENCE=Path(__file__).resolve().parents[1]/'assets'/'smd-board-reference.png'


class ScannerTests(unittest.TestCase):
    def test_shift_along_ic_lead_uses_measured_outline_and_rejects_two_pin_box(self):
        from scanner import corners,observed_pin_boxes
        pads={4:corners((10,10,110,30)),5:corners((10,44,110,64))}
        joint=corners((64,11,164,31))
        boxes=observed_pin_boxes(pads,[joint])
        self.assertEqual(set(boxes),{4})
        np.testing.assert_allclose(boxes[4],joint)
        self.assertFalse(observed_pin_boxes(pads,[corners((64,11,164,62))]))
        self.assertFalse(observed_pin_boxes(pads,[joint,joint]))

    @classmethod
    def setUpClass(cls):
        cls.image=read_image(REFERENCE)
        cls.locator=BoardLocator(REFERENCE)

    def test_zoomed_and_rotated_view_identifies_r1_without_operator(self):
        zoom=cv2.resize(self.image[67:113,60:114],None,fx=6,fy=6)
        for image in (zoom,cv2.rotate(zoom,cv2.ROTATE_90_CLOCKWISE)):
            registration=self.locator.locate(image)
            self.assertIsNotNone(registration)
            targets=self.locator.targets(image,registration,{})
            self.assertEqual([t.component for t in targets],['R1'])
            self.assertEqual(targets[0].joints,[1,2])

    def test_no_guess_from_blank_or_pad_only_image(self):
        blank=np.zeros((400,600,3),np.uint8)
        pad_only=cv2.resize(self.image[81:102,68:107],None,fx=6,fy=6)
        self.assertIsNone(self.locator.locate(blank))
        self.assertIsNone(self.locator.locate(pad_only))

    def test_motion_blur_and_duplicate_camera_frame_do_not_trigger_capture(self):
        gate=StableCapture(hold=.5,min_sharpness=18)
        frame=cv2.resize(self.image,(640,664))
        self.assertFalse(gate.observe(frame,1,now=0)[0])
        self.assertFalse(gate.observe(frame,2,now=.1)[0])
        self.assertFalse(gate.observe(frame,2,now=1)[0])
        self.assertFalse(gate.observe(frame,3,now=1.1)[0]) # Camera stalled, restart hold.
        self.assertFalse(gate.observe(frame,4,now=1.2)[0])
        self.assertFalse(gate.observe(frame,5,now=1.4)[0])
        self.assertFalse(gate.observe(frame,6,now=1.6)[0])
        self.assertTrue(gate.observe(frame,7,now=1.8)[0])
        moved=np.roll(frame,30,axis=1)
        self.assertFalse(gate.observe(moved,8,now=1.9)[0])
        self.assertFalse(gate.observe(np.zeros_like(frame),9,now=2)[0])

    def test_bridge_maps_to_both_pads_but_one_large_good_box_cannot_pass(self):
        target=next(t for t in self.locator.targets(self.image,Registration(np.eye(3),20,1,0),{}) if t.component=='R1')
        good=[Detection(0,.9,(66,80,81,102)),Detection(0,.9,(92,80,107,102))]
        self.assertEqual(assess_pads(good,target)[2],{1:'PASS',2:'PASS'})
        bridge=[Detection(1,.9,(66,80,107,102))]
        self.assertEqual(assess_pads(bridge,target)[2],{1:'FAIL',2:'FAIL'})
        self.assertEqual(assess_pads([Detection(0,.9,(66,80,107,102))],target)[0],'REVIEW')
        self.assertEqual(assess_pads([good[0],good[0]],target)[0],'REVIEW')

    def test_partial_ic_does_not_get_full_coverage_and_completed_pads_are_skipped(self):
        frame=self.image[90:140,326:407]
        matrix=np.array([[1.,0.,-326],[0.,1.,-90],[0.,0.,1.]])
        targets=self.locator.targets(frame,Registration(matrix,20,1,0),{})
        target=next(t for t in targets if t.component=='U1')
        self.assertLess(len(target.joints),14)
        self.assertGreater(len(target.joints),0)
        covered={'U1':{'coverage':{str(j):{'confirmed':True,'status':'PASS'} for j in target.joints}}}
        self.assertNotIn('U1',[t.component for t in self.locator.targets(frame,Registration(matrix,20,1,0),covered)])

    def test_auto_joint_results_keep_previous_fail_and_count_bridge_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=BoardChecklist(Path(folder)/'boards')
            report=Path(folder)/'report';report.mkdir();(report/'result.json').write_text('{}')
            ledger.record('R1','FAIL','bridge',report,2,None,0,[1,2],joint_states={1:'FAIL',2:'FAIL'})
            self.assertEqual(ledger.completed(),1)
            ledger.record('R1','PASS','noise',report,2,None,0,[1,2],joint_states={1:'PASS',2:'PASS'})
            self.assertEqual(ledger.session['results']['R1']['status'],'FAIL')

    def test_reference_alignment_validates_corner_order(self):
        normalized=normalize_reference(self.image,[(10,14),(409,14),(409,439),(10,439)])
        self.assertEqual(normalized.shape,(1341,1293,3))
        for points in [[(10,14),(409,439),(409,14),(10,439)],[(0,0)]*4]:
            with self.assertRaises(ValueError):normalize_reference(self.image,points)


if __name__=='__main__':unittest.main()
