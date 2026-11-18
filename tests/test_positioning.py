from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import App
from board import COMPONENTS
from inspection import read_image
from positioning import LabelHit, LabelReader, PositionLocator, combine_hits, fit_label_anchors, modern_rows, parse_labels
from scanner import BoardLocator, corners, transform

REFERENCE=Path(__file__).resolve().parents[1]/'assets'/'smd-board-reference.png'


class PositioningTests(unittest.TestCase):
    def test_two_engines_agree_on_names_but_conflicting_locations_are_rejected(self):
        a=LabelHit('R1',.95,corners((10,10,30,30)))
        b=LabelHit('R1',.99,corners((11,11,31,31)),source='PP-OCRv6')
        self.assertEqual(combine_hits([a,b]),[b])
        c=LabelHit('R1',.99,corners((110,110,130,130)))
        self.assertEqual(combine_hits([a,c]),[])

    def test_four_names_allow_camera_skew_without_loosening_name_confidence(self):
        reference=self.anchors()
        matrix=np.array([[8.,.1,-10],[.2,6.8,-20],[0.,0.,1.]])
        hits=[LabelHit(name,.99,transform(hit.quad,matrix)) for name,hit in reference.items()]
        registration=fit_label_anchors(reference,hits)
        self.assertIsNotNone(registration)
        np.testing.assert_allclose(registration.matrix,matrix,atol=.001)

    def test_only_explicit_designators_are_read_and_duplicate_names_are_ambiguous(self):
        box=corners((10,10,100,30)).tolist()
        rows=[[box,text,score] for text,score in [
            ('R21 R22',.99),('010',.99),('224',.99),('RI',.99),('R?',.99),
            ('R01',.99),('R2',.6),('Q13',.99),('R21',.99)]]
        hits=parse_labels(rows)
        self.assertEqual({hit.component for hit in hits},{'R22','Q13'})

    def anchors(self):
        return {name:LabelHit(name,.99,corners(box)) for name,box in {
            'R1':(70,60,95,75),'R2':(70,98,95,113),
            'R11':(20,60,50,75),'R12':(20,98,50,113)}.items()}

    def test_three_names_and_geometry_are_needed_and_unverified_neighbours_are_excluded(self):
        reference=self.anchors()
        matrix=np.array([[3.,0.,-20],[0.,3.,-100],[0.,0.,1.]])
        hits=[LabelHit(name,.99,transform(hit.quad,matrix)) for name,hit in reference.items()]
        self.assertIsNone(fit_label_anchors(reference,hits[:2]))
        fit=fit_label_anchors(reference,hits)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit.matrix,matrix,atol=.001)
        hits[2].quad+=np.array([120,160],np.float32)
        hits[3].quad+=np.array([-150,170],np.float32)
        self.assertIsNone(fit_label_anchors(reference,hits))
        locator=PositionLocator(REFERENCE)
        fit.anchors=[{'component':'R1'}]
        image=read_image(REFERENCE)
        fit.matrix=np.eye(3)
        self.assertEqual([t.component for t in locator.targets(image,fit,{})],['R1'])

    def test_ocr_rotation_coordinates_return_to_the_original_frame(self):
        image=np.zeros((100,200,3),np.uint8)
        source=corners((20,30,80,50))
        for turn,matrix in [(0,np.eye(3)),(1,np.array([[0,1,0],[-1,0,199],[0,0,1]])),
                            (2,np.array([[-1,0,199],[0,-1,99],[0,0,1]])),
                            (3,np.array([[0,-1,99],[1,0,0],[0,0,1]]))]:
            polygon=transform(source,matrix)*2
            engine=lambda *args,**kwargs:([[polygon.tolist(),'R22',.99]],None)
            hit=LabelReader(engine).read(image,turn)[0]
            np.testing.assert_allclose(hit.quad,source,atol=.001)

    def test_real_pretrained_ocr_locates_rotated_board_crop_when_feature_matching_fails(self):
        image=read_image(REFERENCE)
        zoom=cv2.resize(image[55:180,5:198],None,fx=3,fy=3)
        locator=PositionLocator(REFERENCE)
        with patch.object(BoardLocator,'locate',return_value=None):
            for frame in (zoom,cv2.rotate(zoom,cv2.ROTATE_90_CLOCKWISE)):
                fit=locator.locate(frame)
                self.assertIsNotNone(fit,locator.detail)
                self.assertEqual(fit.method,'ocr_reference_anchors')
                self.assertGreaterEqual(len(fit.anchors),3)
                self.assertIn('R1',[hit['component'] for hit in fit.anchors])

    def test_reading_one_label_does_not_run_quality_ai_or_complete_checklist(self):
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,auto_scan=True,board_store=Path(folder)/'boards')
            try:
                app.locator=PositionLocator(REFERENCE,reader=SimpleNamespace(
                    read=lambda *args:[LabelHit('R22',.99,corners((80,60,150,100)))]))
                app.frame=app.display_frame=read_image(REFERENCE)
                calls=[]
                app.model=SimpleNamespace(predict=lambda **kwargs:calls.append(kwargs))
                with patch.object(BoardLocator,'locate',return_value=None):
                    app.scan_snapshot(app.frame)
                    deadline=time.monotonic()+3
                    while app.busy and time.monotonic()<deadline:
                        root.update();time.sleep(.01)
                self.assertFalse(app.busy)
                self.assertEqual(calls,[])
                self.assertEqual(app.checklist.session['results'],{})
                self.assertEqual(app.board_layout.states['R22'],'SEEN')
                self.assertIn('R22',app.position_detail.get())
                self.assertEqual(app.auto_targets,[])
                app.reset_scan_board();app.render()
                self.assertEqual(app.position_hits,[])
                self.assertEqual(app.board_layout.states['R22'],'PENDING')
            finally:app.close()


if __name__=='__main__':unittest.main()
