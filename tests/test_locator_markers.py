import unittest,tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from board import COMPONENTS
from scanner import corners
from locator_markers import validate_locator_names,diode_marker_reference
from trained_locator import TrainedLocator
from test_trained_models import result
from diode_cv import inspect_diode


class MarkerTests(unittest.TestCase):
    def test_original_and_extended_models_map_by_name_not_class_order(self):
        self.assertEqual(len(validate_locator_names(list(COMPONENTS))),62)
        names=['minus',*reversed(COMPONENTS),'ic_head','plus']
        mapping=validate_locator_names(names)
        self.assertEqual(mapping[0],'minus');self.assertEqual(len(mapping),65)
        with self.assertRaises(ValueError):validate_locator_names(names+['polarity_plus'])
        with self.assertRaises(ValueError):validate_locator_names(['plus','minus','D1'])

    def test_marks_are_separate_from_targets_and_pair_only_inside_same_component(self):
        quads=[corners(b).tolist() for b in [(20,20,280,180),(40,35,60,55),(230,35,250,55),(80,70,100,90)]]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt';path.write_bytes(b'test')
            prediction=result(quads,[2,0,3,1],[.99]*4)
            locator=TrainedLocator(SimpleNamespace(predict=lambda **kw:[prediction]),{0:'plus',1:'ic_head',2:'D1',3:'minus'},path)
            with patch.object(TrainedLocator,'ACQUIRE_SECONDS',0):
                frame=np.zeros((200,320,3),np.uint8);reg=locator.locate(frame)
            self.assertEqual([h.component for h in locator.hits],['D1'])
            self.assertEqual(len(locator.markers),3)
            self.assertEqual([t.component for t in locator.targets(frame,reg,{})],['D1'])
            reference=diode_marker_reference(locator.hits,locator.markers,'D1')
            np.testing.assert_allclose(reference['axis'],[1,0])
            other=SimpleNamespace(component='D2',quad=corners((225,20,300,100)))
            self.assertIsNone(diode_marker_reference(locator.hits+[other],locator.markers,'D1'))
            self.assertIsNone(diode_marker_reference(locator.hits,locator.markers+[locator.markers[0]],'D1'))

    def sample(self,reverse_band=False,signs=True):
        image=np.full((300,600,3),(90,40,15),np.uint8)
        cv2.rectangle(image,(150,150),(450,220),(20,25,200),-1)
        x=190 if reverse_band else 380
        cv2.rectangle(image,(x,150),(x+25,220),(10,10,10),-1)
        if signs:
            for a,b in [((110,65),(170,65)),((140,35),(140,95)),((430,65),(490,65))]:
                cv2.line(image,a,b,(255,255,255),14)
        return image,corners((10,10,590,290))

    def test_both_printed_signs_determine_polarity_even_after_rotation(self):
        for reversed_band in (False,True):
            frame,quad=self.sample(reversed_band)
            for turn in range(4):
                observed=inspect_diode(frame,quad,None)
                self.assertEqual(observed['status'],'FAIL' if reversed_band else 'PASS')
                self.assertEqual(observed['polarity_reference']['source'],'printed_plus_minus_cv')
                h=frame.shape[0];frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                quad=np.float32([[h-1-y,x] for x,y in quad])

    def test_no_single_sign_or_disabled_pairing_can_certify_polarity(self):
        frame,quad=self.sample(signs=False)
        self.assertEqual(inspect_diode(frame,quad,None)['status'],'REVIEW')
        frame,quad=self.sample();frame[25:100,100:180]=(90,40,15)
        self.assertEqual(inspect_diode(frame,quad,None)['status'],'REVIEW')
        frame,quad=self.sample()
        self.assertEqual(inspect_diode(frame,quad,None,allow_printed_reference=False)['status'],'REVIEW')

if __name__=='__main__':unittest.main()
