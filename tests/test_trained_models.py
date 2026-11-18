from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from inspection import (CLASSES, Detection, prediction_quads, detection_from_quad,
                        validate_names, draw_detection)
from runtime import model_file, load_weights, load_locator_weights
from scanner import assess_component, inspect_target, corners
from trained_locator import TrainedLocator
from test_app import TensorStub


def result(quads, classes, confidence):
    return SimpleNamespace(boxes=None, obb=SimpleNamespace(xyxyxyxy=TensorStub(quads),
                           cls=TensorStub(classes), conf=TensorStub(confidence)))


class TrainedModelTests(unittest.TestCase):
    def test_configured_acquisition_wait_and_immediate_disappearance(self):
        self.assertEqual(TrainedLocator.ACQUIRE_SECONDS,0)
        with tempfile.TemporaryDirectory() as folder:
            frame=np.zeros((200,300,3),np.uint8);q=corners((20,20,180,100))
            locator=self.make_locator(folder,result([q],[0],[.9]),acquire_seconds=.25)
            with patch.object(locator,'clock',return_value=10.) as clock:
                self.assertIsNone(locator.locate(frame))
                self.assertEqual(locator.hits,[])
                clock.return_value=10.24
                self.assertIsNone(locator.locate(frame))
                clock.return_value=10.25
                self.assertIsNotNone(locator.locate(frame))
                locator.model.predict=lambda **kw:[result([],[],[])]
                clock.return_value=10.26
                self.assertIsNone(locator.locate(frame))
                self.assertEqual(locator.hits,[])
                self.assertEqual(locator.pending,{})
                locator.model.predict=lambda **kw:[result([q],[0],[.9])]
                clock.return_value=10.27
                self.assertIsNone(locator.locate(frame))
                clock.return_value=10.4
                self.assertIsNone(locator.locate(frame))
                clock.return_value=10.53
                self.assertIsNotNone(locator.locate(frame))

    def test_below_threshold_score_removes_locator_and_restarts_acquisition(self):
        with tempfile.TemporaryDirectory() as folder:
            frame=np.zeros((200,300,3),np.uint8);q=corners((20,20,180,100))
            locator=self.make_locator(folder,result([q],[0],[.9]),acquire_seconds=.45)
            with patch.object(locator,'clock',return_value=1.) as clock:
                self.assertIsNone(locator.locate(frame))
                clock.return_value=1.5
                self.assertIsNotNone(locator.locate(frame))
                locator.model.predict=lambda **kw:[result([q],[0],[.55])]
                clock.return_value=1.51
                self.assertIsNone(locator.locate(frame))
                self.assertEqual(locator.hits,[])
                locator.model.predict=lambda **kw:[result([q],[0],[.9])]
                clock.return_value=2.
                self.assertIsNone(locator.locate(frame))

    def test_locator_tracks_motion_during_acquisition_while_current_detections_exist(self):
        from test_pin_diode_workflow import texture
        with tempfile.TemporaryDirectory() as folder:
            frame=texture(240,320);q=corners((60,50,200,160))
            locator=self.make_locator(folder,result([q],[0],[.9]),acquire_seconds=.45)
            with patch.object(locator,'clock',return_value=10.) as clock:
                self.assertIsNone(locator.locate(frame))
                moved=cv2.warpAffine(frame,np.float32([[1,0,8],[0,1,6]]),(320,240))
                locator.model.predict=lambda **kw:[result([q+[8,6]],[0],[.9])]
                clock.return_value=10.3
                self.assertIsNone(locator.locate(moved))
                clock.return_value=10.45
                self.assertIsNotNone(locator.locate(moved))
                np.testing.assert_allclose(locator.hits[0].quad,q+[8,6],atol=1)

    def test_changed_identity_waits_again_and_corner_order_remains_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            frame=np.zeros((200,300,3),np.uint8);q=corners((20,20,180,100))
            locator=self.make_locator(folder,result([q],[0],[.9]),acquire_seconds=.45)
            with patch.object(locator,'clock',return_value=10.) as clock:
                locator.locate(frame)
                clock.return_value=10.45
                locator.locate(frame)
                locator.model.predict=lambda **kw:[result([q],[1],[.95])]
                clock.return_value=10.46
                self.assertIsNone(locator.locate(frame))
                self.assertEqual(locator.hits,[])
                clock.return_value=10.92
                self.assertIsNotNone(locator.locate(frame))
                self.assertEqual([hit.component for hit in locator.hits],['R23'])
                locator.model.predict=lambda **kw:[result([np.roll(q+2,2,axis=0)],[1],[.95])]
                clock.return_value=10.93
                locator.locate(frame)
                np.testing.assert_allclose(locator.hits[0].quad,q+1.3,atol=.001)

    def test_changed_threshold_or_large_position_jump_restarts_acquisition(self):
        with tempfile.TemporaryDirectory() as folder:
            frame=np.zeros((200,300,3),np.uint8);q=corners((20,20,180,100))
            locator=self.make_locator(folder,result([q],[0],[.9]),acquire_seconds=.45)
            with patch.object(locator,'clock',return_value=10.) as clock:
                locator.locate(frame)
                clock.return_value=10.45
                self.assertIsNotNone(locator.locate(frame))
                locator.threshold=.8
                clock.return_value=10.5
                self.assertIsNone(locator.locate(frame))
                locator.model.predict=lambda **kw:[result([corners((200,20,290,100))],[0],[.9])]
                clock.return_value=10.9
                self.assertIsNone(locator.locate(frame))
                clock.return_value=11.
                self.assertIsNone(locator.locate(frame))

    def test_actual_training_class_order_maps_by_name(self):
        self.assertEqual(validate_names({0:'bridge',1:'excess',2:'good',3:'missing'}),
                         {0:1,1:2,2:0,3:3})

    def test_obb_results_with_boxes_none_preserve_four_corners_and_empty_predictions(self):
        q=[[10,20],[25,10],[40,30],[25,40]]
        rows=list(prediction_quads(result([q],[2],[.95])))
        np.testing.assert_array_equal(rows[0][0], q)
        detection=detection_from_quad(rows[0][0],0,rows[0][2])
        self.assertEqual(detection.box,(10.,10.,40.,40.))
        self.assertEqual(len(detection.polygon),4)
        canvas=np.zeros((60,60,3),np.uint8)
        draw_detection(canvas,detection)
        self.assertGreater(canvas.sum(),0)
        self.assertEqual(list(prediction_quads(result([],[],[]))),[])

    def test_model_filenames_prefer_user_files_and_keep_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'best.pt').touch()
            self.assertEqual(model_file(root).name,'best.pt')
            (root/'best_inspec.pt').touch();(root/'best_locator.pt').touch()
            self.assertEqual(model_file(root).name,'best_inspec.pt')
            self.assertEqual(model_file(root,'locator').name,'best_locator.pt')

    def test_locator_class_contract_and_wrong_task_are_rejected(self):
        from board import COMPONENTS
        locator=SimpleNamespace(task='obb',names=dict(enumerate(COMPONENTS)))
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=lambda p:locator)}):
            self.assertEqual(set(load_locator_weights('fixture.pt')[1].values()),set(COMPONENTS))
            with self.assertRaises(ValueError):load_weights('fixture.pt')
        invalid=SimpleNamespace(task='classify',names=['good','bridge','excess','missing'])
        with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLO=lambda p:invalid)}):
            with self.assertRaises(ValueError):load_weights('fixture.pt')

    def make_locator(self, folder, prediction, acquire_seconds=0):
        path=Path(folder)/'best_locator.pt';path.write_bytes(b'test-only')
        model=SimpleNamespace(predict=lambda **kw:[prediction])
        locator=TrainedLocator(model,{0:'R22',1:'R23'},path)
        locator.ACQUIRE_SECONDS=acquire_seconds
        return locator

    def test_single_component_needs_no_ocr_and_bad_or_ambiguous_boxes_do_not_get_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            q=corners((20,20,180,100)).tolist()
            frame=np.zeros((200,300,3),np.uint8)
            locator=self.make_locator(folder,result([q],[0],[.99]))
            registration=locator.locate(frame)
            self.assertEqual(registration.method,'trained_locator_obb')
            targets=locator.targets(frame,registration,{})
            self.assertEqual([t.component for t in targets],['R22'])
            self.assertEqual(targets[0].crop.shape,(80,160,3))
            self.assertEqual(locator.targets(frame,registration,{'R22':{'complete':True}}),[])
            for pred in (result([q,q],[0,0],[.99,.98]),
                         result([q,q],[0,1],[.99,.98]),
                         result([corners((0,0,180,100)).tolist()],[0],[.99]),
                         result([q],[0],[.3])):
                self.assertIsNone(self.make_locator(folder,pred).locate(frame))

    def test_quality_crop_restores_obb_and_missing_fails_without_inventing_pin_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            frame=np.zeros((200,300,3),np.uint8)
            locator=self.make_locator(folder,result([corners((20,20,180,100)).tolist()],[0],[.99]))
            registration=locator.locate(frame)
            target=locator.targets(frame,registration,{})[0]
            # The quality model's ID 3 is a whole-footprint missing defect.
            model=SimpleNamespace(predict=lambda **kw:[result([corners((5,5,150,70)).tolist()],[3],[.98])])
            inspected,crop=inspect_target(model,{0:1,1:2,2:0,3:3},frame,target,registration,
                                          .5,'cpu','best_inspec.pt','board01',locator)
            self.assertEqual(inspected[3],'FAIL')
            self.assertEqual(inspected[2][0].box,(25.,25.,170.,90.))
            self.assertEqual(inspected[5]['joint_states'],{1:'REVIEW',2:'REVIEW'})
            self.assertEqual(inspected[5]['joint_identity'],'not_resolved')
            self.assertEqual(inspected[5]['class_names'],list(CLASSES))
            good=[Detection(0,.9,(20,30,45,75)),Detection(0,.9,(140,30,165,75))]
            self.assertEqual(assess_component(good,target)[0],'PASS')
            self.assertEqual(assess_component(good[:1],target)[0],'REVIEW')
            self.assertEqual(assess_component([good[0],good[0]],target)[0],'REVIEW')


if __name__=='__main__':unittest.main()
