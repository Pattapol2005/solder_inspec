from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from frame_registration import FeatureFrame
from inspection import detection_from_quad
from pin_calibration import PinCalibration
from pin_recovery import recover_from_head
from scanner import corners, observed_pin_boxes, transform, pad_boxes
from test_pin_diode_workflow import texture
from test_trained_models import result
from trained_locator import TrainedLocator


class PinRecoveryTests(unittest.TestCase):
    def fixture(self, component='U1'):
        frame=texture(800,1000,seed=202)
        reference=texture(480,640,seed=15)
        if component=='U1':
            points=np.float32([[100,100],[100,160],[500,100]])
            pin1=np.float32([150,160]);step=np.float32([0,70]);opposite=np.float32([800,160])
            dimensions=(180,36)
        else:
            points=np.float32([[500,80],[455,80],[500,400]])
            pin1=np.float32([800,140]);step=np.float32([-60,0]);opposite=np.float32([800,610])
            dimensions=(30,150)
        total=14 if component=='U1' else 28
        head_offset=np.float32([.4,.3])
        source_head=points[0]+head_offset[0]*(points[1]-points[0])+head_offset[1]*(points[2]-points[0])
        marker=pin1+head_offset[0]*step+head_offset[1]*(opposite-pin1)
        calibration=PinCalibration(component,reference,points,'reference.png')
        boxes={}
        dx,dy=np.float32(dimensions)/2
        for index in range(4):
            for pin,origin in ((1+index,pin1),(total-index,opposite)):
                x,y=origin+step*index
                boxes[pin]=corners((x-dx,y-dy,x+dx,y+dy))
        detections=[detection_from_quad(q,0,.95,2,'good') for q in boxes.values()]
        hits=[SimpleNamespace(component=component,quad=corners((5,5,995,795)),confidence=.95)]
        markers=[SimpleNamespace(component='ic_head',quad=corners((* (marker-15),*(marker+15))),confidence=.95)]
        prediction=result([corners((* (source_head-12),*(source_head+12)))],[0],[.9])
        locator=SimpleNamespace(model=SimpleNamespace(predict=lambda **kw:[prediction]),mapping={0:'ic_head'},device='cpu')
        return calibration,frame,hits,markers,detections,locator,boxes

    def test_manual_reference_and_two_rows_recover_physical_numbers_in_all_rotations(self):
        for component in ('U1','U4'):
            for turns in range(4):
                with self.subTest(component=component,turns=turns):
                    calibration,frame,hits,markers,detections,locator,expected=self.fixture(component)
                    for _ in range(turns):
                        h=frame.shape[0]
                        matrix=np.float32([[0,-1,h-1],[1,0,0],[0,0,1]])
                        frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                        for hit in [*hits,*markers]:hit.quad=transform(hit.quad,matrix)
                        expected={pin:transform(q,matrix) for pin,q in expected.items()}
                        detections=[detection_from_quad(transform(d.polygon,matrix),0,.95,2,'good') for d in detections]
                    registration=recover_from_head(calibration,FeatureFrame(frame),hits,markers,detections,locator)
                    self.assertIsNotNone(registration)
                    target=calibration.target(frame,registration,[d.polygon for d in detections])
                    measured=observed_pin_boxes(target.pad_polygons,[d.polygon for d in detections])
                    self.assertEqual(set(measured),set(expected))
                    for pin,q in measured.items():np.testing.assert_allclose(q,expected[pin],atol=.01)
                    self.assertIsNotNone(calibration.last_registration)

    def test_a_dot_alone_a_single_row_or_a_missing_first_pad_cannot_start_numbering(self):
        for change in ('dot_only','single_row','missing_first','two_heads','no_head','shifted_head'):
            with self.subTest(change=change):
                calibration,frame,hits,markers,detections,locator,_=self.fixture()
                if change=='dot_only':detections=[]
                if change=='single_row':detections=detections[::2]
                if change=='missing_first':detections=detections[2:]
                if change=='two_heads':markers=markers+markers
                if change=='no_head':markers=[]
                if change=='shifted_head':markers[0].quad+=np.float32([0,70])
                self.assertIsNone(recover_from_head(calibration,FeatureFrame(frame),hits,markers,detections,locator))
                self.assertIsNone(calibration.last_registration)

    def test_recovered_pose_tracks_overlapping_views_without_the_head_and_keeps_pin_indices(self):
        calibration,world,hits,markers,detections,locator,expected=self.fixture()
        first=world[:480].copy()
        hits[0].quad=corners((5,5,995,475))
        registration=recover_from_head(calibration,FeatureFrame(first),hits,markers,detections,locator)
        self.assertIsNotNone(registration)
        # Follow overlapping views down the package. No new marker is supplied.
        for offset in (60,120,180,240):
            frame=world[offset:offset+480].copy()
            tracked=calibration.locate(FeatureFrame(frame))
            self.assertIsNotNone(tracked,calibration.detail)
            truth=np.float32([[1,0,0],[0,1,-offset],[0,0,1]])@registration.matrix
            centers=np.float32([corners(box).mean(axis=0) for box in pad_boxes('U1').values()])
            np.testing.assert_allclose(transform(centers,tracked.matrix),transform(centers,truth),atol=1.5)
        self.assertIsNone(calibration.locate(FeatureFrame(np.zeros((480,1000,3),np.uint8))))

    def test_only_overlapping_duplicate_ic_boxes_are_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            weights=Path(folder)/'weights.pt';weights.write_bytes(b'test')
            outer=corners((20,20,480,470));inner=corners((30,50,470,450))
            prediction=result([outer,inner],[0,0],[.96,.85])
            model=SimpleNamespace(predict=lambda **kw:[prediction])
            locator=TrainedLocator(model,{0:'U1'},weights)
            locator.locate(np.zeros((500,700,3),np.uint8))
            self.assertEqual(len(locator.hits),1)
            model.predict=lambda **kw:[result([corners((20,20,300,450)),corners((350,20,650,450))],[0,0],[.96,.85])]
            self.assertIsNone(locator.locate(np.zeros((500,700,3),np.uint8)))
            self.assertEqual(len(locator.hits),2)


if __name__=='__main__':unittest.main()
