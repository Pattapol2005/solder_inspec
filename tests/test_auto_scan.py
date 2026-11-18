from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import App
from inspection import read_image
from test_app import TensorStub


class AutoScanTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.root=tk.Tk();self.root.withdraw()
        base=Path(self.folder.name)
        reference=Path(__file__).resolve().parents[1]/'assets'/'smd-board-reference.png'
        self.app=App(self.root,auto_model=False,monitor=False,auto_scan=True,
                     board_store=base/'boards',report_root=base/'logs',reference_file=reference)
        source=read_image(reference)
        self.frame=cv2.resize(source[67:113,60:114],None,fx=6,fy=6)
        self.calls=[]
        def predict(**kwargs):
            image=kwargs['source'];h,w=image.shape[:2];self.calls.append(image.copy())
            boxes=SimpleNamespace(xyxy=TensorStub([[0,0,w*.38,h],[w*.62,0,w,h]]),cls=TensorStub([0,0]),conf=TensorStub([.9,.9]))
            return [SimpleNamespace(boxes=boxes)]
        self.app.model=SimpleNamespace(predict=predict);self.app.model_path='test.pt'

    def tearDown(self):
        self.app.close();self.folder.cleanup()

    def pump(self,condition,timeout=5):
        deadline=time.monotonic()+timeout
        while not condition() and time.monotonic()<deadline:
            self.root.update();time.sleep(.01)
        self.assertTrue(condition())

    def test_camera_stream_captures_r1_and_keeps_running_without_manual_selection(self):
        frame=self.frame
        class CameraStub:
            error=None
            stop_event=threading.Event()
            sequence=0
            def latest_sample(self):
                self.sequence+=1
                return self.sequence,frame.copy()
            def stop(self):self.stop_event.set()
        self.app.camera=CameraStub();self.app.frozen=False
        # The low-resolution product photo is deliberately enlarged for this test.
        self.app.scan_sharpness.set('5')
        self.pump(lambda:self.app.checklist.completed()==1)
        self.assertFalse(self.app.frozen)
        self.assertEqual(self.app.checklist.session['results']['R1']['status'],'PASS')
        self.assertEqual(self.app.result[5]['component_source'],'reference_registration')
        self.assertTrue((self.app.report_path/'crop.png').exists())
        self.assertEqual(len(self.calls),1)
        self.pump(lambda:not self.app.busy)
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:not self.app.busy)
        self.assertEqual(len(self.calls),1)

    def test_static_image_automatic_scan_does_not_need_name_or_boxes(self):
        self.app.frame=self.app.display_frame=self.frame
        self.app.inspect()
        self.pump(lambda:self.app.checklist.completed()==1)
        self.assertEqual(self.app.component.get(),'R1')
        self.assertEqual(self.app.result[3],'PASS')
        self.assertEqual(self.app.result[5]['joints'],[1,2])
        self.app.save_result()
        self.assertEqual(len(list(self.app.report_root.glob('*/result.json'))),1)

    def test_no_model_only_locates_does_not_invent_inspection(self):
        self.app.model=None
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:not self.app.busy)
        self.assertEqual([t.component for t in self.app.auto_targets],['R1'])
        self.assertIn('best.pt',self.app.scan_status.get())
        self.assertEqual(self.app.checklist.session['results'],{})

    def test_new_view_retries_review_without_repeating_unchanged_view(self):
        good_predict=self.app.model.predict
        calls=[]
        def missing(**kwargs):
            calls.append(1)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=TensorStub([]),cls=TensorStub([]),conf=TensorStub([])))]
        self.app.model.predict=missing
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:not self.app.busy)
        self.assertEqual(self.app.checklist.session['results']['R1']['status'],'REVIEW')
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:not self.app.busy)
        self.assertEqual(len(calls),1)
        self.app.frozen=False;self.app.scan_sharpness.set('5')
        self.app.maybe_scan(cv2.flip(self.frame,1),1)
        self.app.frozen=True;self.app.model.predict=good_predict
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:self.app.checklist.completed()==1)
        self.assertEqual(self.app.checklist.session['results']['R1']['status'],'PASS')

    def test_added_model_resumes_static_automatic_inspection(self):
        import os
        model=self.app.model
        self.app.model=None;self.app.frame=self.app.display_frame=self.frame
        self.app.scan_snapshot(self.frame)
        self.pump(lambda:not self.app.busy)
        weights=Path(self.folder.name)/'best.pt'
        weights.write_bytes(b'test weights')
        old=time.time()-10;os.utime(weights,(old,old))
        self.app.model_file=weights
        with patch('app.load_weights',return_value=(model,self.app.class_map)):
            self.app.check_model()
            self.pump(lambda:self.app.checklist.completed()==1)
        self.assertEqual(self.app.checklist.session['results']['R1']['status'],'PASS')

    def test_stop_discards_inflight_capture(self):
        entered=threading.Event();release=threading.Event()
        original=self.app.model.predict
        def predict(**kwargs):
            entered.set();release.wait(3)
            return original(**kwargs)
        self.app.model.predict=predict
        self.app.scan_snapshot(self.frame)
        self.pump(entered.is_set)
        self.app.auto_mode.set(False);self.app.change_scan_mode()
        release.set();self.pump(lambda:not self.app.busy)
        self.assertEqual(self.app.checklist.session['results'],{})
        self.assertFalse(self.app.report_root.exists())

    def test_storage_failure_does_not_mark_auto_pass(self):
        with patch('auto_scan.save_auto_report',side_effect=OSError('disk full')):
            self.app.scan_snapshot(self.frame)
            self.pump(lambda:not self.app.busy)
        self.assertEqual(self.app.checklist.session['results'],{})
        self.assertIn('disk full',self.app.scan_status.get())

    def test_trained_locator_and_obb_quality_update_checklist_without_ocr(self):
        locator_path=Path(self.folder.name)/'best_locator.pt'
        locator_path.write_bytes(b'test-only')
        quad=[[20,20],[180,20],[180,100],[20,100]]
        locator_result=SimpleNamespace(boxes=None,obb=SimpleNamespace(
            xyxyxyxy=TensorStub([quad]),cls=TensorStub([0]),conf=TensorStub([.99])))
        self.app.locator_model=SimpleNamespace(predict=lambda **kwargs:[locator_result])
        self.app.locator_class_map={0:'R22'}
        self.app.locator_file=locator_path
        qc_result=SimpleNamespace(boxes=None,obb=SimpleNamespace(
            xyxyxyxy=TensorStub([[[25,40],[55,40],[55,80],[25,80]],
                                  [[145,40],[175,40],[175,80],[145,80]]]),
            cls=TensorStub([2,2]),conf=TensorStub([.95,.96])))
        self.app.model=SimpleNamespace(predict=lambda **kwargs:[qc_result])
        self.app.class_map={0:1,1:2,2:0,3:3}
        self.app.frame=self.app.display_frame=np.zeros((200,300,3),np.uint8)
        with patch('auto_scan.PositionLocator',side_effect=AssertionError('OCR must not run')):
            self.app.scan_snapshot(self.app.frame)
            self.pump(lambda:not self.app.busy)
        self.assertEqual(self.app.checklist.session['results']['R22']['status'],'PASS')
        self.assertEqual(self.app.result[5]['component_source'],'trained_locator_obb')
        self.assertEqual(self.app.result[5]['joint_identity'],'not_resolved')
        self.assertTrue((self.app.report_path/'crop.png').exists())
        self.assertEqual(self.app.result[2][0].polygon,((25.,40.),(55.,40.),(55.,80.),(25.,80.)))


if __name__=='__main__':unittest.main()
