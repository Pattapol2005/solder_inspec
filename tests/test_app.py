from types import SimpleNamespace
import time
import tkinter as tk
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import json

from app import App


class TensorStub:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class AppTests(unittest.TestCase):
    def test_jobs_reuse_one_worker_and_keep_callbacks_on_ui_thread(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,board_store=Path(folder)/'boards',report_root=Path(folder)/'reports')
            workers=[];callbacks=[];ui=threading.current_thread()
            try:
                for _ in range(4):
                    app.job(lambda: threading.current_thread(),
                            lambda worker:(workers.append(worker),callbacks.append(threading.current_thread())))
                    self.wait_job(root,app)
                self.assertEqual(len(workers),4)
                self.assertTrue(all(worker is workers[0] for worker in workers))
                self.assertIsNot(workers[0],ui)
                self.assertTrue(all(thread is ui for thread in callbacks))
            finally:app.close()
            with self.assertRaises(RuntimeError):app.worker.submit(lambda:None)

    def test_manual_diode_inspection_also_requires_and_uses_cv_direction(self):
        from diode_cv import DiodeDirection
        from test_pin_diode_workflow import DiodeTests
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,board_store=Path(folder)/'boards',report_root=Path(folder)/'reports')
            try:
                frame=DiodeTests().image([340])
                boxes=SimpleNamespace(xyxy=TensorStub([[60,65,105,145],[395,65,440,145]]),
                                      cls=TensorStub([2,2]),conf=TensorStub([.95,.96]))
                names={0:'bridge',1:'excess',2:'good',3:'missing'}
                app.model=SimpleNamespace(names=names,predict=lambda **kwargs:[SimpleNamespace(boxes=boxes,names=names)])
                app.model_path='best_inspec.pt';app.frame=app.display_frame=frame
                app.select_component('D1');app.inspect();self.wait_job(root,app)
                self.assertEqual(app.result[3],'REVIEW')
                self.assertEqual(app.checklist.session['results']['D1']['diode_polarity_state'],'REVIEW')
                app.diode_direction=DiodeDirection(frame,[[80,105],[420,105]],Path(folder)/'reference.png')
                app.inspect();self.wait_job(root,app)
                self.assertEqual(app.result[3],'PASS')
                self.assertEqual(app.checklist.session['results']['D1']['diode_polarity_state'],'PASS')
            finally:app.close()

    def wait_job(self, root, app):
        deadline = time.monotonic() + 5
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        self.assertFalse(app.busy)

    def test_model_auto_discovery_replacement_and_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            root = tk.Tk()
            root.withdraw()
            app = App(root, model_file=path, auto_model=False, monitor=False)
            def check():
                if app.watch_id:
                    root.after_cancel(app.watch_id)
                app.check_model()
            model = SimpleNamespace(task='detect', names=['bridge', 'excess', 'good', 'missing'])
            try:
                check()
                self.assertIsNone(app.model)
                self.assertIn('รอไฟล์', app.model_label.get())
                path.write_bytes(b'test placeholder')
                os.utime(path, (time.time()-10, time.time()-10))
                with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda p: model)}):
                    check()
                    self.wait_job(root, app)
                    self.assertIs(app.model, model)
                    self.assertEqual(app.class_map[0], 1)
                    self.assertEqual(app.model_path, str(path))
                    path.write_bytes(b'replacement placeholder')
                    os.utime(path, (time.time()-5, time.time()-5))
                    check()
                    self.wait_job(root, app)
                    self.assertIs(app.model, model)
                path.unlink()
                check()
                self.assertIsNone(app.model)
                self.assertEqual(str(app.inspect_btn['state']), 'disabled')
            finally:
                app.close()

    def test_invalid_auto_model_is_visible_without_dialog(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            path.write_bytes(b'invalid')
            os.utime(path, (time.time()-10, time.time()-10))
            root = tk.Tk()
            root.withdraw()
            app = App(root, model_file=path, auto_model=False, monitor=False)
            try:
                wrong = SimpleNamespace(task='detect', names=['person'])
                with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda p: wrong)}):
                    app.check_model()
                    self.wait_job(root, app)
                self.assertIsNone(app.model)
                self.assertIn('ไม่สำเร็จ', app.model_label.get())
                self.assertIn('good', app.model_detail.get())
            finally:
                app.close()

    def test_inference_freezes_and_keeps_raw_image(self):
        root = tk.Tk()
        root.withdraw()
        app = App(root, auto_model=False, monitor=False)
        try:
            app.frame = app.display_frame = np.zeros((100, 100, 3), np.uint8)
            boxes = SimpleNamespace(xyxy=TensorStub([[10, 10, 40, 40]]),
                                    cls=TensorStub([0]), conf=TensorStub([.9]))
            # External class 0 is bridge, not good, in this model.
            app.class_map = {0: 1, 1: 3, 2: 0, 3: 2}
            app.model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(boxes=boxes)])
            app.model_path = 'test-only.pt'
            app.expected.set('2')
            app.inspect()
            deadline = time.monotonic() + 5
            while app.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(.02)
            self.assertFalse(app.busy)
            self.assertTrue(app.frozen)
            self.assertEqual(app.result[3], 'FAIL')
            self.assertEqual(app.result[2][0].class_id, 1)
            self.assertEqual(app.result[5]['expected'], 2)
            self.assertEqual(np.count_nonzero(app.result[0]), 0)
            self.assertGreater(np.count_nonzero(app.result[1]), 0)
            self.assertEqual(len(app.table.get_children()), 1)
            self.assertEqual(app.counts[1].get(), '1')
            app.overlay.set(False)
            app.toggle_overlay()
            self.assertEqual(np.count_nonzero(app.display_frame), 0)
        finally:
            app.close()

    def test_selected_component_uses_crop_and_saves_checklist(self):
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,board_store=Path(folder)/'boards',report_root=Path(folder)/'logs')
            try:
                received=[]
                boxes=SimpleNamespace(xyxy=TensorStub([[13,5,23,20],[0,25,8,40]]),cls=TensorStub([0,0]),conf=TensorStub([.9,.9]))
                def predict(**kwargs):
                    received.append(kwargs['source'].copy())
                    return [SimpleNamespace(boxes=boxes)]
                app.model=SimpleNamespace(predict=predict);app.model_path='test.pt'
                app.frame=app.display_frame=np.zeros((60,80,3),np.uint8)
                app.select_component('R1')
                app.roi=(10,15,60,45);app.rotation=90
                app.inspect();self.wait_job(root,app)
                self.assertEqual(received[0].shape,(50,30,3))
                self.assertEqual(app.result[2][0].box,(15.,22.,30.,32.))
                self.assertEqual(app.result[0].shape,(60,80,3))
                self.assertEqual(app.checklist.completed(),1)
                self.assertEqual(app.checklist.session['results']['R1']['status'],'PASS')
                report=app.report_path
                self.assertTrue((report/'crop.png').is_file())
                metadata=json.loads((report/'result.json').read_text('utf-8'))
                self.assertEqual(metadata['component'],'R1')
                self.assertEqual(metadata['joints'],[1,2])
                self.assertEqual(metadata['rotation'],90)
                app.save_result()
                self.assertEqual(app.checklist.session['results']['R1']['history'],[])
                # A new component cannot reuse a previous image's ROI or result.
                app.select_component('U1')
                self.assertIsNone(app.roi);self.assertIsNone(app.result)
                self.assertEqual(app.joints.get(),'')
                self.assertEqual(app.expected.get(),'0')
            finally:app.close()

    def test_save_failure_does_not_mark_component_inspected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,board_store=Path(folder)/'boards',report_root=Path(folder)/'logs')
            try:
                boxes=SimpleNamespace(xyxy=TensorStub([[5,5,15,15],[25,25,35,35]]),cls=TensorStub([0,0]),conf=TensorStub([.9,.9]))
                app.model=SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(boxes=boxes)])
                app.model_path='test.pt';app.frame=app.display_frame=np.zeros((40,40,3),np.uint8)
                app.select_component('R1')
                with patch('app.save_report',side_effect=OSError('disk full')),patch.object(app,'error') as error:
                    app.inspect();self.wait_job(root,app)
                    error.assert_called_once()
                self.assertEqual(app.checklist.completed(),0)
                self.assertEqual(app.checklist.session['results'],{})
                app.save_result()
                self.assertEqual(app.checklist.completed(),1)
            finally:app.close()

    def test_bridge_coverage_requires_operator_and_stays_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk();root.withdraw()
            app=App(root,auto_model=False,monitor=False,board_store=Path(folder)/'boards',report_root=Path(folder)/'logs')
            try:
                boxes=SimpleNamespace(xyxy=TensorStub([[5,5,35,35]]),cls=TensorStub([1]),conf=TensorStub([.9]))
                app.model=SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(boxes=boxes)])
                app.model_path='test.pt';app.frame=app.display_frame=np.zeros((40,40,3),np.uint8)
                app.select_component('R1')
                app.inspect();self.wait_job(root,app)
                self.assertEqual(app.result[3],'FAIL')
                self.assertEqual(app.checklist.completed(),0)
                app.confirm_visible_joints()
                self.assertEqual(app.checklist.completed(),1)
                item=app.checklist.session['results']['R1']
                self.assertEqual(item['status'],'FAIL')
                self.assertEqual(item['confirmed_by'],'operator')
                self.assertTrue((app.report_path/'coverage_confirmation.json').is_file())
                app.save_result()
                self.assertEqual(app.checklist.completed(),1)
                self.assertEqual(len(item['history']),1)
            finally:app.close()


if __name__ == '__main__':
    unittest.main()
