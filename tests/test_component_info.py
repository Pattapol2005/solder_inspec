from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
from app import App
from component_info import component_text,failure_reason
import json


class ComponentInfoTests(unittest.TestCase):
    def test_failure_reason_uses_saved_quality_classes_and_pin_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'result.json').write_text(json.dumps(dict(detections=[
                dict(class_id=1,native_class_id=0,native_class_name='bridge'),
                dict(class_id=0,native_class_id=2,native_class_name='good')])))
            text=failure_reason(dict(report=str(path),status='FAIL',reason='พบข้อบกพร่อง',
                                     coverage={'2':dict(status='FAIL')}))
            self.assertIn('bridge',text)
            self.assertIn('2',text)
            self.assertNotIn('excess',text)

    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.root=tk.Tk()
        self.app=App(self.root,auto_model=False,monitor=False,auto_scan=True,
                     board_store=Path(self.folder.name)/'boards')

    def tearDown(self):
        self.app.close();self.folder.cleanup()

    def test_selected_component_keeps_its_info_and_evidence_as_live_target_changes(self):
        self.app.checklist.session['results']['R1']={'status':'FAIL','reason':'bridge','covered':1,'coverage':{}}
        with patch.object(self.app,'show_evidence') as show:
            self.app.select_component_info('R1')
            show.assert_called_once_with('R1')
            self.assertNotIn('ข้อมูลอุปกรณ์',[self.app.results_notebook.tab(tab,'text') for tab in self.app.results_notebook.tabs()])
            show.reset_mock()
            self.app.component.set('R2');self.app.refresh_checklist()
            self.app.show_component_evidence()
            show.assert_called_once_with('R1')
        self.assertTrue(self.app.auto_mode.get())

    def test_uninspected_component_does_not_open_another_components_evidence(self):
        self.app.select_component_info('R3')
        with patch.object(self.app,'show_evidence') as show:
            self.app.show_component_evidence();show.assert_not_called()
        self.assertIn('ยังไม่มีภาพหลักฐาน',self.app.info.get())
        self.app.begin_new_board()
        self.assertIsNone(self.app.info_component)

    def test_diode_and_physical_pins_show_recorded_states_without_inventing_results(self):
        text=component_text('D1',{'status':'FAIL','diode_polarity_state':'FAIL','coverage':{}})
        self.assertIn('ทิศทางไดโอด: กลับด้าน',text)
        text=component_text('U1',{'coverage':{'1':{'status':'PASS','confirmed':True}}})
        self.assertIn('1: ผ่าน · ยืนยันแล้ว',text)
        self.assertIn('14: ยังไม่ตรวจ · ยังไม่ยืนยัน',text)
        self.assertNotIn('15:',text)
