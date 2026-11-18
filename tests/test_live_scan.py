from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from app import App
from inspection import quality_detections,write_image
from preferences import load_preferences, save_preferences, DEFAULTS
from trained_locator import TrainedLocator
from test_app import TensorStub


NAMES={0:'bridge',1:'excess',2:'good',3:'missing'}


def obb_result(polygons, classes, confidence, names=None):
    return SimpleNamespace(names=names,boxes=None,obb=SimpleNamespace(
        xyxyxyxy=TensorStub(polygons),cls=TensorStub(classes),conf=TensorStub(confidence)))


class PreferenceTests(unittest.TestCase):
    def test_atomic_roundtrip_preserves_other_settings_and_rejects_invalid_confidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            self.assertEqual(load_preferences(path),(DEFAULTS,None))
            path.write_text('{"camera_index":1}')
            save_preferences(path,dict(inspection_confidence=.37,locator_confidence=.72))
            self.assertEqual(json.loads(path.read_text())['camera_index'],1)
            self.assertEqual(load_preferences(path)[0]['inspection_confidence'],.37)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            before=path.read_bytes()
            for value in (float('nan'),float('inf'),0,1.1):
                with self.assertRaises(ValueError):
                    save_preferences(path,dict(inspection_confidence=value,locator_confidence=.6))
            self.assertEqual(before,path.read_bytes())

    def test_result_class_names_override_a_stale_application_mapping(self):
        q=[[[10,10],[30,10],[30,30],[10,30]]]
        for native_id,name in NAMES.items():
            rows=quality_detections(obb_result(q,[native_id],[.9],NAMES),dict(enumerate(range(4))),.25)
            self.assertEqual(rows[0].native_class_id,native_id)
            self.assertEqual(rows[0].native_class_name,name)
            self.assertEqual(rows[0].class_id,{'good':0,'bridge':1,'excess':2,'missing':3}[name])


class LiveScanTests(unittest.TestCase):
    def setUp(self):
        self.acquire_patch=patch.object(TrainedLocator,'ACQUIRE_SECONDS',0)
        self.acquire_patch.start()
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.root=tk.Tk();self.root.withdraw()
        self.app=App(self.root,auto_model=False,monitor=False,auto_scan=True,
                     board_store=self.folder/'boards',report_root=self.folder/'reports',
                     settings_file=self.folder/'settings.json')
        self.app.locator_file=self.folder/'best_locator.pt'
        self.app.locator_file.write_bytes(b'test-only')
        self.app.locator_class_map={0:'R22'}
        quad=[[20,20],[180,20],[180,100],[20,100]]
        self.locator_result=obb_result([quad],[0],[.99])
        self.app.locator_model=SimpleNamespace(predict=lambda **kw:[self.locator_result])
        good=[[[25,40],[55,40],[55,80],[25,80]],[[145,40],[175,40],[175,80],[145,80]]]
        self.quality_result=obb_result(good,[2,2],[.95,.96],NAMES)
        self.quality_calls=[]
        def predict(**kwargs):
            self.quality_calls.append(kwargs)
            return [self.quality_result]
        self.app.model=SimpleNamespace(names=NAMES,predict=predict)
        self.app.class_map={0:1,1:2,2:0,3:3}
        self.app.model_path='best_inspec.pt'
        self.frame=np.zeros((200,300,3),np.uint8)
        self.app.frame=self.app.display_frame=self.frame

    def tearDown(self):
        if self.root is not None:self.app.close()
        self.acquire_patch.stop()
        self.temp.cleanup()

    def pump(self,condition,timeout=4):
        deadline=time.monotonic()+timeout
        while not condition() and time.monotonic()<deadline:
            self.root.update();time.sleep(.005)
        self.assertTrue(condition(),self.app.scan_status.get())

    def inspect(self):
        self.app.scan_snapshot(self.frame.copy())
        self.pump(lambda:not self.app.busy)

    def test_default_locator_appears_disappears_and_reappears_without_delay(self):
        self.inspect()
        self.assertEqual([h.component for h in self.app.position_hits],['R22'])
        self.assertIn('R22',self.app.checklist.session['results'])
        detected=self.locator_result
        self.locator_result=obb_result([],[],[])
        self.inspect()
        self.assertEqual(self.app.position_hits,[])
        self.assertEqual(self.app.live_overlay.payload['locator_hits'],[])
        self.locator_result=detected
        self.inspect()
        self.assertEqual([h.component for h in self.app.position_hits],['R22'])

    def test_moving_frames_run_immediately_and_only_latest_unprocessed_frame_is_used(self):
        entered=threading.Event();release=threading.Event();markers=[]
        original=self.app.model.predict
        def slow(**kwargs):
            markers.append(int(kwargs['source'][0,0,0]))
            if len(markers)==1:entered.set();release.wait(3)
            return original(**kwargs)
        self.app.model.predict=slow
        self.app.frozen=False
        with patch.object(self.app.capture_gate,'observe',side_effect=AssertionError('motion gate must not run')):
            self.app.maybe_scan(np.full_like(self.frame,1),1)
            self.pump(entered.is_set)
            self.app.maybe_scan(np.full_like(self.frame,2),2)
            self.app.maybe_scan(np.full_like(self.frame,3),3)
            release.set();self.pump(lambda:not self.app.busy)
            self.app.maybe_scan(np.full_like(self.frame,3),3)
            self.pump(lambda:not self.app.busy)
        self.assertEqual(markers,[1,3])
        self.assertEqual(self.quality_calls[-1]['source'].shape,self.frame.shape)
        self.assertEqual(self.quality_calls[-1]['iou'],.45)
        self.assertNotIn('เลื่อนบอร์ด',self.app.scan_status.get())

    def test_pass_is_locked_until_reset_and_failed_component_can_recover(self):
        self.inspect()
        self.assertEqual(self.app.checklist.session['results']['R22']['status'],'PASS')
        self.quality_result=obb_result([[[25,40],[175,40],[175,80],[25,80]]],[0],[.95],NAMES)
        self.inspect()
        self.assertEqual(self.app.checklist.session['results']['R22']['status'],'PASS')
        self.assertEqual(len(list((self.folder/'reports').glob('*/result.json'))),1)
        self.app.reset_board_inspection()
        self.inspect()
        self.assertEqual(self.app.checklist.session['results']['R22']['status'],'FAIL')
        self.assertEqual(self.app.checklist.session['results']['R22']['covered'],0)
        self.quality_result=obb_result([[[25,40],[55,40],[55,80],[25,80]],
                                      [[145,40],[175,40],[175,80],[145,80]]],[2,2],[.95,.96],NAMES)
        self.inspect();self.inspect()
        entry=self.app.checklist.session['results']['R22']
        self.assertEqual(entry['status'],'PASS')
        self.assertEqual(entry['covered'],2)
        self.assertEqual(len(entry['history']),1)
        self.assertEqual(len(list((self.folder/'reports').glob('*/result.json'))),3)
        self.assertEqual(len(self.quality_calls),5)

    def test_ambiguous_locator_still_displays_all_model_boxes_and_runs_quality(self):
        self.locator_result.obb.cls=TensorStub([0,0])
        q=self.locator_result.obb.xyxyxyxy.data[0]
        self.locator_result.obb.xyxyxyxy=TensorStub([q,q])
        self.locator_result.obb.conf=TensorStub([.98,.97])
        self.inspect()
        self.assertEqual(len(self.app.position_hits),2)
        self.assertEqual(len(self.quality_calls),1)
        self.assertGreater(np.count_nonzero(self.app.display_frame),0)
        self.assertEqual(len(self.app.table.get_children()),2)
        self.assertEqual(self.app.checklist.session['results'],{})
        self.assertIsNotNone(self.app.live_preview)

    def test_confidence_is_passed_to_both_models_and_remembered_on_reopen(self):
        self.app.threshold.set('0.37');self.app.locator_threshold.set('0.72')
        self.inspect()
        self.assertEqual(self.quality_calls[-1]['conf'],.37)
        self.assertEqual(self.app.locator.threshold,.72)
        self.app.close();self.root=None
        root=tk.Tk();root.withdraw()
        app=App(root,auto_model=False,monitor=False,settings_file=self.folder/'settings.json',
                board_store=self.folder/'boards')
        try:
            self.assertAlmostEqual(float(app.threshold.get()),.37)
            self.assertAlmostEqual(float(app.locator_threshold.get()),.72)
        finally:app.close()

    def test_confidence_change_discards_inflight_results(self):
        entered=threading.Event();release=threading.Event()
        original=self.app.model.predict
        def slow(**kwargs):
            entered.set();release.wait(3)
            return original(**kwargs)
        self.app.model.predict=slow
        self.app.scan_snapshot(self.frame.copy());self.pump(entered.is_set)
        self.app.threshold.set('0.45')
        release.set();self.pump(lambda:not self.app.busy)
        self.assertEqual(self.app.checklist.session['results'],{})
        self.assertIsNone(self.app.live_preview)

    def test_camera_video_keeps_advancing_while_ai_is_busy(self):
        entered=threading.Event();release=threading.Event();original=self.app.model.predict
        def slow(**kwargs):
            entered.set();release.wait(3);return original(**kwargs)
        self.app.model.predict=slow
        frame=self.frame
        class CameraStub:
            error=None;sequence=0
            def latest_sample(self):
                self.sequence+=1
                return self.sequence,np.full_like(frame,self.sequence%180)
            def stop(self):pass
        self.app.camera=CameraStub();self.app.frozen=False
        self.pump(entered.is_set)
        first=int(self.app.frame[0,0,0]);deadline=time.monotonic()+.16
        while time.monotonic()<deadline:self.root.update();time.sleep(.005)
        self.assertTrue(self.app.busy)
        self.assertGreater(int(self.app.frame[0,0,0]),first)
        self.assertEqual(int(self.app.display_frame[0,0,0]),int(self.app.frame[0,0,0]))
        self.app.frozen=True;self.app.cancel_scan();release.set()
        self.pump(lambda:not self.app.busy)

    def test_partial_ic_uses_physical_calibration_even_without_a_locator_hit(self):
        from pin_calibration import PinCalibration
        from frame_registration import FeatureFrame
        from test_pin_diode_workflow import texture
        frame=texture(200,300)
        calibration=PinCalibration('U1',frame,[[60,40],[60,100],[230,40]],self.folder/'pin_reference.png')
        target=calibration.target(frame,calibration.locate(FeatureFrame(frame)))
        self.assertLess(len(target.joints),14)
        self.app.pin_calibrations={'U1':calibration}
        self.locator_result=obb_result([],[],[])
        self.quality_result=obb_result([target.pad_polygons[p].tolist() for p in target.joints],
                                      [2]*len(target.joints),[.95]*len(target.joints),NAMES)
        self.frame=frame;self.app.frame=self.app.display_frame=frame
        self.inspect()
        entry=self.app.checklist.session['results']['U1']
        self.assertEqual(entry['covered'],len(target.joints))
        self.assertFalse(entry['complete'])
        self.assertEqual(set(map(int,entry['coverage'])),set(target.joints))
        self.assertEqual(entry['confirmed_by'],'calibrated_ic_pin_reference')
        self.assertIn('/14',self.app.pin_progress.get())
        self.assertEqual(str(self.app.board_save_btn['state']),'disabled')
        from evidence import read_evidence
        pin=target.joints[0]
        self.assertTrue((self.app.report_path/f'pin_{pin:02d}.png').is_file())
        self.assertLess(read_evidence(self.app.report_path,pin).shape[1],frame.shape[1])
        self.assertTrue(self.app.live_overlay.payload['pin_overlays'])

    def test_partial_ic_without_anchor_shows_solder_proposals_but_does_not_invent_pin_numbers(self):
        self.app.locator_class_map={0:'U4'}
        self.locator_result=obb_result([[[0,20],[180,20],[180,100],[0,100]]],[0],[.99])
        self.inspect()
        self.assertEqual(self.app.ic_component.get(),'U4')
        self.assertEqual(len(self.app.ic_observations['U4']),2)
        self.assertTrue(self.app.pin_table.exists('candidate:0'))
        self.assertIn('ยังไม่ผูกเลขขา',self.app.pin_table.item('candidate:0','values'))
        self.assertNotIn('U4',self.app.checklist.session['results'])
        self.assertIn('ตั้งขา 1',self.app.pin_progress.get())

    def test_one_missing_ic_bank_marks_the_whole_part_without_pin_calibration(self):
        from board_export import inspected_component
        from ic_evidence import bank_reports
        for name in ('U1','U4'):
            with self.subTest(component=name):
                self.app.begin_new_board();self.app.locator=None
                self.app.locator_class_map={0:name}
                # Only part of the IC is visible, with its locator at the edge.
                self.locator_result=obb_result([[[0,0],[299,0],[299,199],[0,199]]],[0],[.99])
                self.quality_result=obb_result([
                    [[15,140],[285,140],[285,190],[15,190]],
                    [[20,25],[40,25],[40,60],[20,60]]],[3,2],[.93,.95],NAMES)
                self.inspect()
                entry=self.app.checklist.session['results'][name]
                self.assertEqual(entry['status'],'FAIL')
                self.assertTrue(entry['component_missing'])
                self.assertEqual(entry['coverage'],{})
                self.assertEqual(entry['covered'],0)
                self.assertTrue(inspected_component(name,entry))
                self.assertEqual(self.app.checklist.completed(),1)
                self.assertIn('MISSING',self.app.pin_progress.get())
                self.assertEqual(bank_reports(name,entry),[[],[]])
                proof=json.loads((self.app.report_path/'result.json').read_text('utf-8'))
                self.assertTrue(proof['component_missing'])
                self.assertFalse(proof['physical_pin_numbers'])

    def test_partial_ic_bridge_fails_immediately_and_identifies_each_problem_region(self):
        for name in ('U1','U4'):
            with self.subTest(component=name):
                self.app.begin_new_board();self.app.locator=None
                self.app.locator_class_map={0:name}
                self.locator_result=obb_result([[[0,0],[299,0],[299,199],[0,199]]],[0],[.99])
                self.quality_result=obb_result([
                    [[10,15],[85,15],[85,65],[10,65]],
                    [[10,85],[85,85],[85,135],[10,135]],
                    [[260,15],[285,15],[285,60],[260,60]]],[0,0,2],[.92,.91,.95],NAMES)
                self.inspect()
                entry=self.app.checklist.session['results'][name]
                self.assertEqual(entry['status'],'FAIL')
                self.assertNotIn('component_missing',entry)
                self.assertIn('bridge',entry['reason'])
                self.assertIn('ซ้าย',entry['reason'])
                self.assertIn('ช่วงบน',entry['reason'])
                self.assertIn('ช่วงกลาง',entry['reason'])
                self.assertIn('FAIL',self.app.pin_progress.get())
                proof=json.loads((self.app.report_path/'result.json').read_text('utf-8'))
                self.assertEqual(len(proof['defect_regions']),2)
                self.assertEqual(proof['defect_regions'][0]['class_name'],'bridge')
                self.assertEqual(proof['defect_regions'][0]['pins'],[])

    def test_missing_on_a_neighbour_does_not_mark_the_ic_missing(self):
        self.app.locator_class_map={0:'U4',1:'Q3'}
        self.locator_result=obb_result([
            [[5,5],[145,5],[145,195],[5,195]],
            [[155,20],[295,20],[295,180],[155,180]]],[0,1],[.99,.99])
        self.quality_result=obb_result([[[170,40],[280,40],[280,140],[170,140]]],[3],[.95],NAMES)
        self.inspect()
        self.assertNotIn('U4',self.app.checklist.session['results'])

    def test_nested_good_is_hidden_from_live_output_and_saved_evidence(self):
        self.quality_result=obb_result([
            [[25,25],[175,25],[175,95],[25,95]],
            [[30,35],[60,35],[60,85],[30,85]],
            [[230,35],[270,35],[270,85],[230,85]]],[3,2,2],[.32,.99,.99],NAMES)
        self.inspect()
        payload=self.app.live_overlay.payload
        self.assertEqual([d.class_id for d in payload['detections']],[3,0])
        proof=json.loads((self.app.report_path/'result.json').read_text('utf-8'))
        self.assertEqual([d['class_id'] for d in proof['detections']],[3])
        self.assertEqual(self.app.checklist.session['results']['R22']['status'],'FAIL')

    def test_overlapping_non_ic_label_does_not_hide_a_missing_ic_bank(self):
        self.app.locator_class_map={0:'U4',1:'Q3'}
        q=[[10,10],[290,10],[290,190],[10,190]]
        self.locator_result=obb_result([q,q],[0,1],[.99,.80])
        self.quality_result=obb_result([[[15,140],[285,140],[285,180],[15,180]]],[3],[.95],NAMES)
        self.inspect()
        self.assertTrue(self.app.checklist.session['results']['U4']['component_missing'])

    def test_completed_board_can_be_saved_with_evidence(self):
        self.app.board_number.set('PCB-001')
        from board import COMPONENTS
        proof=self.folder/'proof';proof.mkdir();(proof/'result.json').write_text('{}')
        write_image(proof/'result.png',np.zeros((120,160,3),np.uint8))
        for name,data in COMPONENTS.items():
            total=data['expected']
            self.app.checklist.record(name,'PASS','test',proof,total,None,0,list(range(1,total+1)),
                joint_states={p:'PASS' for p in range(1,total+1)},
                confirmed_by='calibrated_ic_pin_reference' if name in ('U1','U4') else 'latest_frame_component_count',
                diode_polarity='PASS' if name.startswith('D') else None)
        self.app.refresh_checklist()
        self.assertEqual(str(self.app.board_save_btn['state']),'normal')
        self.app.board_number.set('')
        self.assertEqual(str(self.app.board_save_btn['state']),'disabled')
        self.app.save_complete_board()
        self.assertIsNone(self.app.last_saved_board)
        self.app.board_number.set('PCB-001')
        previous_id=self.app.checklist.session['board_id']
        failed=self.app.checklist.session['results']['U4']
        failed.update(status='FAIL',reason='bridge on pin 1',complete=False,covered=1)
        failed['coverage']={'1':dict(status='FAIL',report=str(proof),confirmed=True,confirmed_by='calibrated_ic_pin_reference')}
        self.app.refresh_checklist()
        self.assertEqual(str(self.app.board_save_btn['state']),'normal')
        with patch('workflow.export_board',side_effect=OSError('disk unavailable')):
            self.app.save_complete_board()
            self.pump(lambda:not self.app.board_saving)
        self.assertEqual(self.app.checklist.session['board_id'],previous_id)
        self.assertEqual(self.app.board_number.get(),'PCB-001')
        self.assertEqual(len(self.app.checklist.session['results']),62)
        self.app.save_complete_board()
        self.pump(lambda:not self.app.board_saving)
        self.assertTrue((self.app.last_saved_board/'board.json').is_file())
        self.assertTrue((self.app.last_saved_board/'ic_pins.csv').is_file())
        saved=json.loads((self.app.last_saved_board/'board.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['board_number'],'PCB-001')
        self.assertEqual(saved['overall_status'],'FAIL')
        self.assertIn('bridge',saved['results']['U4']['failure_reason'])
        self.assertEqual(len(saved['results']),len(COMPONENTS))
        self.assertNotEqual(self.app.checklist.session['board_id'],previous_id)
        self.assertEqual(self.app.board_number.get(),'')
        self.assertEqual(self.app.checklist.session['results'],{})
        self.assertFalse(self.app.diode_live)
        self.assertFalse(self.app.live_evidence_cache)
        self.assertIsNone(self.app.last_saved_signature)
        self.assertEqual(str(self.app.board_save_btn['state']),'disabled')

    def test_reset_retains_board_number_and_new_board_clears_it_and_rejects_old_job(self):
        self.app.board_number.set('PCB-002')
        self.inspect();old_id=self.app.checklist.session['board_id']
        self.app.diode_live={'D1':dict(status='FAIL',reason='กลับด้าน')}
        generation=self.app.scan_generation
        self.app.reset_board_inspection()
        self.assertGreater(self.app.scan_generation,generation)
        self.assertEqual(self.app.checklist.session['board_id'],old_id)
        self.assertEqual(self.app.checklist.session['board_number'],'PCB-002')
        self.assertFalse(self.app.checklist.session['results'])
        self.assertEqual(self.app.checklist.session['inspection_resets'][0]['results']['R22']['status'],'PASS')
        self.assertFalse(self.app.diode_live)
        self.inspect();self.app.begin_new_board()
        self.assertNotEqual(self.app.checklist.session['board_id'],old_id)
        self.assertEqual(self.app.board_number.get(),'')
        self.assertFalse(self.app.checklist.session['results'])
        archived=json.loads((self.app.checklist.folder/(old_id+'.json')).read_text(encoding='utf-8'))
        self.assertEqual(archived['board_number'],'PCB-002')

    def test_ambiguous_diode_has_visible_reason_instead_of_disappearing(self):
        self.app.locator_class_map={0:'D1'}
        q=self.locator_result.obb.xyxyxyxy.data[0]
        self.locator_result=obb_result([q,q],[0,0],[.99,.98])
        self.inspect()
        self.assertIn('ยังไม่ตั้งขั้ว',self.app.diode_summary.get())
        self.assertIn('D1: ยังอ่านด้านไม่ได้',self.app.diode_summary.get())
        self.assertFalse(self.app.checklist.session['results'])

    def test_diode_solder_good_cannot_pass_until_cv_direction_is_confirmed(self):
        from diode_cv import DiodeDirection
        from test_pin_diode_workflow import DiodeTests
        from board_export import inspected_component
        frame=DiodeTests().image([340])
        self.locator_result=obb_result([[[30,30],[470,30],[470,170],[30,170]]],[0],[.99])
        self.app.locator_class_map={0:'D1'}
        self.quality_result=obb_result([[[60,65],[105,65],[105,145],[60,145]],
                                      [[395,65],[440,65],[440,145],[395,145]]],[2,2],[.95,.96],NAMES)
        self.frame=frame;self.app.frame=self.app.display_frame=frame
        self.inspect()
        entry=self.app.checklist.session['results']['D1']
        self.assertEqual(entry['status'],'REVIEW')
        self.assertEqual(entry['diode_polarity_state'],'REVIEW')
        self.app.diode_direction=DiodeDirection(frame,[[80,105],[420,105]],self.folder/'diode.png')
        self.inspect()
        entry=self.app.checklist.session['results']['D1']
        self.assertEqual(entry['status'],'PASS')
        self.assertEqual(entry['diode_polarity_state'],'PASS')
        self.assertIn('D1: ถูกด้าน',self.app.diode_summary.get())
        self.assertTrue(inspected_component('D1',entry))
        proof=json.loads((self.app.report_path/'result.json').read_text(encoding='utf-8'))
        self.assertEqual(proof['diode_polarity']['status'],'PASS')
        previous=self.app.checklist.session['results']['D1'].copy()
        self.frame=DiodeTests().image([140]);self.app.frame=self.app.display_frame=self.frame
        self.inspect()
        self.assertEqual(self.app.checklist.session['results']['D1'],previous)
        self.assertEqual(self.app.live_overlay.payload['diode_overlays'][0]['band_side_image'],'LEFT')

    def test_missing_diode_remains_failed_without_requiring_a_nonexistent_band(self):
        from board_export import inspected_component
        self.app.locator_class_map={0:'D1'}
        self.quality_result=obb_result([[[25,25],[175,25],[175,95],[25,95]],
                                      [[145,40],[175,40],[175,80],[145,80]]],[3,2],[.95,.95],NAMES)
        with (
            patch('diode_cv.inspect_diode',side_effect=AssertionError('missing must skip CV')),
            patch('live_scan.inspect_diode',side_effect=AssertionError('missing must skip preview CV')),
            patch('live_overlay.inspect_diode',side_effect=AssertionError('missing must skip live CV')),
        ):
            self.inspect()
            self.app.live_overlay.draw(self.frame.copy())
        entry=self.app.checklist.session['results']['D1']
        self.assertEqual(entry['status'],'FAIL')
        self.assertEqual(entry['diode_polarity_state'],'NOT_APPLICABLE')
        self.assertTrue(entry['component_missing'])
        self.assertEqual(self.app.live_overlay.diode_observations[0]['status'],'NOT_APPLICABLE')
        self.assertTrue(inspected_component('D1',entry))

    def test_new_pin_calibration_preserves_old_results_as_history_but_requires_a_rescan(self):
        from test_pin_diode_workflow import texture
        proof=self.folder/'proof';proof.mkdir();(proof/'result.json').write_text('{}')
        self.app.checklist.record('U1','PASS','old calibration',proof,14,None,0,list(range(1,15)),
            joint_states={p:'PASS' for p in range(1,15)},confirmed_by='calibrated_ic_pin_reference')
        self.app.install_calibration('U1',texture(200,300),[[60,40],[60,100],[230,40]])
        self.assertNotIn('U1',self.app.checklist.session['results'])
        self.assertIn('U1',self.app.pin_calibrations)
        self.assertEqual(self.app.checklist.session['calibration_resets'][-1]['previous_results'][0]['result']['covered'],14)
        self.assertTrue((self.app.calibration_root/'U1'/'active.json').is_file())


if __name__=='__main__':unittest.main()
