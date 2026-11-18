import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from board import BoardChecklist,COMPONENTS
from board_export import export_board,inspected_component
from diode_cv import inspect_diode,combine_diode_result,FixedDiodeROI
from frame_registration import FeatureFrame,FrameRegistration
from inspection import detection_from_quad,write_image,read_image
from live_overlay import LiveOverlay
from pin_calibration import PinCalibration,save_calibration,read_calibrations,refine_pin_points
from scanner import corners,pad_boxes,transform,inspect_target,save_auto_report,observed_pin_boxes


def texture(height,width,seed=52):
    rng=np.random.default_rng(seed)
    image=rng.integers(0,255,(height,width,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),0)
    for y in range(25,height,70):
        for x in range(25,width,95):
            cv2.putText(image,f'{x}:{y}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
    return image


class PinCalibrationTests(unittest.TestCase):
    def test_partial_ssop_row_refines_pitch_without_numbering_an_absent_joint(self):
        frame=np.zeros((480,640,3),np.uint8)
        points=np.float32([[620,220],[569,220],[620,450]])
        calibration=PinCalibration('U4',frame,points,'unused')
        reg=calibration.locate(FeatureFrame(frame))
        # The first two calibrated pins are off screen; real pitch is 47 px.
        reg.matrix[0,2]+=56
        detections={}
        for pin in range(3,15):
            x=676-(pin-1)*47
            if pin!=9:detections[pin]=corners((x-14,125,x+14,231))
        proposals=list(detections.values())
        target=calibration.target(frame,reg,proposals)
        measured=observed_pin_boxes(target.pad_polygons,proposals)
        self.assertEqual(set(measured),set(detections))
        for pin,quad in measured.items():np.testing.assert_array_equal(quad,detections[pin])
        # Later only the far end is visible. Keep its physical number.
        target=calibration.target(frame,reg,[detections[14]])
        measured=observed_pin_boxes(target.pad_polygons,[detections[14]])
        self.assertEqual(set(measured),{14})

    def test_edge_pad_requires_a_fully_visible_measured_box(self):
        frame=np.zeros((480,640,3),np.uint8)
        calibration=PinCalibration('U4',frame,[[90,22],[140,22],[90,420]],'unused')
        reg=calibration.locate(FeatureFrame(frame))
        full=corners((77,6,103,95))
        clipped=corners((77,0,103,95))
        target=calibration.target(frame,reg)
        self.assertTrue(target is None or 1 not in target.joints)
        self.assertIn(1,calibration.target(frame,reg,[full]).joints)
        target=calibration.target(frame,reg,[clipped])
        self.assertTrue(target is None or 1 not in target.joints)

    def test_real_pin_numbers_accumulate_without_duplicates_across_partial_views(self):
        for component,total,points,origins,shape in [
            ('U1',14,[[150,50],[150,160],[470,50]],[(0,0),(0,150),(0,300),(0,450)],(1100,640)),
            ('U4',28,[[60,370],[160,370],[60,80]],[(0,0),(300,0),(600,0),(900,0)],(480,1700))]:
            with self.subTest(component=component),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);world=texture(*shape)
                anchor=world[:480,:640].copy()
                calibration=PinCalibration(component,anchor,points,root/'reference.png')
                ledger=BoardChecklist(root/'boards')
                observed=set()
                for ox,oy in [*origins,origins[-1]]:
                    frame=world[oy:oy+480,ox:ox+640].copy()
                    registration=calibration.locate(FeatureFrame(frame))
                    self.assertIsNotNone(registration,calibration.detail)
                    target=calibration.target(frame,registration)
                    self.assertIsNotNone(target)
                    self.assertLess(len(target.joints),total)
                    truth=np.array([[1.,0.,-ox],[0.,1.,-oy],[0.,0.,1.]])@calibration.board_to_anchor
                    detections=[]
                    for pin in target.joints:
                        quad=transform(corners(pad_boxes(component)[pin]),truth)
                        np.testing.assert_allclose(target.pad_polygons[pin].mean(axis=0),quad.mean(axis=0),atol=3)
                        detections.append(detection_from_quad(quad,0,.95,2,'good'))
                    model=SimpleNamespace(names={0:'bridge',1:'excess',2:'good',3:'missing'})
                    result,crop=inspect_target(model,{0:1,1:2,2:0,3:3},frame,target,registration,.25,'cpu',
                                               'best_inspec.pt','test',calibration,frame_detections=detections)
                    self.assertEqual(result[3],'PASS')
                    report=save_auto_report(root/'reports',result,crop)
                    ledger.record(component,result[3],result[4],report,len(target.joints),None,0,target.joints,
                                  joint_states=result[5]['joint_states'],replace_joint_states=True,
                                  confirmed_by='calibrated_ic_pin_reference')
                    observed.update(target.joints)
                self.assertEqual(observed,set(range(1,total+1)))
                entry=ledger.session['results'][component]
                self.assertEqual(entry['covered'],total)
                self.assertEqual(entry['status'],'PASS')
                self.assertEqual(len(entry['coverage']),total)
                self.assertTrue(inspected_component(component,entry))

    def test_tracking_loss_does_not_invent_a_pin_position(self):
        tracker=FrameRegistration(texture(480,640))
        self.assertIsNone(tracker.locate(FeatureFrame(np.zeros((480,640,3),np.uint8))))

    def test_calibration_roundtrip_and_invalid_seed_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            frame=texture(480,640)
            save_calibration(folder,'U1',frame,[[150,50],[150,160],[470,50]])
            pins,diode,errors=read_calibrations(folder)
            self.assertEqual(set(pins),{'U1'});self.assertIsNone(diode);self.assertEqual(errors,[])
            self.assertIsNotNone(pins['U1'].locate(FeatureFrame(frame)))
            with self.assertRaises(ValueError):PinCalibration('U1',frame,[[20,20],[20,20],[60,20]],'unused')

    def test_pitch_is_refined_using_multiple_visible_joint_centers(self):
        image=texture(480,640)
        correct=np.float32([[60,370],[160,370],[60,80]])
        calibration=PinCalibration('U4',image,correct,'unused')
        centers=[transform(corners(box),calibration.board_to_anchor).mean(axis=0) for box in pad_boxes('U4').values()]
        centers=[p for p in centers if 10<p[0]<630 and 10<p[1]<470]
        clicked=correct.copy();clicked[1,0]+=6
        fitted,matches=refine_pin_points('U4',clicked,centers)
        self.assertGreaterEqual(matches,6)
        np.testing.assert_allclose(fitted,correct,atol=.01)

    def test_latest_visible_pin_overwrites_fail_but_unknown_view_preserves_confirmed_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);proof=root/'proof';proof.mkdir();(proof/'result.json').write_text('{}')
            ledger=BoardChecklist(root/'boards')
            def record(pin,state):
                ledger.record('U1',state,'test',proof,1,None,0,[pin],joint_states={pin:state},
                              confirmed_by='calibrated_ic_pin_reference',replace_joint_states=True)
            record(1,'FAIL');record(2,'PASS');record(1,'PASS');record(2,'REVIEW')
            entry=ledger.session['results']['U1']
            self.assertEqual(entry['coverage']['1']['status'],'PASS')
            self.assertEqual(entry['coverage']['2']['status'],'PASS')
            self.assertEqual(entry['covered'],2);self.assertFalse(entry['complete'])

    def test_legacy_whole_ic_count_is_not_reused_as_physical_pin_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);proof=root/'proof';proof.mkdir();(proof/'result.json').write_text('{}')
            ledger=BoardChecklist(root/'boards')
            ledger.record('U1','PASS','old whole-view count',proof,14,None,0,list(range(1,15)),
                joint_states={p:'PASS' for p in range(1,15)},confirmed_by='latest_frame_component_count')
            ledger.record('D1','PASS','old solder-only result',proof,2,None,0,[1,2],confirmed_by='latest_frame_component_count')
            before=ledger.path.read_bytes()
            restored=BoardChecklist(root/'boards')
            entry=restored.session['results']['U1']
            self.assertEqual(entry['covered'],0);self.assertFalse(entry['complete'])
            self.assertEqual(entry['status'],'REVIEW');self.assertIn('legacy_unmapped_result',entry)
            self.assertEqual(restored.session['results']['D1']['status'],'REVIEW')
            self.assertEqual(before,ledger.path.read_bytes())


class DiodeTests(unittest.TestCase):
    def test_model_fixed_body_excludes_dark_pads_and_keeps_size_during_motion(self):
        image=self.image([340])
        model_quad=corners((30,30,470,170))
        pads=[corners((60,65,105,145)),corners((395,65,440,145))]
        roi=FixedDiodeROI()
        body=roi.quad(image,model_quad,pad_quads=pads,component='D1')
        self.assertGreater(body[:,0].min(),105)
        self.assertLess(body[:,0].max(),395)
        # Pad darkening must not change the body boundary or selected band.
        image[65:146,395:441]=0
        result=inspect_diode(image,model_quad,[1,0],body_quad=body)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['body_roi_source'],'model_fixed')
        self.assertLess(np.max(result['band_quad'],axis=0)[0],395)
        shift=np.float32([18,12])
        moved=roi.quad(image,model_quad+shift,pad_quads=[p+shift for p in pads],component='D1')
        np.testing.assert_allclose(moved,body+shift,atol=.001)
        jittered_quad=(model_quad-model_quad.mean(0))*1.08+model_quad.mean(0)+shift
        jittered=roi.quad(image,jittered_quad,
                         pad_quads=[p+shift for p in pads],component='D1')
        np.testing.assert_allclose(np.linalg.norm(np.roll(jittered,-1,axis=0)-jittered,axis=1),
                                   np.linalg.norm(np.roll(body,-1,axis=0)-body,axis=1),atol=.001)
        # Briefly missing pad proposals keep the last model-relative offset.
        fallback=roi.quad(image,jittered_quad,component='D1')
        np.testing.assert_allclose(fallback,jittered,atol=.001)

    def test_fixed_region_does_not_expand_when_glass_colour_changes(self):
        image=self.image([340]);q=corners((30,30,470,170))
        roi=FixedDiodeROI()
        pads=[corners((60,65,105,145)),corners((395,65,440,145))]
        body=roi.quad(image,q,pad_quads=pads)
        original=inspect_diode(image,q,[1,0],body_quad=body)
        image[65:146,60:106]=(0,0,0)
        # Warm metallic pixels fooled the previous colour-based ROI extension.
        image[65:146,395:441]=(20,35,150)
        changed=inspect_diode(image,q,[1,0],body_quad=body)
        np.testing.assert_allclose(changed['body_quad'],original['body_quad'])
        self.assertEqual(changed['status'],'PASS')
        self.assertEqual(inspect_diode(image,q,[-1,0],body_quad=body)['status'],'FAIL')

    def test_clipped_fixed_region_does_not_warp_black_padding_into_a_band(self):
        image=self.image([340]);q=corners((30,30,470,170))
        result=inspect_diode(image,q,[1,0],body_quad=corners((-20,70,400,140)))
        self.assertEqual(result['status'],'REVIEW')
        self.assertIsNone(result['band_quad'])
        result=inspect_diode(image,q,[1,0],body_quad=corners((10,70,400,140)))
        self.assertEqual(result['status'],'REVIEW')
        self.assertIsNone(result['band_quad'])

    def test_band_location_is_available_without_claiming_electrical_polarity(self):
        for x,side in [(140,'LEFT'),(340,'RIGHT')]:
            image=self.image([x]);quad=corners((30,30,470,170))
            observed=inspect_diode(image,quad,None)
            self.assertEqual(observed['status'],'REVIEW')
            self.assertEqual(observed['band_side_image'],side)
            self.assertIsNone(observed['expected_cathode_direction'])
            self.assertIsNotNone(observed['body_quad'])
            self.assertGreaterEqual(observed['dark_region_count'],1)
            self.assertFalse(observed['score_is_probability'])
            rotated=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            quad=np.float32([[image.shape[0]-1-y,x] for x,y in quad])
            self.assertEqual(inspect_diode(rotated,quad,None)['band_side_image'],'UP' if side=='LEFT' else 'DOWN')

    def image(self,bands=()):
        image=np.full((200,500,3),(115,45,20),np.uint8)
        cv2.rectangle(image,(60,65),(105,145),(210,210,210),-1)
        cv2.rectangle(image,(395,65),(440,145),(210,210,210),-1)
        cv2.rectangle(image,(100,70),(400,140),(25,30,190),-1)
        for x in bands:cv2.rectangle(image,(x,70),(x+20,140),(12,12,18),-1)
        return image

    def test_cathode_side_correct_reversed_unknown_and_glare(self):
        quad=corners((30,30,470,170))
        self.assertEqual(inspect_diode(self.image([340]),quad,[1,0])['status'],'PASS')
        self.assertEqual(inspect_diode(self.image([140]),quad,[1,0])['status'],'FAIL')
        self.assertEqual(inspect_diode(self.image([340]),quad,[-1,0])['status'],'FAIL')
        self.assertEqual(inspect_diode(self.image(),quad,[1,0])['status'],'REVIEW')
        selected=inspect_diode(self.image([140,340]),quad,[1,0])
        self.assertIn(selected['status'],('PASS','FAIL'))
        self.assertEqual(selected['decision_policy'],'largest_dark_region')
        self.assertEqual(inspect_diode(self.image([340]),quad,None)['status'],'REVIEW')
        glare=self.image([340]);cv2.rectangle(glare,(340,90),(360,100),(255,255,255),-1)
        self.assertEqual(inspect_diode(glare,quad,[1,0])['status'],'PASS')

    def test_polarity_failure_overrides_good_and_unknown_does_not_pass(self):
        frame=self.image([140]);base=(frame,frame.copy(),[],'PASS','good',dict(joints=[1,2],joint_states={1:'PASS',2:'PASS'}))
        failure=combine_diode_result(copy.deepcopy(base),dict(status='FAIL',reason='กลับด้าน',band_quad=None))
        self.assertEqual(failure[3],'FAIL')
        unknown=combine_diode_result(copy.deepcopy(base),dict(status='REVIEW',reason='ยังไม่ตั้งขั้ว',band_quad=None))
        self.assertEqual(unknown[3],'REVIEW');self.assertEqual(unknown[5]['joint_states'],{1:'REVIEW',2:'REVIEW'})

    def test_larger_dark_region_is_selected_over_the_small_opposite_band(self):
        image=self.image([140])
        cv2.rectangle(image,(340,70),(400,140),(2,2,2),-1)
        result=inspect_diode(image,corners((30,30,470,170)),[1,0])
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['band_side_image'],'RIGHT')
        self.assertGreater(result['band_width_px'],40)
        self.assertEqual(result['decision_policy'],'largest_dark_region')

    def test_faint_grey_band_is_decided_using_local_contrast(self):
        image=self.image()
        cv2.rectangle(image,(285,70),(305,140),(75,75,80),-1)
        cv2.rectangle(image,(350,70),(400,140),(2,2,2),-1)
        quad=corners((30,30,470,170))
        self.assertEqual(inspect_diode(image,quad,[1,0])['status'],'PASS')
        self.assertEqual(inspect_diode(image,quad,[-1,0])['status'],'FAIL')

    def test_dark_and_pale_bands_use_the_detected_side_directly(self):
        quad=corners((30,30,470,170))
        for level in (18,65,100,135,160):
            for x,state in ((140,'FAIL'),(340,'PASS')):
                frame=self.image()
                cv2.rectangle(frame,(x,70),(x+30,140),(level,level,level),-1)
                result=inspect_diode(frame,quad,[1,0])
                self.assertEqual(result['status'],state,(level,x,result))
                self.assertEqual(result['decision_policy'],'largest_dark_region')

    def test_largest_region_remains_selected_when_rotated_or_faded(self):
        for brightness in (20,100,160):
            frame=self.image([200])
            cv2.rectangle(frame,(330,70),(400,140),(brightness,brightness,brightness),-1)
            quad=corners((30,30,470,170))
            for turn in range(4):
                axis=np.float32([(1,0),(0,1),(-1,0),(0,-1)][turn])
                result=inspect_diode(frame,quad,axis)
                self.assertEqual(result['status'],'PASS',(brightness,turn,result))
                self.assertGreater(result['band_width_px'],50)
                self.assertEqual(inspect_diode(frame,quad,-axis)['status'],'FAIL')
                h=frame.shape[0]
                frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                quad=np.float32([[h-1-y,x] for x,y in quad])

    def test_band_remains_identified_through_glare_exposure_and_rotation(self):
        for gain in (.7,1.,1.2):
            frame=self.image([340])
            cv2.rectangle(frame,(340,91),(360,104),(255,255,255),-1)
            frame=np.clip(frame.astype(float)*gain,0,255).astype(np.uint8)
            quad=corners((30,30,470,170))
            for turn in range(4):
                axis=np.float32([(1,0),(0,1),(-1,0),(0,-1)][turn])
                self.assertEqual(inspect_diode(frame,quad,axis)['status'],'PASS')
                self.assertEqual(inspect_diode(frame,quad,-axis)['status'],'FAIL')
                h=frame.shape[0]
                frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                quad=np.float32([[h-1-y,x] for x,y in quad])


class OverlayTests(unittest.TestCase):
    def test_missing_hides_nested_boxes_markers_and_pin_overlays(self):
        from unittest.mock import patch
        frame=np.zeros((240,320,3),np.uint8)
        missing=detection_from_quad(corners((40,100,260,200)),3,.32)
        inside=detection_from_quad(corners((60,120,90,180)),0,.99)
        outside=detection_from_quad(corners((270,110,300,180)),0,.90)
        overlay=LiveOverlay()
        overlay.update(dict(raw=frame,locator_hits=[],detections=[inside,missing,outside],
                            marker_hits=[SimpleNamespace(quad=corners((60,120,90,180)),component='plus')],
                            pin_overlays=[dict(quad=corners((60,120,90,180)),component='U4',pin=1,state='PASS')]))
        with patch.object(overlay,'matrix',return_value=np.eye(3)),patch('live_overlay.draw_detection') as draw:
            result=overlay.draw(frame)
        self.assertEqual([call.args[1].class_id for call in draw.call_args_list],[3,0])
        np.testing.assert_array_equal(result,frame)

    def test_live_overlay_preserves_fixed_model_region(self):
        from unittest.mock import patch
        sample=DiodeTests();frame=sample.image([340])
        quad=corners((30,30,470,170))
        body=FixedDiodeROI().quad(frame,quad,pad_quads=[corners((60,65,105,145)),corners((395,65,440,145))])
        initial=inspect_diode(frame,quad,[1,0],body_quad=body)
        overlay=LiveOverlay()
        overlay.update(dict(raw=frame,locator_hits=[],detections=[],
                            diode_overlays=[dict(component='D1',quad=quad.tolist(),**initial)]))
        changed=sample.image([140])
        with patch.object(overlay,'matrix',return_value=np.eye(3)):overlay.draw(changed)
        item=overlay.diode_observations[0]
        self.assertEqual(item['status'],'FAIL')
        self.assertEqual(item['body_roi_source'],'model_fixed')
        np.testing.assert_allclose(item['fixed_body_quad'],body)

    def test_live_band_is_remeasured_on_current_frame_even_when_board_pass_is_locked(self):
        from unittest.mock import patch
        sample=DiodeTests();first=sample.image([340]);second=sample.image([140])
        quad=corners((30,30,470,170));initial=inspect_diode(first,quad,[1,0])
        overlay=LiveOverlay()
        payload=dict(raw=first,locator_hits=[],detections=[],diode_overlays=[dict(component='D1',quad=quad.tolist(),locked=True,**initial)])
        overlay.update(payload)
        with patch.object(overlay,'matrix',return_value=np.eye(3)):
            overlay.draw(second)
        observed=overlay.diode_observations[0]
        self.assertEqual(observed['band_side_image'],'LEFT')
        self.assertEqual(observed['status'],'FAIL')
        self.assertTrue(observed['locked'])
        self.assertEqual(payload['diode_overlays'][0]['status'],'PASS')
        with patch.object(overlay,'matrix',return_value=None):overlay.draw(second)
        self.assertEqual(overlay.diode_observations,[])

    def test_latest_boxes_follow_live_camera_translation(self):
        frame=texture(240,320)
        overlay=LiveOverlay()
        detection=detection_from_quad(corners((80,80,120,120)),0,.9)
        overlay.update(dict(raw=frame,locator_hits=[],detections=[detection]))
        moved=cv2.warpAffine(frame,np.float32([[1,0,12],[0,1,7]]),(320,240))
        matrix=overlay.matrix(moved)
        self.assertIsNotNone(matrix)
        np.testing.assert_allclose(matrix[:2,2],[12,7],atol=2)
        drawn=overlay.draw(moved)
        self.assertEqual(drawn.shape,moved.shape)
        np.testing.assert_array_equal(drawn[220,290],moved[220,290])


class BoardExportTests(unittest.TestCase):
    def test_old_16_pin_u1_result_requires_rescan_instead_of_relabeling_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            ledger.session['results']['U1']['total']=16
            ledger.save();before=ledger.path.read_bytes()
            restored=BoardChecklist(ledger.folder)
            item=restored.session['results']['U1']
            self.assertEqual(item['total'],14)
            self.assertEqual(item['status'],'REVIEW')
            self.assertFalse(item['coverage'])
            self.assertEqual(item['legacy_pin_count_result']['total'],16)
            self.assertEqual(before,ledger.path.read_bytes())

    def test_blank_board_number_cannot_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            for value in ('','   ','x\n123'):
                ledger.session['board_number']=value
                with self.assertRaises(ValueError):export_board(ledger.session,root/'completed')
            self.assertFalse((root/'completed').exists())

    def test_missing_reference_is_named_before_any_board_files_are_copied(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            missing=root/'missing'/'reference.png'
            ledger.session['calibration_profiles']={'U1':dict(kind='U1',image=str(missing))}
            before=copy.deepcopy(ledger.session)
            with self.assertRaisesRegex(ValueError,'U1') as caught:
                export_board(ledger.session,root/'completed')
            self.assertIn(str(missing),str(caught.exception))
            self.assertFalse((root/'completed').exists())
            self.assertEqual(before,ledger.session)

    def test_locked_physical_pass_keeps_its_evidence_but_failed_pin_can_recover(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=BoardChecklist(root/'boards');ledger.lock_pass=True
            proof=root/'proof';proof.mkdir();(proof/'result.json').write_text('{}')
            args=dict(name='U1',reason='test',report=proof,expected=2,roi=None,rotation=0,joints=[1,2],
                      confirmed_by='calibrated_ic_pin_reference',replace_joint_states=True)
            ledger.record(state='FAIL',joint_states={1:'PASS',2:'FAIL'},**args)
            first=copy.deepcopy(ledger.session['results']['U1']['coverage']['1'])
            ledger.record(state='REVIEW',joint_states={1:'FAIL',2:'PASS'},**args)
            coverage=ledger.session['results']['U1']['coverage']
            self.assertEqual(coverage['1'],first)
            self.assertEqual(coverage['2']['status'],'PASS')

    def completed(self,root):
        ledger=BoardChecklist(root/'boards');proof=root/'proof';proof.mkdir()
        ledger.session['board_number']='PCB-TEST-001'
        (proof/'result.json').write_text('{}')
        write_image(proof/'raw.png',texture(120,160))
        write_image(proof/'result.png',texture(120,160))
        for name,data in COMPONENTS.items():
            total=data['expected'];physical=name in ('U1','U4')
            ledger.record(name,'PASS','test',proof,total,None,0,list(range(1,total+1)),
                joint_states={p:'PASS' for p in range(1,total+1)},
                confirmed_by='calibrated_ic_pin_reference' if physical else 'latest_frame_component_count',
                diode_polarity='PASS' if name.startswith('D') else None)
        return ledger

    def test_complete_export_contains_board_summary_all_42_physical_pins_and_portable_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            reference=root/'reference.png'
            cv2.imwrite(str(reference),texture(100,100))
            ledger.session['calibration_profiles']={'U1':dict(kind='U1',points=[[1,1],[1,2],[2,1]],image=str(reference))}
            before=copy.deepcopy(ledger.session)
            output=export_board(ledger.session,root/'completed')
            saved=json.loads((output/'board.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['overall_status'],'PASS');self.assertEqual(saved['inspected_components'],62)
            self.assertEqual({p.stem for p in output.glob('*.png')},set(COMPONENTS))
            for name in COMPONENTS:self.assertGreater(read_image(output/(name+'.png')).size,0)
            with (output/'ic_pins.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),42)
            self.assertEqual({int(r['pin']) for r in rows if r['component']=='U4'},set(range(1,29)))
            for entry in saved['results'].values():self.assertTrue((output/entry['report']/'result.json').is_file())
            self.assertTrue((output/saved['calibration_profiles']['U1']['image']).is_file())
            self.assertEqual(before,ledger.session)

    def test_scanned_parts_with_incomplete_details_are_archived_as_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            for change in ('pin','diode','unmapped'):
                session=copy.deepcopy(ledger.session)
                if change=='pin':session['results']['U1']['coverage'].pop('14')
                elif change=='diode':session['results']['D1']['diode_polarity_state']='REVIEW'
                else:
                    for item in session['results']['U4']['coverage'].values():item['confirmed_by']='latest_frame_component_count'
                output=export_board(session,root/'completed')
                saved=json.loads((output/'board.json').read_text('utf-8'))
                self.assertEqual(saved['overall_status'],'REVIEW')
                if change=='pin':
                    with (output/'ic_pins.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
                    missing=next(r for r in rows if r['component']=='U1' and r['pin']=='14')
                    self.assertEqual(missing['status'],'PENDING')
                    self.assertEqual(missing['confirmed'],'False')

    def test_unseen_component_blocks_export_even_when_other_parts_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            ledger.session['results'].pop('R1')
            with self.assertRaisesRegex(ValueError,'R1'):export_board(ledger.session,root/'completed')

    def test_missing_detection_image_does_not_publish_or_clear_the_board(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root)
            (root/'proof'/'result.png').unlink()
            before=copy.deepcopy(ledger.session)
            with self.assertRaises((OSError,ValueError)):export_board(ledger.session,root/'completed')
            self.assertFalse(list((root/'completed').glob('board_*/board.json')))
            self.assertEqual(ledger.session,before)

    def test_complete_failed_board_is_exported_as_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ledger=self.completed(root);proof=root/'proof'
            ledger.record('R1','FAIL','known defect in full component',proof,2,None,0,[1,2],
                joint_states={1:'REVIEW',2:'REVIEW'},confirmed_by='latest_frame_component_count',replace_previous=True)
            output=export_board(ledger.session,root/'completed')
            saved=json.loads((output/'board.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['overall_status'],'FAIL')
            self.assertEqual(saved['results']['R1']['covered'],0)


if __name__=='__main__':unittest.main()
