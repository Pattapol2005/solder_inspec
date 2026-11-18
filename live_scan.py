"""One current frame per job: no motion gate, no queue of obsolete camera frames."""
import cv2
import numpy as np
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from time import monotonic as missing_clock

from inspection import quality_detections, draw_detection, draw_pin_number,detection_polygon,hidden_by_missing,draw_pending_missing
from scanner import inspect_target, observed_pin_boxes, save_auto_report,ScanTarget,Registration
from board import COMPONENTS
from frame_registration import FeatureFrame
from diode_cv import assess_diode,combine_diode_result,inspect_diode,draw_diode,FixedDiodeROI
from runtime import inference_options
from locator_markers import diode_marker_reference,assigned_markers
from pin_recovery import recover_from_head

_reference_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='board-reference')
IC_MISSING_CONFIRM_SECONDS=.45


def _missing_box_key(detection):
    return np.float32(detection.polygon).tobytes()


def defective_ic_targets(frame,hits,detections):
    found={}
    for detection in detections:
        if detection.class_id==0:continue
        center=tuple(map(float,np.float32(detection.polygon).mean(axis=0)))
        owners=[h for h in hits if h.component in ('U1','U4') and
                cv2.pointPolygonTest(np.float32(h.quad),center,False)>=0]
        if len(owners)!=1:continue
        hit=owners[0]
        if sum(h.component==hit.component for h in hits)!=1:continue
        found[hit.component]=hit
    targets=[]
    h,w=frame.shape[:2]
    for name,hit in found.items():
        x1,y1=np.maximum(0,np.floor(hit.quad.min(axis=0))).astype(int)
        x2,y2=np.minimum([w,h],np.ceil(hit.quad.max(axis=0))).astype(int)
        if x2<=x1 or y2<=y1:continue
        targets.append(ScanTarget(name,list(range(1,COMPONENTS[name]['expected']+1)),hit.quad,
                       frame[y1:y2,x1:x2].copy(),np.array([[1.,0.,x1],[0.,1.,y1],[0.,0.,1.]]),{}))
    return targets


def _calibrated_views(frame,pins,diode,cancel):
    if cancel.is_set():return [],[],None,None
    features=FeatureFrame(frame)
    jobs=[];notes=[]
    for name,calibration in (pins or {}).items():
        if cancel.is_set():return [],[],None,None
        reg=calibration.locate(features)
        if reg is not None:jobs.append((reg,calibration))
        notes.append(name+': '+calibration.detail)
    axis=diode.axis(features) if diode is not None and not cancel.is_set() else None
    return jobs,notes,axis,features


def inspect_live_frame(frame, locator, model, class_map, threshold, locator_threshold,
                       device, model_path, board_id, ledger, evidence_cache, report_root, cancel,
                       pin_calibrations=None,diode_direction=None,ic_missing_since=None):
    observed_at=missing_clock()
    missing_since=dict(ic_missing_since or {})
    seen_missing=set();pending_missing={};pending_missing_boxes={}
    options=inference_options(device)
    locked={name for name,item in ledger.items() if item.get('status')=='PASS' and item.get('complete') and
            (not name.startswith('D') or item.get('diode_polarity_state')=='PASS')}
    pin_calibrations=dict(pin_calibrations or {})
    reference_future=(_reference_executor.submit(_calibrated_views,frame,pin_calibrations,diode_direction,cancel)
                      if pin_calibrations or diode_direction is not None else None)
    locator.threshold=locator_threshold
    registration=locator.locate(frame)
    markers=list(getattr(locator,'markers',[]))
    if cancel.is_set():
        if reference_future is not None:reference_future.cancel()
        return None
    detections=[]
    if model is not None:
        prediction=model.predict(source=frame,conf=threshold,iou=.45,imgsz=1024,device=device,verbose=False,**options)[0]
        detections=quality_detections(prediction,class_map,threshold,getattr(model,'names',None))
    if cancel.is_set():
        if reference_future is not None:reference_future.cancel()
        return None
    annotated=frame.copy()
    targets=locator.targets(frame,registration,ledger,include_completed=True) if registration is not None else []
    targets=[t for t in targets if t.component not in ('U1','U4') and t.component not in locked]
    jobs=[(target,registration,locator) for target in targets]
    defect_targets=defective_ic_targets(frame,locator.hits,detections)
    missing_names={t.component for t in defect_targets if any(d.class_id==3 and
                   cv2.pointPolygonTest(np.float32(t.quad),tuple(map(float,np.float32(d.polygon).mean(axis=0))),False)>=0
                   for d in detections)}
    defect_registration=Registration(np.eye(3),1,1.,0.,method='trained_locator_obb')
    for target in defect_targets:
        if target.component in missing_names and target.component not in locked:
            targets.append(target);jobs.append((target,defect_registration,locator))
    pin_overlays=[];diode_overlays=[];calibration_notes=[]
    pin_jobs,calibration_notes,diode_axis,features=reference_future.result() if reference_future is not None else ([],[],None,None)
    located={reference.component for _,reference in pin_jobs}
    for name,reference in pin_calibrations.items():
        if name in located or features is None or cancel.is_set():continue
        recovered=recover_from_head(reference,features,locator.hits,markers,detections,locator)
        calibration_notes=[note for note in calibration_notes if not note.startswith(name+':')]
        calibration_notes.append(name+': '+reference.detail)
        if recovered is not None:
            pin_jobs.append((recovered,reference))
    observed_by_component={}
    if not hasattr(locator,'diode_roi'):locator.diode_roi=FixedDiodeROI()
    proposals=[np.float32(d.polygon) for d in detections] if pin_jobs else []
    for reg,reference in pin_jobs:
        if reference.component in missing_names:continue
        target=reference.target(frame,reg,proposals)
        if target is None:continue
        measured=observed_pin_boxes(target.pad_polygons,proposals)
        observed_by_component[target.component]=measured
        coverage=ledger.get(target.component,{}).get('coverage',{})
        for pin,quad in target.pad_polygons.items():
            if pin in measured and coverage.get(str(pin),{}).get('status')=='PASS' and coverage[str(pin)].get('confirmed'):
                pin_overlays.append(dict(component=target.component,pin=pin,quad=measured[pin].tolist(),state='PASS'))
        missing_on_pads=any(d.class_id==3 and any(cv2.intersectConvexConvex(
            np.float32(d.polygon),np.float32(pad))[0]>1 for pad in target.pad_polygons.values()) for d in detections)
        if reference.component not in locked and missing_on_pads:
            # A partial IC may have only previously passed pins in this view.
            # Still time a whole-part missing candidate without discarding those
            # saved pin results before the confirmation interval has elapsed.
            jobs.append((target,reg,reference));targets.append(target)
            continue
        pending=[pin for pin in target.joints if not (coverage.get(str(pin),{}).get('status')=='PASS' and coverage[str(pin)].get('confirmed'))]
        if pending:
            target=replace(target,joints=pending,pad_polygons={p:target.pad_polygons[p] for p in pending})
            jobs.append((target,reg,reference));targets.append(target)
    assigned={target.component for target,_,_ in jobs}
    for target in defect_targets:
        if target.component not in locked and target.component not in assigned:
            targets.append(target);jobs.append((target,defect_registration,locator))
    detected_ics={h.component for h in locator.hits if h.component in ('U1','U4')}
    ic_observations={}
    for name in detected_ics:
        hits=[h for h in locator.hits if h.component==name]
        candidates=[]
        if len(hits)==1:
            for d in detections:
                center=tuple(map(float,np.float32(d.polygon).mean(axis=0)))
                owners=[h.component for h in locator.hits if cv2.pointPolygonTest(np.float32(h.quad),center,False)>=0]
                if owners==[name]:candidates.append(dict(quad=d.polygon,class_id=d.class_id,confidence=d.confidence))
        ic_observations[name]=candidates
    for name in detected_ics-set(pin_calibrations or {})-locked-{t.component for t in defect_targets}:
        calibration_notes.append(f'{name}: พบ {len(ic_observations[name])} กรอบรอยต่อ · ตั้งค่าโปรแกรม > ขา IC / ไดโอด เพื่อผูกเลขขาจริง')
    reports=[]
    for target,reg,reference in jobs if model is not None else []:
        if cancel.is_set():return None
        measured=observed_by_component.get(target.component,{})
        pending_boxes={pin:measured[pin] for pin in target.joints if pin in measured}
        result,crop=inspect_target(model,class_map,frame,target,reg,threshold,device,
                                   model_path,board_id,reference,frame_detections=detections,observed=pending_boxes)
        if result[5].get('component_missing'):
            name=target.component
            seen_missing.add(name)
            since=missing_since.setdefault(name,observed_at)
            elapsed=max(0.,observed_at-since)
            if elapsed < IC_MISSING_CONFIRM_SECONDS-1e-9:
                pending_missing[name]=elapsed
                for detection in result[2]:
                    if detection.class_id==3:
                        pending_boxes_key=_missing_box_key(detection)
                        pending_missing_box=dict(component=name,quad=detection.polygon,elapsed=elapsed,required=IC_MISSING_CONFIRM_SECONDS)
                        pending_missing_boxes[pending_boxes_key]=pending_missing_box
                calibration_notes.append(f'{name}: รอยืนยัน missing {elapsed:.2f}/{IC_MISSING_CONFIRM_SECONDS:.2f} วินาที')
                # Keep the previous pin/board result intact. Independent solder
                # defects are still processed immediately while missing waits.
                if not any(d.class_id in (1,2) for d in result[2]):continue
                confirmed=[d for d in detections if d.class_id!=3 or _missing_box_key(d) not in pending_missing_boxes]
                result,crop=inspect_target(model,class_map,frame,target,reg,threshold,device,
                    model_path,board_id,reference,frame_detections=confirmed,observed=pending_boxes)
        result[5].update(locator_threshold=locator_threshold,continuous=True,
                         inspection_input='full_camera_frame')
        if target.component.startswith('D'):
            marker_ref=None
            if any(d.class_id==3 for d in result[2]):
                polarity=assess_diode(frame,target.quad,result[2],None)
            else:
                marker_ref=diode_marker_reference(locator.hits,markers,target.component)
                expected_axis=marker_ref['axis'] if marker_ref is not None else diode_axis
                body_quad=locator.diode_roi.quad(frame,target.quad,pad_quads=[d.polygon for d in result[2]],
                                                component=target.component)
                polarity=assess_diode(frame,target.quad,result[2],expected_axis,body_quad=body_quad)
            if marker_ref is not None:polarity['polarity_reference']=marker_ref
            if polarity.get('expected_cathode_direction') is None and polarity['status']=='REVIEW':
                reference_note=('ยังไม่ตั้งขั้ว + / - ของบอร์ด' if diode_direction is None else
                                'ตำแหน่งอ้างอิงขั้วหาย · เลื่อนกลับหรือตั้งขั้วใหม่')
                polarity['reason']+=' · '+reference_note
            result[5]['diode_reference']=(dict(image=str(diode_direction.path),points=diode_direction.points.tolist())
                                          if diode_direction is not None else None)
            result=combine_diode_result(result,polarity)
            item=dict(component=target.component,quad=target.quad.tolist(),**polarity);diode_overlays.append(item)
        if result[5]['physical_pin_numbers']:
            for pin,quad in pending_boxes.items():
                state=result[5]['joint_states'][pin]
                pin_overlays.append(dict(component=target.component,pin=pin,quad=quad.tolist(),state=state))
                draw_pin_number(result[1],target.component,pin,quad,(80,220,240))
        key=(target.component,result[3],tuple(sorted(d.class_id for d in result[2])),
             tuple(sorted(result[5]['joint_states'].items())),threshold,locator_threshold,
             result[5].get('diode_polarity',{}).get('status'),id(model),reference.fingerprint)
        cached=evidence_cache.get(target.component)
        changed=cached is None or cached[0]!=key
        report=save_auto_report(report_root,result,crop) if changed else cached[1]
        reports.append((result,report,key,changed))
    # Always explain a visible diode, even if its locator region is ambiguous or
    # clipped and cannot safely be assigned a polarity result.
    assessed={item['component'] for item in diode_overlays}
    for hit in locator.hits:
        if not hit.component.startswith('D') or hit.component in assessed:continue
        is_locked=hit.component in locked
        eligible=any(h.component==hit.component for h in locator.candidates)
        # Preview keeps measuring the current band even after checklist PASS is
        # latched. It does not create a report or overwrite the board result.
        owned=[d for d in detections if cv2.pointPolygonTest(np.float32(hit.quad),
               tuple(map(float,np.float32(d.polygon).mean(axis=0))),False)>=0]
        marker_ref=None
        if any(d.class_id==3 for d in owned):
            observed=assess_diode(frame,hit.quad,owned,None)
        else:
            marker_ref=diode_marker_reference(locator.hits,markers,hit.component) if eligible else None
            expected_axis=marker_ref['axis'] if marker_ref is not None else (diode_axis if eligible else None)
            body_quad=locator.diode_roi.quad(frame,hit.quad,remember=eligible,pad_quads=[d.polygon for d in owned],component=hit.component)
            observed=inspect_diode(frame,hit.quad,expected_axis,allow_printed_reference=eligible,body_quad=body_quad)
        if marker_ref is not None:observed['polarity_reference']=marker_ref
        if not eligible:observed['reason']='กรอบไม่ครบหรือชื่อซ้ำ · '+observed['reason']
        item=dict(component=hit.component,quad=hit.quad.tolist(),locked=is_locked,allow_printed_reference=eligible,**observed)
        diode_overlays.append(item)
        assessed.add(hit.component)
    missing_since={name:since for name,since in missing_since.items() if name in seen_missing}
    detections=[d for d in detections if d.class_id!=3 or _missing_box_key(d) not in pending_missing_boxes]
    for name,candidates in ic_observations.items():
        ic_observations[name]=[item for item in candidates if item['class_id']!=3 or
                               np.float32(item['quad']).tobytes() not in pending_missing_boxes]
    missing_quads=[detection_polygon(d) for d in detections if d.class_id==3]
    for hit in locator.hits:
        if hidden_by_missing(hit.quad,missing_quads):continue
        cv2.polylines(annotated,[np.int32(hit.quad)],True,(255,210,80),2)
        x,y=np.maximum(0,np.min(hit.quad,axis=0)).astype(int)
        cv2.putText(annotated,f'{hit.component} {hit.confidence:.2f}',(x,max(20,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,210,80),2)
    for detection in detections:draw_detection(annotated,detection)
    for item in pending_missing_boxes.values():draw_pending_missing(annotated,item)
    for item in pin_overlays:
        if hidden_by_missing(item['quad'],missing_quads):continue
        draw_pin_number(annotated,item['component'],item['pin'],item['quad'],(80,220,240))
    for item in diode_overlays:draw_diode(annotated,item)
    return dict(locator=locator,registration=registration,targets=targets,reports=reports,
                locator_hits=list(locator.hits),raw=frame,annotated=annotated,detections=detections,
                pin_overlays=pin_overlays,diode_overlays=diode_overlays,calibration_notes=calibration_notes,
                locked_components=sorted({h.component for h in locator.hits}&locked),ic_observations=ic_observations,
                ic_missing_since=missing_since,pending_ic_missing=pending_missing,
                pending_missing_boxes=list(pending_missing_boxes.values()),
                marker_hits=markers,ic_head_markers={name:[m.quad.tolist() for m in assigned_markers(locator.hits,markers,name) if m.component=='ic_head']
                                                   for name in detected_ics})
