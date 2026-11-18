"""Recover a manually numbered IC from its head marker and two measured lead rows."""
import cv2
import numpy as np

from inspection import prediction_quads
from locator_markers import assigned_markers, normalize_name
from runtime import inference_options
from scanner import corners, pad_boxes, Registration


def reference_head_offset(calibration, locator):
    key=(id(locator.model),tuple(sorted(locator.mapping.items())))
    cached=getattr(calibration,'_reference_head_cache',None)
    if cached is not None and cached[0]==key:return cached[1]
    points=np.float32(calibration.points)
    basis=np.column_stack((points[1]-points[0],points[2]-points[0]))
    prediction=locator.model.predict(source=calibration.reference_image,conf=.25,imgsz=1024,
        device=locator.device,verbose=False,**inference_options(locator.device))[0]
    offsets=[]
    for quad,class_id,confidence in prediction_quads(prediction):
        if normalize_name(locator.mapping.get(class_id,''))!='ic_head' or confidence<.25:continue
        offset=np.linalg.solve(basis,np.float32(quad).mean(axis=0)-points[0])
        # This is the operator's saved pin-1 end, not an arbitrary dot on an IC.
        if -.25<=offset[0]<=.95 and .05<=offset[1]<=.95:offsets.append(offset)
    found=offsets[0] if len(offsets)==1 else None
    calibration._reference_head_cache=(key,found)
    return found


def fit_head_rows(component, points, head_offset, head, polygons, shape):
    """Return current pins 1, 2 and N only when both head-end rows agree."""
    h,w=shape[:2]
    pads=[];axes=[]
    for polygon in polygons:
        q=np.float32(polygon).reshape(4,2)
        if not np.isfinite(q).all() or q.min()<3 or q[:,0].max()>w-4 or q[:,1].max()>h-4:continue
        edges=np.roll(q,-1,axis=0)-q
        lengths=np.linalg.norm(edges,axis=1)
        longest=edges[int(np.argmax(lengths))]
        if lengths.min()<3 or lengths.max()/lengths.min()<1.5:continue
        angle=np.arctan2(longest[1],longest[0])
        axes.append([np.cos(2*angle),np.sin(2*angle)])
        pads.append(q.mean(axis=0))
    if len(pads)<6:return None
    mean=np.mean(axes,axis=0)
    if np.linalg.norm(mean)<.85:return None
    angle=np.arctan2(mean[1],mean[0])/2
    across=np.array([np.cos(angle),np.sin(angle)])
    along=np.array([-across[1],across[0]])
    centers=np.float32(pads)
    ordered=np.argsort(centers@across)
    gaps=np.diff((centers@across)[ordered]);cut=int(np.argmax(gaps))+1
    rows=[centers[ordered[:cut]],centers[ordered[cut:]]]
    if min(map(len,rows))<3:return None
    separation=float(np.diff([np.mean(row@across) for row in rows])[0])
    distances=np.concatenate([np.diff(np.sort(row@along)) for row in rows])
    if distances.min()<4:return None
    pitch=float(np.median(np.sort(distances)[:max(2,(len(distances)+1)//2)]))
    if separation<2*pitch or any(np.ptp(row@across)>.35*pitch for row in rows):return None
    reference=np.float32(points)
    ref_pitch=np.linalg.norm(reference[1]-reference[0])
    ref_across=np.linalg.norm(reference[2]-reference[0])
    if not .65<(separation/pitch)/(ref_across/ref_pitch)<1.5:return None
    low=np.mean([np.min(row@along) for row in rows])
    high=np.mean([np.max(row@along) for row in rows])
    head=np.float32(head)
    if abs(float(head@along)-(low+high)/2)<.25*pitch:return None
    if head@along>(low+high)/2:along=-along
    half=7 if component=='U1' else 14
    fitted=[]
    for row in rows:
        row=row[np.argsort(row@along)]
        steps=np.rint(np.diff(row@along)/pitch).astype(int)
        if steps.min()<1 or steps.max()>3:return None
        ranks=np.r_[0,np.cumsum(steps)]
        if ranks[-1]>=half:return None
        model=np.column_stack((np.ones(len(row)),ranks))
        origin,step=np.linalg.lstsq(model,row,rcond=None)[0]
        if np.max(np.linalg.norm(model@np.stack((origin,step))-row,axis=1))>.18*pitch:return None
        fitted.append((origin,step))
    step=(fitted[0][1]+fitted[1][1])/2
    if abs(np.linalg.norm(step)/pitch-1)>.15:return None
    if np.linalg.norm(fitted[0][1]-fitted[1][1])>.18*pitch:return None
    ref_cross=np.linalg.det(np.stack((reference[1]-reference[0],reference[2]-reference[0])))
    for first in (0,1):
        pin1=fitted[first][0];opposite=fitted[1-first][0]
        if abs(float((opposite-pin1)@along))>.25*pitch:continue
        if np.linalg.det(np.stack((step,opposite-pin1)))*ref_cross<=0:continue
        offset=np.linalg.solve(np.column_stack((step,opposite-pin1)),head-pin1)
        if abs(float(offset[0]-head_offset[0]))>.28 or abs(float(offset[1]-head_offset[1]))>.27:continue
        if not .05<float(offset[1])<.95:continue
        return np.float32([pin1,pin1+step,opposite])
    return None


def recover_from_head(calibration, features, hits, markers, detections, locator):
    component=calibration.component
    owners=[hit for hit in hits if hit.component==component]
    heads=[m for m in assigned_markers(hits,markers,component) if m.component=='ic_head']
    if len(owners)!=1 or len(heads)!=1:return None
    polygons=[]
    for detection in detections:
        if detection.class_id!=0:continue
        center=tuple(map(float,np.float32(detection.polygon).mean(axis=0)))
        names={hit.component for hit in hits if cv2.pointPolygonTest(np.float32(hit.quad),center,False)>=0}
        if names=={component}:polygons.append(detection.polygon)
    if len(polygons)<6:
        calibration.detail='พบ ic_head · รอเห็นขาอย่างน้อย 3 ขาต่อฝั่งที่ปลายขา 1'
        return None
    offset=reference_head_offset(calibration,locator)
    if offset is None:
        calibration.detail='ภาพอ้างอิงยังจับคู่ ic_head ใกล้ขา 1 ไม่ได้ · ตั้งจุดอ้างอิงให้เห็นหัว IC'
        return None
    points=fit_head_rows(component,calibration.points,offset,np.float32(heads[0].quad).mean(axis=0),polygons,features.shape)
    if points is None:
        calibration.detail='พบหัว IC แต่แนวขายังจับคู่กับขา 1 ที่ตั้งไว้ไม่ชัด'
        return None
    source=np.float32([corners(pad_boxes(component)[pin]).mean(axis=0) for pin in (1,2,calibration.total)])
    matrix=np.vstack((cv2.getAffineTransform(source,points),[0,0,1]))
    from frame_registration import FrameRegistration
    calibration.recovery_tracker=FrameRegistration(features)
    calibration.recovery_pose=matrix
    calibration.row_pitch.clear()
    calibration.detail='ระบุขาจริงด้วย ic_head และสองแถวขา · อ้างอิงขา 1 ที่ตั้งไว้'
    registration=Registration(matrix,len(polygons),1.,0.,method='ic_pin_reference',
        anchors=[dict(component=component,pins=[1,2,calibration.total],source='ic_head_and_calibrated_rows',
                      head=np.asarray(heads[0].quad).tolist(),points=points.tolist())])
    calibration.last_registration=registration
    return registration
