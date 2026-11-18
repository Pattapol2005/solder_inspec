"""Local board registration and automatic microscope capture; no full-board view required."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import time

import cv2
import numpy as np

from board import COMPONENTS
from inspection import (Detection, CLASSES, read_image, save_report, write_image,
                        prediction_quads, detection_from_quad, draw_detection, quality_detections,visible_detections)


def corners(box):
    x1,y1,x2,y2=box
    return np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


def transform(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1,1,2),matrix).reshape(-1,2)


def pad_boxes(name):
    """Default pad locations on this specific training-board reference image."""
    x1,y1,x2,y2=COMPONENTS[name]['box'];cy=(y1+y2)/2
    if name[0] in ('R','D'):
        return {1:(x1+2,cy-7,x1+15,cy+7),2:(x2-15,cy-7,x2-2,cy+7)}
    if name[0]=='Q':
        cx=(x1+x2)/2
        return {1:(x1+1,y2-13,x1+10,y2-2),2:(x2-10,y2-13,x2-1,y2-2),
                3:(cx-5,y1+1,cx+5,y1+13)}
    if name=='U1':
        ys=np.linspace(103,169,7)
        return {**{i+1:(333,y-3,353,y+3) for i,y in enumerate(ys)},
                **{14-i:(381,y-3,401,y+3) for i,y in enumerate(ys)}}
    xs=np.linspace(268,336,14)
    return {**{i+1:(x-2,402,x+2,431) for i,x in enumerate(xs)},
            **{28-i:(x-2,342,x+2,362) for i,x in enumerate(xs)}}


class StableCapture:
    def __init__(self,hold=.6,min_sharpness=18,motion_limit=2.0):
        self.hold=hold;self.min_sharpness=min_sharpness;self.motion_limit=motion_limit
        self.reset()

    def reset(self):
        self.previous=None;self.sequence=None;self.since=None;self.last_fresh=None

    def observe(self,frame,sequence,now=None):
        now=time.monotonic() if now is None else now
        if sequence==self.sequence:return False,'รอภาพใหม่จากกล้อง'
        self.sequence=sequence
        h,w=frame.shape[:2]
        gray=cv2.cvtColor(cv2.resize(frame,(320,max(1,round(h*320/w)))),cv2.COLOR_BGR2GRAY)
        sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        gap=self.last_fresh is None or now-self.last_fresh>.4
        self.last_fresh=now
        if self.previous is None or self.previous.shape!=gray.shape or gap:
            motion=float('inf')
        else:
            delta=gray.astype(np.float32)-self.previous
            motion=float(np.mean(np.abs(delta-np.median(delta))))
        self.previous=gray
        if sharpness<self.min_sharpness:
            self.since=None
            return False,f'ภาพยังไม่คม · {sharpness:.0f} / {self.min_sharpness:.0f}'
        if motion>self.motion_limit:
            self.since=None
            return False,'กำลังเลื่อนบอร์ด · รอภาพหยุดนิ่ง'
        if self.since is None:self.since=now
        ready=now-self.since>=self.hold
        return ready,'ภาพนิ่งพร้อมตรวจ' if ready else 'รอภาพนิ่งสักครู่…'


@dataclass
class Registration:
    matrix: np.ndarray
    inliers: int
    ratio: float
    error: float
    method: str = 'reference_registration'
    anchors: list = field(default_factory=list)


@dataclass
class ScanTarget:
    component: str
    joints: list
    quad: np.ndarray
    crop: np.ndarray
    crop_to_frame: np.ndarray
    pad_polygons: dict


class BoardLocator:
    def __init__(self,path):
        self.path=Path(path)
        self.reference=read_image(self.path)
        self.fingerprint=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sift=cv2.SIFT_create(nfeatures=6500,contrastThreshold=.02)
        feature_image=self.reference
        if feature_image.shape[1]<900:
            feature_image=cv2.resize(feature_image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        gray=cv2.cvtColor(feature_image,cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        scale=np.array([gray.shape[1]/431,gray.shape[0]/447])
        for data in COMPONENTS.values():
            box=corners(data['box'])*scale
            cv2.fillConvexPoly(mask,np.int32(box),0)
        # Preserve high-resolution user references. Coordinates are normalized to the board map.
        self.keypoints,self.descriptors=self.sift.detectAndCompute(gray,mask)
        self.ref_scale=np.array([431/gray.shape[1],447/gray.shape[0]],np.float32)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)

    def locate(self,frame):
        h,w=frame.shape[:2];scale=min(1.,1100/max(h,w))
        query=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
        kp,des=self.sift.detectAndCompute(cv2.cvtColor(query,cv2.COLOR_BGR2GRAY),None)
        if des is None or self.descriptors is None or len(des)<10:return None
        pairs=self.matcher.knnMatch(des,self.descriptors,k=2)
        good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.68*pair[1].distance]
        # Reciprocal checks remove matches to repeated resistor/pad patterns.
        reverse=self.matcher.match(self.descriptors,des)
        good=[m for m in good if reverse[m.trainIdx].trainIdx==m.queryIdx]
        if len(good)<10:return None
        src=np.float32([self.keypoints[m.trainIdx].pt for m in good])*self.ref_scale
        dst=np.float32([kp[m.queryIdx].pt for m in good])/scale
        matrix,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0/scale,maxIters=3000,confidence=.999)
        if matrix is None or mask is None or not np.isfinite(matrix).all():return None
        valid=mask.ravel().astype(bool);count=int(valid.sum());ratio=count/len(good)
        if count<10 or ratio<.65:return None
        projected=transform(src[valid],matrix)
        error=float(np.median(np.linalg.norm(projected-dst[valid],axis=1)))
        hull=cv2.convexHull(dst[valid])
        if cv2.contourArea(hull)<w*h*.025 or error>3.0/scale:return None
        try:
            inverse=np.linalg.inv(matrix)
            view=transform(corners((0,0,w,h)),inverse)
        except np.linalg.LinAlgError:return None
        if not np.isfinite(view).all() or not cv2.isContourConvex(view):return None
        if cv2.contourArea(view)<100 or cv2.contourArea(view)>431*447*2:return None
        # Reject mirror/reflection and physically implausible skew at the visible center.
        center=transform(np.float32([[w/2,h/2]]),inverse)[0]
        basis=transform([center,center+[10,0],center+[0,10]],matrix)
        a,b=basis[1]-basis[0],basis[2]-basis[0]
        cross=float(a[0]*b[1]-a[1]*b[0])
        lengths=[np.linalg.norm(a),np.linalg.norm(b)]
        if cross<=0 or min(lengths)<.5 or max(lengths)/min(lengths)>2.5:return None
        return Registration(matrix,count,ratio,error)

    def targets(self,frame,registration,results):
        h,w=frame.shape[:2];found=[]
        for name,data in COMPONENTS.items():
            coverage=results.get(name,{}).get('coverage',{})
            pads={};pending=[]
            for joint,box in pad_boxes(name).items():
                polygon=transform(corners(box),registration.matrix)
                if not np.isfinite(polygon).all():continue
                if (polygon[:,0].min()<6 or polygon[:,1].min()<6 or
                    polygon[:,0].max()>w-6 or polygon[:,1].max()>h-6):continue
                if cv2.contourArea(polygon)<35:continue
                pads[joint]=polygon
                if not coverage.get(str(joint),{}).get('confirmed',False):pending.append(joint)
            if not pending:continue
            # Keep only visible pads; partial IC images are valid independent captures.
            if len(pads)==data['expected']:
                refbox=data['box']
            else:
                boxes=[pad_boxes(name)[j] for j in pads]
                refbox=(min(b[0] for b in boxes)-3,min(b[1] for b in boxes)-3,
                        max(b[2] for b in boxes)+3,max(b[3] for b in boxes)+3)
            quad=transform(corners(refbox),registration.matrix)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):continue
            # A perspective-normalized crop avoids adjacent components in a rotated bounding box.
            cw=max(12,round(max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3]))))
            ch=max(12,round(max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1]))))
            if cw*ch>frame.shape[0]*frame.shape[1]*2:continue
            destination=corners((0,0,cw,ch))
            to_crop=cv2.getPerspectiveTransform(quad,destination)
            crop=cv2.warpPerspective(frame,to_crop,(cw,ch))
            found.append(ScanTarget(name,sorted(pads),quad,crop,np.linalg.inv(to_crop),pads))
        return found


def assess_pads(detections,target,polygons=None,observed=None):
    """Each visible pad needs its own good box. Defects spanning several pads remain FAIL."""
    assignments={j:[] for j in target.joints};unmapped_defect=False
    for index,d in enumerate(detections):
        matches=[]
        x1,y1,x2,y2=d.box
        for j,poly in target.pad_polygons.items():
            center=poly.mean(axis=0)
            inside=(x1<=center[0]<=x2 and y1<=center[1]<=y2) if polygons is None else cv2.pointPolygonTest(np.float32(polygons[index]),tuple(map(float,center)),False)>=0
            if inside:matches.append(j)
        if observed is not None:
            measured=[pin for pin,quad in observed.items() if np.array_equal(quad,polygons[index])]
            matches=measured if d.class_id==0 else list(set(matches).union(measured))
        if d.class_id and not matches:unmapped_defect=True
        # A single huge GOOD box cannot certify multiple solder joints.
        if d.class_id==0 and len(matches)!=1:continue
        for j in matches:assignments[j].append(d)
    states={}
    for j,items in assignments.items():
        if any(d.class_id!=0 for d in items):states[j]='FAIL'
        elif len(items)==1 and not unmapped_defect:states[j]='PASS'
        else:states[j]='REVIEW'
    state='FAIL' if any(d.class_id for d in detections) else ('PASS' if all(s=='PASS' for s in states.values()) else 'REVIEW')
    reason={'PASS':'ตรวจพบรอยต่อปกติครบจุดที่เห็น','FAIL':'พบรอยบัดกรีที่สงสัยว่าบกพร่อง','REVIEW':'จับคู่รอยต่อกับจุดตรวจยังไม่ครบ'}[state]
    return state,reason,states


def observed_pin_boxes(pads, polygons):
    """Allow longitudinal calibration drift while keeping adjacent pins distinct."""
    matches = {pin: [] for pin in pads}
    geometry = []
    for pin, points in pads.items():
        pad = np.float32(points)
        edges = np.roll(pad,-1,axis=0) - pad
        long_edge = edges[np.argmax(np.linalg.norm(edges,axis=1))]
        length = float(np.linalg.norm(long_edge))
        area = cv2.contourArea(pad)
        if area < 1 or length < 1:continue
        normal = np.array([-long_edge[1],long_edge[0]]) / length
        geometry.append((pin,pad,area,pad.mean(axis=0),normal,area/length))
    for points in polygons:
        polygon = np.float32(points)
        center = polygon.mean(axis=0)
        area = cv2.contourArea(polygon)
        if area < 1:continue
        owners = []
        for pin,pad,pad_area,pad_center,normal,width in geometry:
            intersection,_ = cv2.intersectConvexConvex(pad,polygon)
            overlap = intersection / min(area,pad_area)
            if overlap < .18:continue
            # Even a partial overlap with a second pin vetoes a numbered box.
            across = abs(float((center-pad_center) @ normal))
            eligible = overlap >= .35 and across <= width*.55 and .20 <= area/pad_area <= 3
            owners.append((pin,eligible))
        if len(owners)==1 and owners[0][1]:
            matches[owners[0][0]].append(polygon)
    return {pin: items[0] for pin, items in matches.items() if len(items) == 1}


def inspect_target(model,class_map,frame,target,registration,threshold,device,model_path,board_id,locator,frame_detections=None,observed=None):
    if frame_detections is None:
        from runtime import inference_options
        prediction=model.predict(source=target.crop,conf=threshold,iou=.45,imgsz=1024,device=device,verbose=False,**inference_options(device))[0]
        source_detections=quality_detections(prediction,class_map,threshold,getattr(model,'names',None))
    else:source_detections=frame_detections
    detections=[]
    polygons=[]
    trained = registration.method == 'trained_locator_obb'
    physical_pins=registration.method=='ic_pin_reference'
    for detection in source_detections:
        points=np.float32(detection.polygon)
        if frame_detections is None:points=transform(points,target.crop_to_frame)
        if trained and cv2.pointPolygonTest(np.float32(target.quad), tuple(map(float,points.mean(axis=0))), False) < 0:
            continue
        if physical_pins:
            overlaps=False
            for pad in target.pad_polygons.values():
                area,_=cv2.intersectConvexConvex(points,np.float32(pad))
                if area>1:overlaps=True;break
            if not overlaps:continue
        polygons.append(points)
        detections.append(detection_from_quad(points,detection.class_id,detection.confidence,
                                               detection.native_class_id,detection.native_class_name))
    detections=visible_detections(detections)
    polygons=[np.float32(d.polygon) for d in detections]
    measured = {}
    if physical_pins:
        measured = observed_pin_boxes(target.pad_polygons,polygons) if observed is None else observed
    if trained:
        state,reason,joint_states=assess_component(detections,target)
    else:
        state,reason,joint_states=assess_pads(detections,target,polygons,measured if physical_pins else None)
    component_missing=target.component in ('U1','U4') and any(d.class_id==3 for d in detections)
    if component_missing:
        state='FAIL'
        reason='พบ missing ที่ฝั่งขาที่เห็น · ตัดสิน '+target.component+' เป็น missing ทั้งตัว'
        joint_states={pin:'REVIEW' for pin in target.joints}
        physical_pins=False
    defect_regions=[]
    lo,hi=np.min(target.quad,axis=0),np.max(target.quad,axis=0)
    defect_names={1:'ตะกั่วเชื่อมขา (bridge)',2:'ตะกั่วมากเกิน (excess)',3:'ไม่พบอุปกรณ์/รอยบัดกรี (missing)'}
    for detection in detections:
        if detection.class_id==0:continue
        polygon=np.float32(detection.polygon)
        position=(polygon.mean(axis=0)-lo)/np.maximum(hi-lo,1)
        horizontal='ซ้าย' if position[0]<1/3 else ('ขวา' if position[0]>2/3 else 'กลาง')
        vertical='บน' if position[1]<1/3 else ('ล่าง' if position[1]>2/3 else 'กลาง')
        pins=[]
        if physical_pins:
            for pin,pad in target.pad_polygons.items():
                intersection,_=cv2.intersectConvexConvex(polygon,np.float32(pad))
                if intersection>=cv2.contourArea(np.float32(pad))*.25:pins.append(pin)
        location='ด้าน'+horizontal+' ช่วง'+vertical+'ของภาพ'
        if pins:location+=' · ขา '+', '.join(map(str,sorted(pins)))
        defect_regions.append(dict(class_name=CLASSES[detection.class_id],confidence=detection.confidence,
                                   location=location,pins=pins,polygon=polygon.tolist()))
    if defect_regions and not component_missing:
        state='FAIL'
        reason='\n'.join(defect_names[d.class_id]+' · '+region['location']
                         for d,region in zip([d for d in detections if d.class_id!=0],defect_regions))
    annotated=frame.copy()
    if not physical_pins:
        cv2.polylines(annotated,[np.int32(target.quad)],True,(160,230,40),2)
    for d in detections:
        draw_detection(annotated,d)
    label='MISSING' if component_missing else state
    cv2.putText(annotated,f'{target.component}: {label}',tuple(np.int32(target.quad[0])),cv2.FONT_HERSHEY_SIMPLEX,.7,
                (40,60,240) if component_missing else (160,230,40),2)
    metadata=dict(component=target.component,board_id=board_id,region=COMPONENTS[target.component]['region'],
                  expected=len(target.joints),threshold=threshold,model=str(Path(model_path).resolve()),
                  scope='visible_joints_only',resolution=[frame.shape[1],frame.shape[0]],device=device,
                  component_source=registration.method,automatic=True,joints=target.joints,
                  joint_states=joint_states,roi_quad=target.quad.tolist(),crop_to_frame=target.crop_to_frame.tolist(),
                  confirmed_by='calibrated_ic_pin_reference' if physical_pins else ('full_component_detection_count' if trained else 'automatic_pad_mapping'),
                  joint_identity='physical_ic_pin_numbers' if physical_pins else ('not_resolved' if trained else 'reference_pad_mapping'),
                  class_names=list(CLASSES),
                  defect_regions=defect_regions,
                  model_class_names=getattr(model,'names',None),
                  physical_pin_numbers=physical_pins,merge_visible_pins=physical_pins,
                  pad_polygons={int(pin):measured.get(pin,poly).tolist() for pin,poly in target.pad_polygons.items()} if physical_pins else {},
                  calibration_points=getattr(locator,'points',None) if physical_pins else None,
                  reference=str(locator.path.resolve()),reference_sha256=locator.fingerprint,
                  registration=dict(inliers=registration.inliers,ratio=registration.ratio,error=registration.error,
                                    method=registration.method,anchors=registration.anchors,matrix=registration.matrix.tolist()))
    if component_missing:metadata.update(component_missing=True,confirmed_by='component_missing_detection')
    result=(frame,annotated,detections,state,reason,metadata)
    return result,target.crop


def assess_component(detections,target):
    """Whole-footprint coverage only; an OBB does not establish physical pin IDs."""
    states={j:'REVIEW' for j in target.joints}
    if any(d.class_id != 0 for d in detections):
        return 'FAIL','พบข้อบกพร่องในอุปกรณ์ · ยังไม่ระบุหมายเลขขาที่เสีย',states
    if len(detections) != len(target.joints):
        return 'REVIEW',f'พบรอยต่อปกติ {len(detections)}/{len(target.joints)} จุด · ยังตรวจไม่ครบ',states
    # Overlapping GOOD proposals must not inflate the apparent joint count.
    for i,a in enumerate(detections):
        ax,ay,bx,by=a.box
        area_a=max(0,bx-ax)*max(0,by-ay)
        for b in detections[i+1:]:
            cx,cy,dx,dy=b.box
            intersection=max(0,min(bx,dx)-max(ax,cx))*max(0,min(by,dy)-max(ay,cy))
            smaller=min(area_a,max(0,dx-cx)*max(0,dy-cy))
            if smaller and intersection/smaller>.5:
                return 'REVIEW','กรอบรอยต่อปกติซ้อนกัน · ยังยืนยันจำนวนจุดไม่ได้',states
    return 'PASS','พบรอยต่อปกติครบอุปกรณ์ที่เห็นเต็มกรอบ · ยืนยันด้วยจำนวนจุด', {j:'PASS' for j in target.joints}


def save_auto_report(folder,result,crop):
    from evidence import result_evidence
    report=save_report(folder,*result)
    write_image(report/'crop.png',crop)
    write_image(report/'component_result.png',result_evidence(result))
    if result[5].get('physical_pin_numbers'):
        for pin in result[5].get('pad_polygons',{}):
            write_image(report/f'pin_{int(pin):02d}.png',result_evidence(result,pin))
    return report
