"""Physical SOP14/SSOP28 pin indices from three operator-confirmed reference pads."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

from inspection import read_image,write_image
from scanner import Registration,ScanTarget,pad_boxes,corners,transform,observed_pin_boxes
from frame_registration import FrameRegistration
from calibration_files import load_calibration_profile

TOTAL_PINS={'U1':14,'U4':28}


def align_pin_row(pads, pins, proposals):
    """Refine a calibrated row's pitch. Its first pin stays tied to the reference."""
    origin = pads[pins[0]].mean(axis=0)
    step = pads[pins[1]].mean(axis=0) - origin
    pitch = np.linalg.norm(step)
    if pitch < 4:return
    axis = step / pitch
    normal = np.array([-axis[1],axis[0]])
    depth = np.ptp(pads[pins[0]] @ normal)
    positions = []
    for poly in proposals:
        poly=np.float32(poly)
        delta = poly.mean(axis=0) - origin
        along = float(delta @ axis / pitch)
        if not -.7 <= along <= len(pins)-.3:continue
        if abs(delta @ normal) > .85*depth:continue
        if not (.18*pitch <= np.ptp(poly @ axis) <= 1.08*pitch and
                .22*depth <= np.ptp(poly @ normal) <= 2*depth):continue
        positions.append(along)
    if len(positions)<4:return
    positions = np.sort(positions)
    gaps = np.diff(positions)
    adjacent = gaps[(gaps>=.85)&(gaps<=1.15)]
    if len(adjacent)<3:return
    ratio = np.median(adjacent)
    # Work from the calibrated end even when only the far end is visible.
    slots = np.rint(positions/ratio).astype(int)
    valid = ((slots>=0)&(slots<len(pins))&(np.abs(positions-slots*ratio)<.30))
    if valid.mean()<.8:return
    slots,positions = slots[valid],positions[valid]
    if len(slots)<4 or len(np.unique(slots))!=len(slots) or np.ptp(slots)<3:return
    # Fit pitch through the anchored end. A free intercept on a partial row
    # can silently shift all physical pin numbers by one.
    ratio = float(slots @ positions / (slots @ slots))
    if not .88 <= ratio <= 1.12 or np.max(np.abs(positions-slots*ratio))>.23:return
    return ratio


def refine_pin_points(component,points,centers):
    """Fit pitch from several visible joints rather than extrapolating click error."""
    points=np.float32(points);centers=np.float32(centers).reshape(-1,2)
    if len(centers)<6:return points,0
    total=TOTAL_PINS[component];boxes=pad_boxes(component)
    source=np.float32([corners(boxes[i]).mean(axis=0) for i in (1,2,total)])
    initial=cv2.getAffineTransform(source,points)
    pitch=float(np.linalg.norm(points[1]-points[0]))
    pairs=[];used=set()
    for pin,box in boxes.items():
        ref=corners(box).mean(axis=0)
        expected=cv2.transform(ref.reshape(1,1,2),initial).reshape(2)
        distances=np.linalg.norm(centers-expected,axis=1);index=int(np.argmin(distances))
        if distances[index]<pitch*.28:pairs.append((float(distances[index]),pin,ref,index))
    selected=[]
    for distance,pin,ref,index in sorted(pairs,key=lambda row:row[0]):
        if index not in used:selected.append((pin,ref,centers[index]));used.add(index)
    half=total//2
    if len(selected)<6 or min(sum(p<=half for p,_,_ in selected),sum(p>half for p,_,_ in selected))<2:return points,0
    matrix,mask=cv2.estimateAffine2D(np.float32([p[1] for p in selected]),np.float32([p[2] for p in selected]),
                       method=cv2.RANSAC,ransacReprojThreshold=max(2,pitch*.06),maxIters=500)
    if matrix is None or mask is None or mask.sum()<6:return points,0
    fitted=cv2.transform(source.reshape(-1,1,2),matrix).reshape(-1,2)
    fitted_pitch=np.linalg.norm(fitted[1]-fitted[0])
    if np.max(np.linalg.norm(fitted-points,axis=1))>pitch*.25 or not .85<fitted_pitch/pitch<1.15:return points,0
    return fitted,int(mask.sum())


class PinCalibration:
    def __init__(self,component,image,points,path):
        self.component=component;self.total=TOTAL_PINS[component]
        self.path=Path(path)
        self.reference_image=image.copy()
        self.fingerprint=hashlib.sha256(image.tobytes()).hexdigest()
        points=np.float32(points).reshape(3,2)
        pitch=points[1]-points[0];across=points[2]-points[0]
        h,w=image.shape[:2]
        if (not np.isfinite(points).all() or points.min()<0 or points[:,0].max()>=w or points[:,1].max()>=h):
            raise ValueError('จุดอ้างอิงต้องอยู่ในภาพ')
        cross=abs(float(pitch[0]*across[1]-pitch[1]*across[0]))
        if np.linalg.norm(pitch)<8 or np.linalg.norm(across)<15 or cross<.4*np.linalg.norm(pitch)*np.linalg.norm(across):
            raise ValueError('คลิกขา 1, 2 และขาฝั่งตรงข้ามให้ชัด จุดต้องไม่ซ้อนหรืออยู่แนวเดียวกัน')
        boxes=pad_boxes(component)
        source=np.float32([corners(boxes[i]).mean(axis=0) for i in (1,2,self.total)])
        self.board_to_anchor=np.vstack((cv2.getAffineTransform(source,points),[0,0,1]))
        self.tracker=FrameRegistration(image)
        self.recovery_tracker=None;self.recovery_pose=None;self.last_registration=None
        self.row_pitch={}
        self.points=points.tolist();self.detail='พร้อมตรวจ '+component+' รายขาจริง'

    def locate(self,features):
        match=self.tracker.locate(features)
        self.detail=self.tracker.detail
        pose=self.board_to_anchor
        if match is None and self.recovery_tracker is not None:
            match=self.recovery_tracker.locate(features)
            pose=self.recovery_pose
            self.detail='อ้างอิง ic_head · '+self.recovery_tracker.detail
        if match is None:
            self.last_registration=None
            return None
        self.last_registration=Registration(match.matrix@pose,match.inliers,match.ratio,match.error,
                            method='ic_pin_reference',anchors=[{'component':self.component,'pins':[1,2,self.total]}])
        return self.last_registration

    def target(self,frame,registration,proposals=()):
        h,w=frame.shape[:2];pads={}
        projected={pin:transform(corners(box),registration.matrix) for pin,box in pad_boxes(self.component).items()}
        if any(not np.isfinite(poly).all() for poly in projected.values()):return None
        half=self.total//2
        for row in (range(1,half+1),range(self.total,half,-1)):
            ratio=align_pin_row(projected,row,proposals) if len(proposals)>=4 else None
            if ratio is not None:self.row_pitch[row.start]=ratio
            ratio=self.row_pitch.get(row.start,1.)
            step=projected[row[1]].mean(axis=0)-projected[row[0]].mean(axis=0)
            for index,pin in enumerate(row):
                projected[pin]=projected[pin]+step*(index*(ratio-1))
        edge_pads=[]
        for pin,polygon in projected.items():
            if cv2.contourArea(polygon)<25:continue
            if (polygon[:,0].min()<4 or polygon[:,1].min()<4 or polygon[:,0].max()>w-4 or polygon[:,1].max()>h-4):
                edge_pads.append(pin)
                continue
            pads[pin]=polygon
        if edge_pads and len(proposals):
            measured=observed_pin_boxes(projected,proposals)
            for pin in edge_pads:
                box=measured.get(pin)
                if box is not None and (box.min(axis=0)>=3).all() and (box.max(axis=0)<=[w-4,h-4]).all():
                    pads[pin]=box
        if not pads:return None
        all_points=np.concatenate(list(pads.values()))
        x1,y1=np.maximum(0,np.floor(all_points.min(axis=0)-3)).astype(int)
        x2,y2=np.minimum([w,h],np.ceil(all_points.max(axis=0)+3)).astype(int)
        quad=corners((x1,y1,x2,y2))
        return ScanTarget(self.component,sorted(pads),quad,frame[y1:y2,x1:x2].copy(),
                          np.array([[1.,0.,x1],[0.,1.,y1],[0.,0.,1.]]),pads)


def save_calibration(root,kind,image,points,extra=None):
    folder=Path(root)/kind/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    image_path=folder/'reference.png'
    if kind in TOTAL_PINS:PinCalibration(kind,image,points,image_path)
    elif kind=='diode':
        p=np.float32(points).reshape(2,2)
        if not np.isfinite(p).all() or np.linalg.norm(p[1]-p[0])<15:raise ValueError('กำหนดขั้ว + และขั้ว - ให้ห่างกันอย่างน้อย 15 px')
    else:raise ValueError('ไม่รู้จักชนิดการตั้งอ้างอิง')
    write_image(image_path,image)
    data=dict(kind=kind,points=np.asarray(points).tolist(),image='reference.png',
              image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
    if kind in TOTAL_PINS:data['total_pins']=TOTAL_PINS[kind]
    if extra:data.update(extra)
    (folder/'calibration.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    active=Path(root)/kind/'active.json';temp=active.with_suffix('.tmp')
    active_data=dict(data,image=(Path(folder.name)/'reference.png').as_posix())
    temp.write_text(json.dumps(active_data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(active)
    return dict(data,image=str(image_path.resolve()))


def read_calibrations(root):
    pins={};diode=None;issues=[]
    for kind in (*TOTAL_PINS,'diode'):
        active=Path(root)/kind/'active.json'
        if not active.is_file():continue
        try:
            data=load_calibration_profile(active)
            if kind in TOTAL_PINS and data.get('total_pins',16 if kind=='U1' else 28)!=TOTAL_PINS[kind]:
                raise ValueError('จำนวนขาอ้างอิงเดิมไม่ตรง ต้องตั้งขา 1 ใหม่')
            image=read_image(data['image'])
            if kind in TOTAL_PINS:pins[kind]=PinCalibration(kind,image,data['points'],data['image'])
            else:
                from diode_cv import DiodeDirection
                diode=DiodeDirection(image,data['points'],data['image'])
        except (ValueError,OSError,KeyError,TypeError) as exc:issues.append(f'{kind}: {exc}')
    return pins,diode,issues
