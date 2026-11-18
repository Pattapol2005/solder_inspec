"""Use trained component designators directly, without requiring OCR or a full board."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import time

import cv2
import numpy as np

from board import COMPONENTS
from inspection import prediction_quads
from scanner import Registration, ScanTarget
from runtime import inference_options
from locator_markers import normalize_name
from live_overlay import LiveOverlay


def quad_iou(a,b):
    intersection,_=cv2.intersectConvexConvex(np.float32(a),np.float32(b))
    union=cv2.contourArea(np.float32(a))+cv2.contourArea(np.float32(b))-intersection
    return intersection/union if union>0 else 0.


class TrainedLocator:
    ACQUIRE_SECONDS=0

    def __init__(self, model, mapping, path, device='cpu', threshold=.60):
        self.model, self.mapping = model, mapping
        self.path, self.device, self.threshold = Path(path), device, threshold
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.hits, self.candidates = [], []
        self.markers=[]
        self.motion=LiveOverlay()
        self.pending={}
        self.clock=time.monotonic
        self.last_threshold=threshold
        self.detail = 'ระบุอุปกรณ์ด้วย ' + self.path.name

    def locate(self, frame):
        resolution_changed=self.motion.payload is not None and self.motion.payload['raw'].shape!=frame.shape
        if self.threshold!=self.last_threshold or resolution_changed:
            self.hits=[];self.pending.clear();self.motion.clear()
            self.last_threshold=self.threshold
        matrix=self.motion.matrix(frame)
        prediction = self.model.predict(source=frame, conf=self.threshold, imgsz=1024,
                                        device=self.device, verbose=False,**inference_options(self.device))[0]
        now=self.clock()
        self.hits, self.candidates = [], []
        self.markers=[]
        observed={}
        h,w = frame.shape[:2]
        for quad, class_id, confidence in prediction_quads(prediction):
            if confidence<self.threshold or class_id not in self.mapping:continue
            name=normalize_name(self.mapping[class_id])
            hit=SimpleNamespace(component=name,confidence=confidence,quad=quad,source='YOLO OBB')
            if name not in COMPONENTS:
                if name in ('plus','minus','ic_head'):self.markers.append(hit)
                continue
            observed.setdefault(name,[]).append(hit)
        grouped={};pending={}
        for name,hits in observed.items():
            primary=max(hits,key=lambda hit:hit.confidence)
            previous=self.pending.get(name)
            since=now
            if previous is not None:
                old_quad=previous.quad
                if matrix is not None:
                    old_quad=cv2.perspectiveTransform(np.float32(old_quad).reshape(-1,1,2),matrix).reshape(4,2)
                if quad_iou(old_quad,primary.quad)>=.5:
                    since=previous.since
                    if matrix is not None and len(hits)==1:
                        versions=[np.roll(points,offset,axis=0) for points in
                                  (primary.quad,primary.quad[::-1]) for offset in range(4)]
                        aligned=min(versions,key=lambda points:float(np.mean(np.linalg.norm(points-old_quad,axis=1))))
                        primary.quad=np.float32(aligned*.65+old_quad*.35)
            pending[name]=SimpleNamespace(since=since,quad=primary.quad.copy())
            if now>=since+self.ACQUIRE_SECONDS:
                self.hits.extend(hits)
                grouped[name]=hits
        # Missing observations disappear immediately and restart acquisition.
        self.pending=pending
        self.motion.update(dict(raw=frame))
        ambiguous = {name for name, hits in grouped.items() if len(hits) != 1}
        candidates = [hits[0] for name,hits in grouped.items() if name not in ambiguous]
        # Conflicting component regions must not certify the same solder joints twice.
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                intersection, _ = cv2.intersectConvexConvex(a.quad, b.quad)
                smaller = min(cv2.contourArea(a.quad), cv2.contourArea(b.quad))
                if smaller > 0 and intersection/smaller > .35:
                    ambiguous.update((a.component,b.component))
        for hit in candidates:
            q = hit.quad
            if hit.component in ambiguous or not cv2.isContourConvex(q) or cv2.contourArea(q) < 100:
                continue
            if q[:,0].min() < 2 or q[:,1].min() < 2 or q[:,0].max() > w-2 or q[:,1].max() > h-2:
                continue  # partial components cannot establish complete coverage
            self.candidates.append(hit)
        names = ', '.join(h.component for h in self.candidates)
        self.detail = f'{self.path.name} · {names or "ยังไม่มีอุปกรณ์ที่เห็นครบและมั่นใจเพียงพอ"}'
        if ambiguous:
            self.detail += ' · ชื่อ/กรอบซ้ำ: ' + ', '.join(sorted(ambiguous))
        waiting=len(self.pending)-len(grouped)
        if waiting:self.detail+=f' · รอพบต่อเนื่อง {self.ACQUIRE_SECONDS:.2f} วินาที: {waiting} อุปกรณ์'
        if not self.candidates:
            return None
        return Registration(np.eye(3), len(self.candidates), 1., 0., method='trained_locator_obb',
                            anchors=[{'component':h.component, 'confidence':h.confidence,
                                      'quad':h.quad.tolist(),'source':h.source} for h in self.candidates])

    def targets(self, frame, registration, results, include_completed=False):
        targets = []
        h,w = frame.shape[:2]
        for hit in self.candidates:
            if not include_completed and results.get(hit.component, {}).get('complete', False):
                continue
            # Preserve the camera angle for the OBB quality model. Mask membership
            # is checked in frame coordinates after inference to exclude neighbours.
            x1,y1 = np.floor(hit.quad.min(axis=0)).astype(int)
            x2,y2 = np.ceil(hit.quad.max(axis=0)).astype(int)
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            crop = frame[y1:y2,x1:x2].copy()
            if crop.size == 0:
                continue
            expected = COMPONENTS[hit.component]['expected']
            targets.append(ScanTarget(hit.component, list(range(1,expected+1)), hit.quad,
                                      crop, np.array([[1.,0.,x1],[0.,1.,y1],[0.,0.,1.]]), {}))
        return targets
