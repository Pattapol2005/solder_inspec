"""Conservative planar tracking for calibrated board locations, not frame counting."""
from dataclasses import dataclass
import threading
import cv2
import numpy as np


@dataclass
class Match:
    matrix: np.ndarray
    inliers: int
    ratio: float
    error: float


class FeatureFrame:
    def __init__(self, image):
        h,w=image.shape[:2]
        self.shape=(h,w)
        self.scale=min(1.,640/max(h,w))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.gray=cv2.resize(gray,(max(1,round(w*self.scale)),max(1,round(h*self.scale))))
        keypoints,self.descriptors=cv2.ORB_create(nfeatures=1800,fastThreshold=12).detectAndCompute(self.gray,None)
        self.points=np.float32([k.pt for k in keypoints]).reshape(-1,2)/self.scale


class FrameRegistration:
    def __init__(self, image):
        self.anchor=image if isinstance(image,FeatureFrame) else FeatureFrame(image)
        self.bank=[(self.anchor,np.eye(3))]
        self.last=None
        self.detail='พร้อมติดตามตำแหน่ง'
        self.lock=threading.Lock()

    def locate(self, current):
        with self.lock:return self._locate(current)

    def _locate(self,current):
        if isinstance(current,np.ndarray):current=FeatureFrame(current)
        # Prefer the original anchor whenever it is visible, avoiding drift.
        candidates=[self.bank[0],*reversed(self.bank[-3:]),self.bank[len(self.bank)//2]]
        entries=[];seen=set()
        for entry in candidates:
            if id(entry[0]) not in seen:entries.append(entry);seen.add(id(entry[0]))
        for reference,anchor_to_reference in entries:
            if reference.gray.shape==current.gray.shape and np.array_equal(reference.gray,current.gray):
                found=Match(anchor_to_reference.copy(),len(reference.points),1.,0.)
            else:
                found=self._match(reference,current,anchor_to_reference)
            if found is not None:
                self.last=found
                self.detail=f'ติดตามตำแหน่ง {found.inliers} จุด'
                if not self.bank or self._needs_keyframe(found.matrix,current):
                    self.bank.append((current,found.matrix.copy()))
                    if len(self.bank)>15:self.bank.pop(1)  # keep original anchor
                return found
        self.last=None
        self.detail='ตำแหน่งอ้างอิงหาย · เลื่อนกลับบริเวณเดิม หรือตั้งอ้างอิงใหม่'
        return None

    def _needs_keyframe(self, matrix, current):
        previous=self.bank[-1][1]
        try:relative=matrix@np.linalg.inv(previous)
        except np.linalg.LinAlgError:return False
        angle=abs(float(np.arctan2(relative[1,0],relative[0,0])))
        scale=float(np.linalg.norm(relative[:2,0]))
        return (np.linalg.norm(relative[:2,2])>min(current.shape)*.12 or
                angle>np.deg2rad(8) or not .9<scale<1.1)

    @staticmethod
    def _match(reference,current,anchor_to_reference):
        if reference.descriptors is None or current.descriptors is None or min(len(reference.points),len(current.points))<12:return None
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(reference.descriptors,current.descriptors,k=2)
        good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.70*pair[1].distance and pair[0].distance<60]
        if len(good)<12:return None
        # Unique target features prevent repeated pins from voting multiple times.
        used=set();unique=[]
        for m in sorted(good,key=lambda m:m.distance):
            if m.trainIdx not in used:unique.append(m);used.add(m.trainIdx)
        if len(unique)<12:return None
        src=np.float32([reference.points[m.queryIdx] for m in unique])
        dst=np.float32([current.points[m.trainIdx] for m in unique])
        matrix,mask=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
                     ransacReprojThreshold=3/max(current.scale,.2),maxIters=2000,confidence=.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all():return None
        valid=mask.ravel().astype(bool);count=int(valid.sum());ratio=count/len(unique)
        if count<12 or ratio<.65:return None
        if cv2.contourArea(cv2.convexHull(src[valid]))<reference.shape[0]*reference.shape[1]*.015:return None
        transform=np.vstack((matrix,[0,0,1]))
        scale=np.linalg.norm(matrix[:,0])
        if not .65<scale<1.55 or np.linalg.det(matrix[:,:2])<=0:return None
        projected=cv2.transform(src[valid].reshape(-1,1,2),matrix).reshape(-1,2)
        error=float(np.median(np.linalg.norm(projected-dst[valid],axis=1)))
        if error>2.5/max(current.scale,.2):return None
        return Match(transform@anchor_to_reference,count,ratio,error)
