"""Keep the video live and move the latest inference boxes with short-range optical flow."""
import time
import cv2
import numpy as np

from inspection import draw_detection,detection_from_quad,draw_pin_number,visible_detections,detection_polygon,hidden_by_missing,draw_pending_missing
from scanner import transform
from diode_cv import inspect_diode,draw_diode


class LiveOverlay:
    def __init__(self):
        self.clear()

    def clear(self):
        self.payload=None;self.gray=None;self.points=None;self.at=0
        self.diode_observations=[]

    @staticmethod
    def gray_frame(frame):
        h,w=frame.shape[:2];scale=min(1.,320/max(h,w))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray,(max(1,round(w*scale)),max(1,round(h*scale)))),scale

    def update(self,payload):
        self.payload=payload;self.at=time.monotonic()
        self.gray,self.scale=self.gray_frame(payload['raw'])
        self.points=cv2.goodFeaturesToTrack(self.gray,maxCorners=180,qualityLevel=.02,minDistance=7)

    def matrix(self,frame):
        if self.payload is None or frame.shape!=self.payload['raw'].shape:return None
        gray,scale=self.gray_frame(frame)
        if np.array_equal(gray,self.gray):return np.eye(3)
        if self.points is None or len(self.points)<12 or time.monotonic()-self.at>1:return None
        current,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,
                winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,.03))
        if current is None:return None
        back,status_back,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,current,None,winSize=(21,21),maxLevel=3)
        if back is None:return None
        valid=(status.ravel()>0)&(status_back.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.5)
        if valid.sum()<12:return None
        matrix,inliers=cv2.estimateAffinePartial2D(self.points[valid],current[valid],
                  method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=300)
        if matrix is None or inliers is None or inliers.mean()<.7:return None
        zoom=np.linalg.norm(matrix[:,0])
        if not .75<zoom<1.33 or np.linalg.det(matrix[:,:2])<=0:return None
        matrix=np.vstack((matrix,[0,0,1]))
        s=np.diag([scale,scale,1.])
        return np.linalg.inv(s)@matrix@s

    def draw(self,frame):
        self.diode_observations=[]
        matrix=self.matrix(frame)
        if matrix is None:return frame
        image=frame.copy();payload=self.payload
        detections=visible_detections(payload['detections'])
        missing_quads=[detection_polygon(d) for d in detections if d.class_id==3]
        for marker in payload.get('marker_hits',[]):
            if hidden_by_missing(marker.quad,missing_quads):continue
            quad=transform(marker.quad,matrix)
            cv2.polylines(image,[np.int32(quad)],True,(210,130,230),1)
            x,y=np.maximum(0,quad.min(axis=0)).astype(int)
            cv2.putText(image,marker.component,(x,max(15,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(210,130,230),1)
        for hit in payload.get('locator_hits',[]):
            if hidden_by_missing(hit.quad,missing_quads):continue
            quad=transform(hit.quad,matrix)
            locked=hit.component in payload.get('locked_components',[])
            color=(60,220,80) if locked else (255,210,80)
            cv2.polylines(image,[np.int32(quad)],True,color,2)
            x,y=np.maximum(0,quad.min(axis=0)).astype(int)
            text=hit.component+' PASS LOCKED' if locked else f'{hit.component} {hit.confidence:.2f}'
            cv2.putText(image,text,(x,max(20,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1)
        for d in detections:
            center=tuple(map(float,np.float32(d.polygon).mean(axis=0)))
            containing=[h.component for h in payload.get('locator_hits',[]) if cv2.pointPolygonTest(np.float32(h.quad),center,False)>=0]
            if containing and all(n in payload.get('locked_components',[]) for n in containing):continue
            points=transform(np.float32(d.polygon),matrix)
            draw_detection(image,detection_from_quad(points,d.class_id,d.confidence,d.native_class_id,d.native_class_name))
        for item in payload.get('pending_missing_boxes',[]):
            draw_pending_missing(image,dict(item,quad=transform(np.float32(item['quad']),matrix)))
        for item in payload.get('pin_overlays',[]):
            if hidden_by_missing(item['quad'],missing_quads):continue
            quad=transform(np.float32(item['quad']),matrix)
            color={'PASS':(70,220,80),'FAIL':(40,60,240),'REVIEW':(30,180,240)}[item['state']]
            cv2.polylines(image,[np.int32(quad)],True,color,2)
            draw_pin_number(image,item['component'],item['pin'],quad,color)
        for item in payload.get('diode_overlays',[]):
            quad=transform(np.float32(item['quad']),matrix)
            if item.get('status')=='NOT_APPLICABLE':
                observed=dict(item,quad=quad.tolist())
                self.diode_observations.append(observed)
                draw_diode(image,observed)
                continue
            axis=item.get('expected_cathode_direction')
            if axis is not None:axis=matrix[:2,:2]@np.float32(axis)
            body_quad=item.get('fixed_body_quad')
            if body_quad is not None:body_quad=transform(np.float32(body_quad),matrix)
            observed=inspect_diode(frame,quad,axis,allow_printed_reference=item.get('allow_printed_reference',True),body_quad=body_quad)
            observed.update(component=item['component'],quad=quad.tolist(),locked=item.get('locked',False))
            self.diode_observations.append(observed)
            draw_diode(image,observed)
        return image
