"""Conservative paired + / - silkscreen matching inside one diode footprint."""
from functools import lru_cache
import cv2
import numpy as np


@lru_cache(maxsize=96)
def _template(size,stroke,plus):
    image=np.zeros((size,size),np.float32)
    a=round(size*(.5-stroke/2));b=size-a;m=round(size*.12)
    image[a:b,m:size-m]=1
    if plus:image[m:size-m,a:b]=1
    return image


def printed_polarity(frame,quad,body_axis):
    """Require both signs, matched size/alignment, and no opposite interpretation.

    A body stripe, a single dash, or the camera's horizontal axis alone cannot
    establish electrical polarity. Thresholds are shape scores, not probabilities.
    """
    u=np.asarray(body_axis,dtype=np.float32);u=u/np.linalg.norm(u)
    v=np.array([-u[1],u[0]]);q=np.float32(quad)
    a,b=(q@u).min(),(q@u).max();c,d=(q@v).min(),(q@v).max()
    if b-a<80 or d-c<45:return None
    scale=min(1.,400/(b-a))
    matrix=np.float32([[*u,-a],[*v,-c]])*scale
    roi=cv2.warpAffine(frame,matrix,(max(1,round((b-a)*scale)),max(1,round((d-c)*scale))))
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    binary=((hsv[:,:,2]>220)&(hsv[:,:,1]<110)).astype(np.float32)
    H,W=binary.shape;interpretations=[]
    for y0,y1 in [(0,int(H*.46)),(int(H*.54),H)]:
        best={}
        for side,(x0,x1) in enumerate([(0,int(W*.38)),(int(W*.62),W)]):
            region=binary[y0:y1,x0:x1]
            for plus in (False,True):
                candidate=None
                for fraction in (.12,.15,.18,.21,.24,.27):
                    size=max(7,round(W*fraction))
                    if size>min(region.shape):continue
                    for stroke in (.22,.30,.38):
                        response=cv2.matchTemplate(region,_template(size,stroke,plus),cv2.TM_CCOEFF_NORMED)
                        _,score,_,(x,y)=cv2.minMaxLoc(response)
                        if candidate is None or score>candidate['score']:
                            candidate=dict(score=score,size=size,center=np.float32([x0+x+size/2,y0+y+size/2]))
                best[(side,plus)]=candidate
        for plus_side in (0,1):
            plus=best[(plus_side,True)];minus=best[(1-plus_side,False)]
            if plus is None or minus is None or min(plus['score'],minus['score'])<.73:continue
            if not .70<plus['size']/minus['size']<1.43:continue
            if abs(plus['center'][1]-minus['center'][1])>max(plus['size'],minus['size'])*.30:continue
            # The same shapes must prefer the claimed signs over their opposites.
            if plus['score']-best[(plus_side,False)]['score']<.10:continue
            if minus['score']-best[(1-plus_side,True)]['score']<.10:continue
            interpretations.append((plus_side,plus,minus))
    if not interpretations or len({row[0] for row in interpretations})!=1:return None
    side,plus,minus=max(interpretations,key=lambda r:min(r[1]['score'],r[2]['score']))
    inverse=cv2.invertAffineTransform(matrix)
    centers=cv2.transform(np.float32([plus['center'],minus['center']]).reshape(-1,1,2),inverse).reshape(2,2)
    return dict(axis=(u if side==0 else -u).tolist(),source='printed_plus_minus_cv',
                plus_center=centers[0].tolist(),minus_center=centers[1].tolist(),
                plus_score=float(plus['score']),minus_score=float(minus['score']))
