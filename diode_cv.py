"""Glass-diode cathode-band inspection; electrical direction needs a board reference."""
from pathlib import Path
import cv2
import numpy as np

from frame_registration import FrameRegistration
from scanner import transform


def model_axis(quad):
    points=np.float32(quad).reshape(4,2)
    edges=np.roll(points,-1,axis=0)-points
    edge=edges[np.argmax(np.linalg.norm(edges,axis=1))]
    length=float(np.linalg.norm(edge))
    if length<1 or not np.isfinite(length):raise ValueError('กรอบไดโอดไม่ถูกต้อง')
    axis=edge/length
    return axis if axis[np.argmax(np.abs(axis))]>=0 else -axis


class FixedDiodeROI:
    """One body size per camera resolution, anchored to the locator OBB."""
    def __init__(self):
        self.resolution=None
        self.size=None
        self.offsets={}

    def quad(self,frame,model_quad,remember=True,pad_quads=(),component=None):
        points=np.float32(model_quad).reshape(4,2)
        u=model_axis(points);v=np.array([-u[1],u[0]])
        a,b=(points@u).min(),(points@u).max()
        c,d=(points@v).min(),(points@v).max()
        resolution=frame.shape[:2]
        if resolution!=self.resolution:
            self.resolution=resolution;self.size=None;self.offsets={}
        offset=self.offsets.get(component,(.52,.72))
        suggested_size=(float((b-a)*.54),float((d-c)*.22))
        if len(pad_quads)==2:
            pads=sorted([np.float32(p) for p in pad_quads],key=lambda p:float((p@u).mean()))
            left,right=(pads[0]@u).max(),(pads[1]@u).min()
            pad_y=[float((p@v).mean()) for p in pads]
            if right-left>(b-a)*.20 and abs(pad_y[1]-pad_y[0])<(d-c)*.20:
                offset=(float(((left+right)/2-a)/(b-a)),float((np.mean(pad_y)-c)/(d-c)))
                suggested_size=(float((right-left)*.94),float(np.mean([np.ptp(p@v) for p in pads])*.60))
                if remember:self.offsets[component]=offset
        size=self.size or suggested_size
        if remember and self.size is None:self.size=size
        center=u*(a+(b-a)*offset[0])+v*(c+(d-c)*offset[1])
        half_width,half_height=size[0]/2,size[1]/2
        return np.float32([center-u*half_width-v*half_height,center+u*half_width-v*half_height,
                           center+u*half_width+v*half_height,center-u*half_width+v*half_height])


class DiodeDirection:
    def __init__(self,image,points,path):
        self.points=np.float32(points).reshape(2,2)
        if not np.isfinite(self.points).all() or np.linalg.norm(self.points[1]-self.points[0])<15:
            raise ValueError('ตั้งจุดขั้ว + และขั้ว - ให้ห่างกันชัดเจน')
        self.path=Path(path)
        self.tracker=FrameRegistration(image)
        self.detail='พร้อมตรวจขั้วไดโอดด้วย OpenCV'

    def axis(self,features):
        match=self.tracker.locate(features)
        self.detail=self.tracker.detail
        if match is None:return None
        p=transform(self.points,match.matrix)
        axis=p[1]-p[0]
        return axis/np.linalg.norm(axis)


def glass_axis(frame,quad):
    """Find the body axis for visual band location only, never electrical polarity."""
    q=np.float32(quad);h,w=frame.shape[:2]
    x1,y1=np.maximum(0,np.floor(q.min(axis=0))).astype(int)
    x2,y2=np.minimum([w,h],np.ceil(q.max(axis=0))+1).astype(int)
    if x2<=x1 or y2<=y1:return None
    roi=frame[y1:y2,x1:x2]
    mask=np.zeros(roi.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.int32(q-[x1,y1]),255)
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    red=(((hsv[:,:,0]<25)|(hsv[:,:,0]>160))&(hsv[:,:,1]>70)&(hsv[:,:,2]>35)&(mask>0))
    y,x=np.where(red)
    if len(x)<30:return None
    points=np.column_stack((x,y)).astype(np.float32)
    _,vectors,values=cv2.PCACompute2(points,mean=None)
    if len(values)<2 or values[0,0]<2*max(1.,values[1,0]):return None
    axis=vectors[0]
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    return axis


def inspect_diode(frame,quad,axis,allow_printed_reference=True,body_quad=None):
    result=dict(status='REVIEW',reason='ยังไม่ได้ยืนยันทิศขั้วไดโอดจากบอร์ด',confidence=0.,band_quad=None,source='opencv_dark_band')
    reference_known=axis is not None
    if axis is None:axis=model_axis(body_quad) if body_quad is not None else glass_axis(frame,quad)
    if axis is None:
        result['reason']='ยังแยกแนวตัวไดโอดแก้วไม่ได้';return result
    if not reference_known and allow_printed_reference:
        from board_polarity import printed_polarity
        markers=printed_polarity(frame,quad,axis)
        if markers is not None:
            axis=markers['axis'];reference_known=True
            result['polarity_reference']=markers
    u=np.asarray(axis,dtype=float)
    if not np.isfinite(u).all() or np.linalg.norm(u)<1e-8:return result
    if body_quad is not None:
        direction=model_axis(body_quad)
        u=direction if direction@u>=0 else -direction
    u=u/np.linalg.norm(u);v=np.array([-u[1],u[0]])
    q=np.asarray(body_quad if body_quad is not None else quad,dtype=float).reshape(4,2)
    if body_quad is not None:
        h,w=frame.shape[:2]
        if not np.isfinite(q).all() or (q.min(axis=0)<0).any() or (q.max(axis=0)>[w-1,h-1]).any():
            result['reason']='กรอบตัวไดโอดยังเห็นไม่ครบในภาพ';return result
        if any(cv2.pointPolygonTest(np.float32(quad),tuple(map(float,p)),True)<-1 for p in q):
            result['reason']='กรอบตัวไดโอดอยู่นอกบริเวณอุปกรณ์ที่โมเดลระบุ';return result
    along=q@u;across=q@v
    a,b=along.min(),along.max();c,d=across.min(),across.max()
    width,height=max(1,round(b-a)),max(1,round(d-c))
    result['expected_cathode_direction']=u.tolist() if reference_known else None
    if width<25 or height<8 or width*height>frame.shape[0]*frame.shape[1]*2:
        result['reason']='บริเวณไดโอดเล็กหรือไม่ชัดพอ';return result
    matrix=np.array([[u[0],u[1],-a],[v[0],v[1],-c]],dtype=np.float32)
    roi=cv2.warpAffine(frame,matrix,(width,height))
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # White silkscreen and blue PCB are excluded before finding the glass body.
    red=((hsv[:,:,0]<25)|(hsv[:,:,0]>160))&(hsv[:,:,1]>70)&(hsv[:,:,2]>35)
    if body_quad is not None:
        x1,y1,x2,y2=0,0,width,height
        result.update(body_roi_source='model_fixed',body_size_px=[width,height],
                      fixed_body_quad=np.asarray(body_quad).tolist())
    else:
        ys,xs=np.where(red)
        if len(xs)<max(30,width*height*.008):
            result['reason']='ยังแยกตัวไดโอดแก้วจากแสงสะท้อนไม่ได้';return result
        x1,x2=np.quantile(xs,[.01,.99]);y1,y2=np.quantile(ys,[.10,.90])
        if x2-x1<20 or y2-y1<4 or (x2-x1)/(y2-y1)<1.3:
            result['reason']='รูปร่างตัวไดโอดไม่ชัด';return result
        x1=max(0,int(x1));x2=min(width,int(x2)+1)
        y1=max(0,int(y1));y2=min(height,int(y2)+1)
        metal=(hsv[y1:y2,:,1]<100)&(hsv[y1:y2,:,2]>170)
        metal_columns=np.flatnonzero(np.mean(metal,axis=0)>.5)
        red_left,red_right=np.quantile(xs,[.01,.99])
        left_terminal=metal_columns[metal_columns<red_left]
        right_terminal=metal_columns[metal_columns>red_right]
        if len(left_terminal):x1=int(left_terminal[-1])+1
        if len(right_terminal):x2=int(right_terminal[0])
    value=hsv[y1:y2,x1:x2,2].astype(float)
    if min(value.shape)<4:
        result['reason']='ตัวไดโอดเล็กเกินไป';return result
    inverse=cv2.invertAffineTransform(matrix)
    body=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    result.update(body_quad=cv2.transform(body.reshape(-1,1,2),inverse).reshape(4,2).tolist(),
                  dark_threshold=90,score_is_probability=False)
    glass=red[y1:y2,x1:x2] & (hsv[y1:y2,x1:x2,1]>175) & (value>45)
    if np.count_nonzero(glass)<20 and body_quad is None:
        result['reason']='แยกเนื้อแก้วข้างแถบดำไม่ได้';return result
    # Set darkness relative to the surrounding glass so faded bands remain
    # visible. Use one mask for every region, then compare their pixel areas.
    threshold=max(90.,min(205.,float(np.median(value[glass]))*.90)) if np.any(glass) else 90.
    mask=value<threshold
    dark=np.mean(mask,axis=0)
    length=value.shape[1]
    smoothing=max(1,int(length*.018))
    profile=np.convolve(dark,np.ones(smoothing)/smoothing,mode='same')
    boundaries=np.diff(np.r_[False,profile>.50,False].astype(int))
    candidates=[]
    for start,end in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
        area=int(np.count_nonzero(mask[:,start:end]))
        if end-start>=max(2,length*.015) and area>=max(6,value.size*.01):
            candidates.append((area,int(start),int(end)))
    if not candidates:
        result['reason']='ยังไม่พบบริเวณมืดบนตัวไดโอด';return result
    area,start,end=max(candidates,key=lambda item:(item[0],item[2]-item[1]))
    position=(start+end)/(2*length)
    score=area/(value.shape[0]*(end-start))
    result.update(dark_threshold=float(threshold),dark_area_px=area,
                  dark_region_count=len(candidates),decision_policy='largest_dark_region',
                  band_position_reference='body_roi_extent')
    band=np.float32([[x1+start,y1],[x1+end,y1],[x1+end,y2],[x1+start,y2]])
    peak=start+int(np.argmax(profile[start:end]))
    peak_line=np.float32([[x1+peak,y1],[x1+peak,y2]])
    direction=u*(1 if position>.5 else -1)
    side=('RIGHT' if direction[0]>0 else 'LEFT') if abs(direction[0])>=abs(direction[1]) else ('DOWN' if direction[1]>0 else 'UP')
    if .4<=position<=.6:side='CENTER'
    result.update(confidence=float(score),band_position=float(position),
                  band_width_px=int(end-start),band_width_fraction=float((end-start)/length),
                  band_quad=cv2.transform(band.reshape(-1,1,2),inverse).reshape(4,2).tolist(),
                  peak_line=cv2.transform(peak_line.reshape(-1,1,2),inverse).reshape(2,2).tolist(),
                  peak_dark_fraction=float(dark[peak]),band_side_image=side)
    if not reference_known:
        result['reason']='พบแถบขั้วลบด้าน'+{'RIGHT':'ขวา','LEFT':'ซ้าย','UP':'บน','DOWN':'ล่าง','CENTER':'กลาง'}[side]+'ของภาพ · ยังไม่ยืนยันทิศขั้วบนบอร์ด'
        return result
    if position>.60:result.update(status='PASS',reason='แถบขั้วลบอยู่ด้านที่ถูกต้อง')
    elif position<.40:result.update(status='FAIL',reason='ไดโอดกลับด้าน · แถบขั้วลบอยู่ผิดฝั่ง')
    else:result['reason']='แถบมืดอยู่ใกล้กลางตัว ยังยืนยันขั้วไม่ได้'
    return result


def draw_diode(image,item):
    """Cyan body, green cathode candidate, yellow dark-profile peak."""
    for key,color in [('body_quad',(255,220,0)),('band_quad',(0,220,0))]:
        if item.get(key) is not None:cv2.polylines(image,[np.int32(item[key])],True,color,2)
    if item.get('peak_line') is not None:
        p=np.int32(item['peak_line']);cv2.line(image,tuple(p[0]),tuple(p[1]),(0,255,255),1)
    quad=np.float32(item.get('quad',item.get('body_quad',[[0,0]])))
    x,y=np.maximum(0,quad.min(axis=0)).astype(int)
    side=item.get('band_side_image','?')
    title={'PASS':'POLARITY OK','FAIL':'REVERSED','REVIEW':'POLARITY ?','NOT_APPLICABLE':'NO DIODE'}[item['status']]
    color=(50,220,80) if item['status']=='PASS' else ((40,60,240) if item['status']=='FAIL' else (30,180,240))
    width=item.get('band_width_fraction')
    width_text=f' {width:.0%}' if width is not None else ''
    cv2.putText(image,item.get('component','')+' BAND '+side+width_text+' | '+title,(x,max(18,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    return image


def combine_diode_result(result,polarity):
    """A polarity failure overrides solder GOOD; an unknown direction cannot pass."""
    raw,annotated,detections,state,reason,metadata=result
    metadata['diode_polarity']=polarity
    if polarity['status']=='FAIL':state,reason='FAIL',polarity['reason']
    elif polarity['status']=='NOT_APPLICABLE':
        state,reason='FAIL',polarity['reason']
        metadata.update(component_missing=True,confirmed_by='component_missing_detection')
    elif polarity['status']=='REVIEW' and state!='FAIL':
        state,reason='REVIEW',polarity['reason']
        metadata['joint_states']={j:'REVIEW' for j in metadata['joints']}
    draw_diode(annotated,dict(component=metadata.get('component',''),quad=metadata.get('roi_quad',[[0,25]]),**polarity))
    return raw,annotated,detections,state,reason,metadata


def assess_diode(frame,quad,detections,axis,body_quad=None):
    missing=[d for d in detections if d.class_id==3]
    if missing:
        return dict(status='NOT_APPLICABLE',reason='โมเดลพบ missing · ไม่มีตัวไดโอดให้ตรวจขั้ว',
                    confidence=min(d.confidence for d in missing),band_quad=None,source='missing_class')
    return inspect_diode(frame,quad,axis,body_quad=body_quad)
