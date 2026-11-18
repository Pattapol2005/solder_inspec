"""Focused evidence views; model inference and saved frame coordinates stay unchanged."""
import json
from pathlib import Path
import cv2
import numpy as np
from inspection import read_image,write_image
from board import COMPONENTS


def crop_evidence(image,quad,title='',padding=6):
    points=np.asarray(quad,dtype=np.float32).reshape(-1,2)
    if len(points)<3 or not np.isfinite(points).all():raise ValueError('กรอบหลักฐานไม่ถูกต้อง')
    h,w=image.shape[:2]
    x1,y1=np.maximum(0,np.floor(points.min(axis=0)-padding)).astype(int)
    x2,y2=np.minimum([w,h],np.ceil(points.max(axis=0)+padding)+1).astype(int)
    if x2<=x1 or y2<=y1:raise ValueError('กรอบหลักฐานอยู่นอกภาพ')
    crop=image[y1:y2,x1:x2].copy()
    mask=np.zeros(crop.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.int32(points-[x1,y1]),255)
    mask=cv2.dilate(mask,np.ones((padding*2+1,padding*2+1),np.uint8))
    crop[mask==0]=(16,29,43)
    if title:
        crop=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT,value=(16,29,43))
        cv2.putText(crop,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,240,245),1)
    return crop


def result_evidence(result,pin=None):
    meta=result[5];name=meta.get('component','')
    if pin is not None:
        pads=meta.get('pad_polygons',{})
        quad=pads.get(str(pin),pads.get(int(pin)))
        if quad is None:raise ValueError('หลักฐานนี้ไม่มีตำแหน่งขาที่เลือก')
        states=meta.get('joint_states',{})
        title=f'{name} / {pin}: '+states.get(str(pin),states.get(int(pin),''))
    else:
        quad=meta.get('roi_quad');title=name+': '+result[3]
        if quad is None:return result[1]
    return crop_evidence(result[1],quad,title)


def read_evidence(folder,pin=None):
    folder=Path(folder)
    selected=folder/(f'pin_{int(pin):02d}.png' if pin is not None else 'component_result.png')
    if selected.is_file():return read_image(selected)
    # Older reports can also be cropped without rewriting their originals.
    meta=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    image=read_image(folder/'result.png')
    return result_evidence((None,image,[],meta.get('status',''),'',meta),pin)


def component_detection_image(name,entry,root):
    """Retain the annotated capture and include every saved IC pin capture."""
    reports={}
    if name in ('U1','U4'):
        from ic_evidence import bank_reports
        for bank in bank_reports(name,entry):
            for report,pins in bank:
                reports.setdefault(report,set()).update(pins)
    reports.setdefault(Path(entry['report']),set())
    panels=[]
    for report,pins in reports.items():
        folder=Path(root)/report
        meta=json.loads((folder/'result.json').read_text('utf-8'))
        if meta.get('component',name)!=name:
            raise ValueError('ชื่ออุปกรณ์ในหลักฐานไม่ตรงกับ '+name)
        source=folder/'result.png'
        frame=read_image(source) if source.is_file() else read_evidence(folder)
        height,width=frame.shape[:2]
        scale=min(1.,1024/width,768/height)
        if scale<1:
            frame=cv2.resize(frame,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA)
        caption='Pins '+', '.join(map(str,sorted(pins))) if pins else ''
        panels.append((frame,caption))
    width=max(480,max(frame.shape[1] for frame,_ in panels))
    height=44+sum(frame.shape[0]+(30 if caption else 0)+10 for frame,caption in panels)
    image=np.full((height,width,3),(16,29,43),np.uint8)
    state=entry.get('status','REVIEW')
    color={'PASS':(80,220,80),'FAIL':(60,70,240),'REVIEW':(40,190,240)}.get(state,(230,240,245))
    label='MISSING' if entry.get('component_missing') else state
    cv2.putText(image,f'{name}: {label}',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,color,2,cv2.LINE_AA)
    y=44
    for frame,caption in panels:
        if caption:
            cv2.putText(image,caption,(12,y+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,240,245),1,cv2.LINE_AA)
            y+=30
        x=(width-frame.shape[1])//2
        image[y:y+frame.shape[0],x:x+frame.shape[1]]=frame
        y+=frame.shape[0]+10
    return image


def save_board_images(session,folder,overwrite=False):
    """Also supports adding images to existing portable board archives."""
    folder=Path(folder)
    for name in COMPONENTS:
        target=folder/(name+'.png')
        if target.is_file() and not overwrite:continue
        image=component_detection_image(name,session['results'][name],folder)
        temporary=folder/(name+'.partial.png')
        write_image(temporary,image)
        temporary.replace(target)
