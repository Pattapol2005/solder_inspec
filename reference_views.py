"""Calibrate a board reference from microscope views without training a detector."""
from pathlib import Path
import hashlib
import json
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from board import COMPONENTS
from inspection import read_image, write_image
from positioning import nominal_label_center


def align_view(frame, markers, existing=None):
    if len(markers)<3:raise ValueError('ระบุชื่อและกลางตัวหนังสืออย่างน้อย 3 จุดที่ไม่อยู่ในเส้นเดียวกัน')
    if any(name not in COMPONENTS for name in markers):raise ValueError('ชื่ออุปกรณ์ต้องอยู่ในผังบอร์ด')
    points=np.float32(list(markers.values()))
    height,width=frame.shape[:2]
    if points.shape!=(len(markers),2) or not np.isfinite(points).all():raise ValueError('พิกัดอ้างอิงไม่ถูกต้อง')
    if np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height):raise ValueError('จุดอ้างอิงต้องอยู่ในภาพ')
    if cv2.contourArea(cv2.convexHull(points))<width*height*.005:
        raise ValueError('เลือกชื่อที่กระจายทั้งแนวนอนและแนวตั้ง อย่าเลือกเฉพาะแถวเดียว')
    destination=np.float32([nominal_label_center(name) for name in markers])*3
    matrix,inliers=cv2.estimateAffine2D(points,destination,method=cv2.RANSAC,ransacReprojThreshold=12)
    if matrix is None or inliers is None or not inliers.all():raise ValueError('ตำแหน่งชื่อไม่สอดคล้องกัน ตรวจชื่อและจุดที่คลิก')
    linear=matrix[:,:2]
    singular=np.linalg.svd(linear,compute_uv=False)
    if np.linalg.det(linear)<=0 or singular.min()<.03 or singular.max()/singular.min()>2:
        raise ValueError('ทิศหรือสัดส่วนไม่ตรงผัง ตรวจชื่อและทิศบอร์ดอีกครั้ง')
    size=(1293,1341)
    warped=cv2.warpAffine(frame,matrix,size,borderValue=(255,255,255))
    mask=cv2.warpAffine(np.full(frame.shape[:2],255,np.uint8),matrix,size)
    mask=cv2.erode(mask,np.ones((7,7),np.uint8))
    result=existing.copy() if existing is not None else np.full((1341,1293,3),255,np.uint8)
    if result.shape!=(1341,1293,3):raise ValueError('ภาพอ้างอิงเดิมมีขนาดไม่ถูกต้อง')
    result[mask==255]=warped[mask==255]
    return result,matrix


def save_reference_view(path,frame,markers):
    path=Path(path);manifest_path=path.with_suffix('.json')
    previous={};existing=None
    if path.is_file() and manifest_path.is_file():
        previous=json.loads(manifest_path.read_text('utf-8'))
        if previous.get('method') in ('microscope_views','full_board') and previous.get('image_sha256')==hashlib.sha256(path.read_bytes()).hexdigest():
            existing=read_image(path)
    image,matrix=align_view(frame,markers,existing)
    anchors=previous.get('anchors',{}) if existing is not None else {}
    for name in markers:anchors[name]=list(map(float,nominal_label_center(name)))
    manifest=dict(schema=1,method='microscope_views',anchors=anchors,
                  views=(previous.get('views',[]) if existing is not None else [])+
                  [dict(names=list(markers),image_to_reference=matrix.tolist())])
    temp_image=path.with_name(path.stem+'.tmp.png')
    temp_manifest=manifest_path.with_suffix('.tmp')
    write_image(temp_image,image)
    manifest['image_sha256']=hashlib.sha256(temp_image.read_bytes()).hexdigest()
    temp_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
    temp_image.replace(path);temp_manifest.replace(manifest_path)
    return path


class MicroscopeReferenceWindow:
    def __init__(self,app):
        if app.frame is None:
            app.error('เชื่อมต่อกล้องหรือเปิดภาพก่อนตั้งอ้างอิงบริเวณนี้');return
        self.app=app;self.frame=app.frame.copy();self.markers={}
        self.window=tk.Toplevel(app.root)
        self.window.title('ตั้งภาพอ้างอิงจาก microscope · ทำครั้งเดียวต่อบริเวณ')
        self.window.configure(bg='#edf2f5')
        tk.Label(self.window,text='เลือกชื่อ แล้วคลิกกลางตัวหนังสือบนภาพ',bg='#edf2f5',fg='#192c3e',
                 font=(app.ui_font,15,'bold')).pack(anchor='w',padx=20,pady=(16,4))
        tk.Label(self.window,text='ใช้ภาพนิ่งนี้ระบุอย่างน้อย 3 ชื่อ กระจายทั้งแถวและคอลัมน์ เช่น R1, R11, R2\n'
                 'ตั้งครั้งเดียวแล้วสแกนอัตโนมัติได้ · เลื่อนบอร์ดแล้วเพิ่มภาพอ้างอิงบริเวณถัดไปได้',
                 justify='left',bg='#edf2f5',fg='#748497',font=(app.ui_font,10)).pack(anchor='w',padx=20,pady=(0,12))
        body=tk.Frame(self.window,bg='#edf2f5');body.pack(fill='both',expand=True,padx=20)
        image=Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB));image.thumbnail((850,560))
        self.scale=np.array([image.width/self.frame.shape[1],image.height/self.frame.shape[0]])
        self.photo=ImageTk.PhotoImage(image)
        self.canvas=tk.Canvas(body,width=image.width,height=image.height,highlightthickness=0)
        self.canvas.pack(side='left',padx=(0,16));self.canvas.bind('<Button-1>',self.click)
        side=tk.Frame(body,bg='white',padx=12,pady=12);side.pack(side='left',fill='both')
        tk.Label(side,text='ชื่อที่กำลังระบุ',bg='white',font=(app.ui_font,10)).pack(anchor='w')
        self.name=tk.StringVar(value='R1')
        ttk.Combobox(side,textvariable=self.name,values=list(COMPONENTS),state='readonly',width=16).pack(fill='x',pady=8)
        self.rows=tk.Listbox(side,width=24,height=14,font=(app.ui_font,10),bd=0,highlightthickness=0)
        self.rows.pack(fill='both',expand=True)
        tk.Button(side,text='ใช้ชื่อและจุดที่ OCR พบ',command=self.use_ocr,relief='flat',bg='#d8eee9',fg='#087f73').pack(fill='x',pady=(12,5))
        tk.Button(side,text='ลบรายการที่เลือก',command=self.remove,relief='flat',bg='#eef3f6').pack(fill='x')
        actions=tk.Frame(self.window,bg='#edf2f5');actions.pack(fill='x',padx=20,pady=16)
        tk.Button(actions,text='ยกเลิก',command=self.window.destroy,relief='flat',padx=16,pady=8).pack(side='left')
        tk.Button(actions,text='บันทึกภาพอ้างอิงบริเวณนี้',command=self.commit,bg='#087f73',fg='white',
                  relief='flat',padx=16,pady=8).pack(side='right')
        self.draw()

    def click(self,event):
        self.markers[self.name.get()]=(np.array([event.x,event.y])/self.scale).tolist()
        self.draw()

    def use_ocr(self):
        if not self.app.position_hits:
            self.app.error('ยังไม่มีชื่อ OCR ให้ปิดหน้าต่างนี้แล้วกดทดสอบตำแหน่งก่อน หรือคลิกระบุเอง');return
        for hit in self.app.position_hits:
            if hit.component in COMPONENTS and hit.confidence>=.95:self.markers[hit.component]=hit.quad.mean(axis=0).tolist()
        self.draw()

    def remove(self):
        selected=self.rows.curselection()
        if selected:del self.markers[list(self.markers)[selected[0]]];self.draw()

    def draw(self):
        self.canvas.delete('all');self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.rows.delete(0,'end')
        for name,point in self.markers.items():
            x,y=np.array(point)*self.scale
            self.canvas.create_oval(x-5,y-5,x+5,y+5,fill='#087f73',outline='white',width=2)
            self.canvas.create_text(x+9,y-10,text=name,anchor='sw',fill='#43e8cc',font=(self.app.ui_font,11,'bold'))
            self.rows.insert('end',f'{name} · {point[0]:.0f}, {point[1]:.0f}')

    def commit(self):
        if self.app.busy:return
        try:
            path=Path(__file__).parent/'assets'/'board-reference-user.png'
            save_reference_view(path,self.frame,self.markers)
            self.app.set_scan_reference(path)
            self.window.destroy()
            self.app.info.set('บันทึกอ้างอิงบริเวณนี้แล้ว · พื้นที่ที่ยังไม่มีอ้างอิงต้องตั้งเพิ่มก่อนตรวจ')
        except Exception as exc:self.app.error(exc)
