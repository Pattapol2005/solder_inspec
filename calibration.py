"""One-time alignment of a high-resolution board reference to the training board map."""
from pathlib import Path
import hashlib
import json
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from inspection import read_image, write_image


def normalize_reference(image,points):
    points=np.float32(points)
    if points.shape!=(4,2) or not np.isfinite(points).all() or not cv2.isContourConvex(points):
        raise ValueError('เลือกมุมบอร์ดทั้งสี่ตามลำดับ โดยไม่ให้เส้นไขว้กัน')
    h,w=image.shape[:2]
    if (points[:,0].min()<0 or points[:,1].min()<0 or points[:,0].max()>=w or points[:,1].max()>=h or cv2.contourArea(points)<400):
        raise ValueError('มุมบอร์ดต้องอยู่ในภาพและมีพื้นที่เพียงพอ')
    a,b=points[1]-points[0],points[3]-points[0]
    if a[0]*b[1]-a[1]*b[0]<=0:raise ValueError('ลำดับมุมต้องเป็น บนซ้าย บนขวา ล่างขวา ล่างซ้าย ตามแผนที่บอร์ด')
    size=(1293,1341)
    destination=np.float32([[10,14],[409,14],[409,439],[10,439]])*3
    matrix=cv2.getPerspectiveTransform(points,destination)
    return cv2.warpPerspective(image,matrix,size,borderValue=(255,255,255))


class ReferenceWindow:
    def __init__(self,app):
        path=filedialog.askopenfilename(parent=app.root,title='เลือกภาพบอร์ดจริงแบบเห็นทั้งบอร์ดสำหรับตั้งอ้างอิงครั้งแรก',filetypes=[('Images','*.png *.jpg *.jpeg *.bmp')])
        if not path:return
        try:self.image=read_image(path)
        except Exception as exc:app.error(exc);return
        self.app=app;self.points=[]
        self.window=tk.Toplevel(app.root);self.window.title('ตั้งภาพอ้างอิงบอร์ด · ทำครั้งเดียว')
        tk.Label(self.window,text='คลิกมุมบอร์ด: บนซ้าย · บนขวา · ล่างขวา · ล่างซ้าย\nให้ตัวอักษรบนบอร์ดอยู่ในทิศเดียวกับแผนที่ · ไม่ต้องเห็นทั้งบอร์ดขณะสแกน',
                 font=(app.ui_font,11),justify='left').pack(padx=16,pady=12)
        rgb=Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB));rgb.thumbnail((1000,620))
        self.sx=rgb.width/self.image.shape[1];self.sy=rgb.height/self.image.shape[0]
        self.photo=ImageTk.PhotoImage(rgb)
        self.canvas=tk.Canvas(self.window,width=rgb.width,height=rgb.height,highlightthickness=0)
        self.canvas.pack(padx=16)
        self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.canvas.bind('<Button-1>',self.click)
        actions=tk.Frame(self.window);actions.pack(fill='x',padx=16,pady=12)
        tk.Button(actions,text='เริ่มเลือกมุมใหม่',command=self.reset).pack(side='left')
        self.save=tk.Button(actions,text='ใช้ภาพอ้างอิงนี้',command=self.commit,state='disabled')
        self.save.pack(side='right')

    def click(self,event):
        if len(self.points)>=4:return
        self.points.append((event.x/self.sx,event.y/self.sy))
        self.canvas.create_oval(event.x-5,event.y-5,event.x+5,event.y+5,fill='#087f73',outline='white',tags='marks')
        self.canvas.create_text(event.x+9,event.y-9,text=str(len(self.points)),fill='#087f73',font=(self.app.ui_font,14,'bold'),tags='marks')
        if len(self.points)==4:self.save.configure(state='normal')

    def reset(self):
        self.points=[];self.canvas.delete('marks');self.save.configure(state='disabled')

    def commit(self):
        if self.app.busy:return
        try:
            image=normalize_reference(self.image,self.points)
            path=Path(__file__).parent/'assets'/'board-reference-user.png'
            write_image(path,image)
            path.with_suffix('.json').write_text(json.dumps(dict(schema=1,method='full_board',
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),anchors={})),encoding='utf-8')
            self.app.set_scan_reference(path)
            self.window.destroy()
        except Exception as exc:self.app.error(exc)
