"""Operator anchors on a snapshot; the main camera continues to run."""
import tkinter as tk
from tkinter import ttk,messagebox
import cv2
import numpy as np
from PIL import Image,ImageTk


class PointCalibrationWindow:
    def __init__(self,app,kind):
        if app.frame is None:
            app.error('เชื่อมต่อกล้องหรือเปิดภาพก่อนตั้งจุดอ้างอิง');return
        self.app=app;self.kind=kind;self.points=[]
        payload=app.live_overlay.payload
        self.image=(payload['raw'] if payload is not None else app.frame).copy()
        self.joint_centers=[np.mean(d.polygon,axis=0).tolist() for d in payload['detections'] if d.class_id in (0,2)] if payload is not None else []
        if kind=='diode':
            self.labels=['ขั้ว + บนบอร์ด','ขั้ว - ที่ควรติดตั้ง']
            help_text='คลิกกึ่งกลางแพดขั้ว + แล้วแพดขั้ว - ตามเครื่องหมายบนบอร์ด ไม่ใช่ตามทิศที่ไดโอดติดอยู่'
        else:
            total=14 if kind=='U1' else 28
            self.labels=['ขา 1','ขา 2',f'ขา {total}']
            help_text=f'คลิกกึ่งกลางรอยต่อขา 1 → ขา 2 → ขา {total} ที่อยู่ฝั่งตรงข้ามขา 1\nต้องรู้ทิศขา 1 จากร่อง/จุดบน IC ไม่จำเป็นต้องเห็น IC ทั้งตัว'
        self.window=tk.Toplevel(app.root);self.window.title('ตั้งจุดอ้างอิง '+kind)
        tk.Label(self.window,text=help_text,font=(app.ui_font,11),justify='left',wraplength=900).pack(anchor='w',padx=14,pady=10)
        h,w=self.image.shape[:2]
        self.scale=min(1.,900/w,600/h)
        dw,dh=round(w*self.scale),round(h*self.scale)
        photo=Image.fromarray(cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)).resize((dw,dh),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(photo)
        self.canvas=tk.Canvas(self.window,width=dw,height=dh,highlightthickness=0)
        self.canvas.pack(padx=14);self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.canvas.bind('<Button-1>',self.click)
        self.note=tk.StringVar(value='เลือก '+self.labels[0])
        tk.Label(self.window,textvariable=self.note,font=(app.ui_font,11)).pack(anchor='w',padx=14,pady=8)
        buttons=tk.Frame(self.window);buttons.pack(fill='x',padx=14,pady=(0,12))
        ttk.Button(buttons,text='เลือกใหม่',command=self.clear).pack(side='left')
        self.save_button=ttk.Button(buttons,text='บันทึกจุดอ้างอิง',command=self.save,state='disabled')
        self.save_button.pack(side='right')

    def clear(self):
        self.points=[];self.canvas.delete('mark');self.note.set('เลือก '+self.labels[0]);self.save_button.configure(state='disabled')

    def click(self,event):
        if len(self.points)>=len(self.labels):return
        point=np.array([event.x/self.scale,event.y/self.scale])
        if self.kind!='diode' and self.joint_centers:
            centers=np.float32(self.joint_centers);distances=np.linalg.norm(centers-point,axis=1)
            index=int(np.argmin(distances))
            if distances[index]<20:point=centers[index]
        self.points.append(point.tolist())
        x,y=point*self.scale
        n=len(self.points)-1
        self.canvas.create_oval(x-6,y-6,x+6,y+6,outline='#ff2450',width=3,tags='mark')
        self.canvas.create_text(x+9,y-9,text=self.labels[n],anchor='w',fill='#ff2450',font=(self.app.ui_font,11,'bold'),tags='mark')
        if len(self.points)==len(self.labels):
            self.note.set('ตรวจลำดับจุดแล้วกดบันทึก');self.save_button.configure(state='normal')
        else:self.note.set('เลือก '+self.labels[len(self.points)])

    def save(self):
        try:self.app.install_calibration(self.kind,self.image,self.points,self.joint_centers)
        except Exception as exc:
            messagebox.showerror('จุดอ้างอิงไม่ถูกต้อง',str(exc),parent=self.window);return
        self.window.destroy()
