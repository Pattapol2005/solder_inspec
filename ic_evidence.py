"""Browse the source photographs for the two pin banks of an inspected IC."""
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from board import COMPONENTS
from evidence import read_evidence
from inspection import read_image


def bank_reports(name, entry):
    if entry.get('component_missing'):return [[],[]]
    half = COMPONENTS[name]['expected'] // 2
    banks = []
    for start in (1, half + 1):
        reports = {}
        for pin in range(start, start + half):
            item = entry.get('coverage', {}).get(str(pin), {})
            if item.get('confirmed_by') in ('latest_frame_component_count', 'full_component_detection_count'):
                continue
            if item.get('report'):
                reports.setdefault(Path(item['report']), []).append(pin)
        # Open the photo that shows the most pins first.
        banks.append(sorted(reports.items(), key=lambda pair: -len(pair[1])))
    return banks


def bank_photo(folder, name, pins, *, raw=False):
    meta = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if meta.get('component') != name:
        raise ValueError('ชื่ออุปกรณ์ในภาพหลักฐานไม่ตรงกัน')
    pads = meta.get('pad_polygons', {})
    if not pads:
        if raw:raise ValueError('ไม่มีตำแหน่งขาสำหรับภาพช่วงนี้')
        return read_evidence(folder)
    # Use original frame coordinates so two banks sharing a capture get
    # distinct crops. Keep the annotated image and its inspection markings.
    image = read_image(folder / ('raw.png' if raw else 'result.png'))
    points = np.concatenate([np.float32(pads[str(pin)]).reshape(4, 2) for pin in pins])
    if not np.isfinite(points).all():
        raise ValueError('ตำแหน่งขาในหลักฐานไม่ถูกต้อง')
    lo = np.maximum(0, np.floor(points.min(axis=0)).astype(int) - 14)
    hi = np.minimum(image.shape[1::-1], np.ceil(points.max(axis=0)).astype(int) + 15)
    if np.any(hi <= lo):
        raise ValueError('กรอบหลักฐานอยู่นอกภาพ')
    return image[lo[1]:hi[1], lo[0]:hi[0]]


def show_banks(window, name, entry, family):
    pane = tk.Frame(window, bg='#102333')
    pane.pack(side='left', fill='both', expand=True)
    half = COMPONENTS[name]['expected'] // 2
    def section(index, reports):
        card = tk.Frame(pane, bg='#102333')
        card.grid(row=index, column=0, sticky='nsew')
        bar = tk.Frame(card, bg='#edf2f5')
        bar.pack(fill='x')
        title = 'ช่วงบน' if index == 0 else 'ช่วงล่าง'
        tk.Label(bar, text=f'{title} · ขา {1+index*half}–{(index+1)*half}', bg='#edf2f5',
                 font=(family, 11, 'bold')).pack(side='left')
        caption = tk.StringVar()
        tk.Label(card, textvariable=caption, bg='#102333', fg='white').pack(side='bottom', fill='x')
        canvas = tk.Canvas(card, bg='#102333', highlightthickness=0, width=560, height=180)
        canvas.pack(fill='both', expand=True)
        page = 0
        original = None
        error = ''
        def draw(event=None):
            w,h = canvas.winfo_width(),canvas.winfo_height()
            canvas.delete('all')
            if original is None:
                canvas.create_text(w/2,h/2,text=error,fill='white',font=(family,11))
                return
            scale = min(w/original.width,h/original.height,6)
            size = tuple(max(1,round(v*scale)) for v in original.size)
            canvas.photo = ImageTk.PhotoImage(original.resize(size, Image.Resampling.BILINEAR))
            canvas.create_image(w/2,h/2,image=canvas.photo)
        def display(step=0):
            nonlocal page,original,error
            page += step
            original = None
            error = 'ยังไม่มีภาพหลักฐานของช่วงนี้'
            if reports:
                folder,pins = reports[page]
                caption.set(f'ภาพ {page+1}/{len(reports)}   ·   ขา '+', '.join(map(str,pins)))
                try:
                    frame = bank_photo(folder,name,pins)
                    original = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                except (OSError, ValueError, KeyError, cv2.error):
                    error = 'อ่านภาพหลักฐานนี้ไม่ได้'
            prev.configure(state='normal' if page>0 else 'disabled')
            nxt.configure(state='normal' if page+1<len(reports) else 'disabled')
            draw()
        nxt = ttk.Button(bar,text='ถัดไป ›',width=8,command=lambda:display(1))
        prev = ttk.Button(bar,text='‹ ก่อนหน้า',width=8,command=lambda:display(-1))
        if len(reports)>1:
            nxt.pack(side='right');prev.pack(side='right')
        canvas.bind('<Configure>',draw)
        display()
    pane.columnconfigure(0,weight=1)
    for index,reports in enumerate(bank_reports(name,entry)):
        pane.rowconfigure(index,weight=1,uniform='bank')
        section(index,reports)
    width=min(1180,int(window.winfo_screenwidth()*.9))
    height=min(800,int(window.winfo_screenheight()*.83))
    window.geometry(f'{width}x{height}')
