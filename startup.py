"""Visible, asynchronous startup diagnostics."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk
from branding import ResponsiveLogo
from runtime import probe_system
from typography import configure_typography, fit_single_line, LATIN_FONT

# Deliberate presentation pacing; diagnostics themselves remain asynchronous.
STEP_HOLD_MS = 750
RESULT_HOLD_MS = 250
FINAL_HOLD_MS = 1800

PROJECT_TITLE_TH = ('ระบบตรวจสอบคุณภาพรอยบัดกรีบนแผงวงจร PCB ชนิด SMD '
                    'ด้วยการประมวลผลและปัญญาดิษฐ์')
PROJECT_TITLE_EN = ('AI-Based and Image Processing System for Quality Inspection of Solder Joints '
                    'on Surface-Mount Device (SMD) Printed Circuit Boards')


class Startup:
    def __init__(self, root, on_ready):
        self.root, self.on_ready = root, on_ready
        self.ui_font = configure_typography(root)
        self.events = queue.Queue()
        self.closed = False
        self.ready = False
        self.auto_timer = None
        root.title('ตรวจสอบระบบ — ระบบตรวจสอบคุณภาพรอยบัดกรีบนแผงวงจร PCB ชนิด SMD')
        root.geometry('1040x890')
        root.configure(bg='#edf2f5')
        self.frame = tk.Frame(root, bg='#ffffff', padx=36, pady=26)
        self.frame.pack(fill='both', expand=True, padx=22, pady=22)
        self.logo = ResponsiveLogo(self.frame,'white',370,padding=(0,6),rounded=False)
        self.logo.pack(anchor='w',pady=(0,24))
        self.logo.follow_width(self.frame,.40,280,440,inset=72)
        self.title_label = tk.Label(self.frame, text=PROJECT_TITLE_TH, bg='#ffffff',
                                    fg='#102333', anchor='w')
        fit_single_line(self.title_label, self.ui_font, 22, bold=True)
        self.title_label.pack(fill='x')
        english_title = tk.Label(self.frame, text=PROJECT_TITLE_EN, font=(LATIN_FONT, 11), bg='#ffffff',
                                fg='#647b8a', justify='left', anchor='w')
        english_title.pack(fill='x', pady=(8, 8))
        english_title.bind('<Configure>', lambda event: english_title.configure(wraplength=max(100,event.width)))
        tk.Label(self.frame, text='กำลังเตรียมสถานีตรวจสอบรอยบัดกรี', font=(self.ui_font, 13), bg='#ffffff', fg='#647b8a').pack(anchor='w', pady=(3, 20))
        self.rows = {}
        for name in ('CPU', 'RAM', 'GPU', 'CUDA / PyTorch', 'Libraries', 'Storage', 'best_inspec.pt', 'best_locator.pt'):
            row = tk.Frame(self.frame, bg='#f3f6f8', padx=12, pady=8)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=name, width=16, anchor='w', font=(LATIN_FONT, 10, 'bold'), bg='#f3f6f8', fg='#23394b').pack(side='left')
            value = tk.Label(row, text='รอตรวจสอบ', anchor='w', justify='left', wraplength=710, font=(self.ui_font, 10), bg='#f3f6f8', fg='#647b8a')
            value.pack(side='left', fill='x', expand=True)
            self.rows[name] = value
        self.check_states = {name: 'pending' for name in self.rows}
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('Startup.Horizontal.TProgressbar', background='#087f73', troughcolor='#e7eff2', borderwidth=0)
        self.progress = ttk.Progressbar(self.frame, maximum=len(self.rows), mode='determinate', style='Startup.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(18, 10))
        self.note = tk.StringVar(value='เมื่อเข้าหน้าตรวจ โปรแกรมจะเชื่อมต่อ USB 2.0 Camera อัตโนมัติ')
        tk.Label(self.frame, textvariable=self.note, bg='#ffffff', fg='#647b8a', font=(self.ui_font, 10),
                 wraplength=900, justify='left').pack(anchor='w')
        self.enter = tk.Button(self.frame, text='เข้าสู่หน้าตรวจจับ', command=self.finish, state='disabled',
                               bg='#087f73', fg='white', relief='flat', font=(self.ui_font, 11, 'bold'), padx=20, pady=10)
        self.enter.pack(anchor='e', pady=(14, 0))
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(60, self.poll)
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        try:
            result = probe_system(Path(__file__).resolve().parent,
                                  lambda *row: self.events.put(('row', row)))
            self.events.put(('done', result))
        except Exception as exc:
            self.events.put(('fatal', str(exc)))

    def poll(self):
        # Present one real diagnostic event at a time, allowing each row to be read.
        delay = 60
        try:
            kind, data = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == 'row':
                name, state, detail = data
                self.check_states[name] = state
                self.rows[name].configure(text=detail, fg={'ok': '#087f73', 'warning': '#a7701d',
                                                         'error': '#c83249', 'checking': '#53798b'}.get(state, '#c83249'))
                if state == 'checking':
                    delay = STEP_HOLD_MS
                else:
                    self.progress['value'] += 1
                    delay = RESULT_HOLD_MS
            elif kind == 'done':
                self.complete(data)
            else:
                self.complete(dict(device='cpu', device_name='CPU', model=None, signature=None),
                              failure=str(data))
        if not self.closed:
            self.timer = self.root.after(delay, self.poll)

    def complete(self, payload, failure=None):
        self.payload = payload
        self.ready = True
        self.enter.configure(state='normal')
        if self.auto_timer is not None:
            self.root.after_cancel(self.auto_timer)
            self.auto_timer = None
        if failure:
            self.note.set('ตรวจสอบไม่สำเร็จ: ' + failure + ' · กดเข้าสู่หน้าตรวจจับเอง')
        elif all(state == 'ok' for state in self.check_states.values()):
            self.note.set('ผ่านการตรวจสอบทั้งหมด · กำลังเข้าสู่หน้าตรวจจับอัตโนมัติ…')
            self.auto_timer = self.root.after(FINAL_HOLD_MS, self.finish)
        else:
            self.note.set('มีคำเตือนหรือข้อผิดพลาด · อ่านรายละเอียด แล้วกดเข้าสู่หน้าตรวจจับเอง')

    def finish(self):
        if self.closed or not self.ready:
            return
        self.closed = True
        self.root.after_cancel(self.timer)
        if self.auto_timer is not None:
            self.root.after_cancel(self.auto_timer)
            self.auto_timer = None
        self.frame.destroy()
        self.on_ready(self.payload)

    def close(self):
        self.closed = True
        self.root.after_cancel(self.timer)
        if self.auto_timer is not None:
            self.root.after_cancel(self.auto_timer)
        self.root.destroy()
