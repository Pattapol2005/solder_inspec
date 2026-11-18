"""Thai desktop interface for USB microscope SMD inspection."""
from pathlib import Path
import queue
import gc
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from ui import build_ui, COLORS
from runtime import ResourceMonitor, load_weights, load_locator_weights, model_file as find_model_file
from camera import Camera, MICROSCOPE_NAME
from board import BoardChecklist, COMPONENTS, crop_input, restore_box, parse_joints
from typography import configure_typography, LATIN_FONT
from auto_scan import AutoScanMixin
from preferences import load_preferences, save_preferences, confidence_value
from workflow import WorkflowMixin
from runtime import inference_options
from inspection import (CLASSES, THAI, REGIONS, Detection, decision, read_image,
                        save_report, validate_names, write_image, prediction_quads,
                        detection_from_quad, draw_detection, quality_detections)

ROOT = Path(__file__).resolve().parent


class App(WorkflowMixin,AutoScanMixin):
    def __init__(self, root, model_file=None, auto_model=True, runtime_info=None, monitor=True, board_store=None, report_root=None, auto_scan=False, reference_file=None, locator_file=None, settings_file=None, auto_camera=False):
        self.root = root
        self.ui_font = configure_typography(root)
        root.title('ระบบตรวจสอบคุณภาพรอยบัดกรีบนแผงวงจร PCB ชนิด SMD | USB Microscope')
        root.geometry('1440x900')
        root.minsize(1080, 760)
        self.camera = None
        self.named_camera = False
        self.frame = None
        self.display_frame = None
        self.frozen = True
        self.model = None
        self.model_path = ''
        self.runtime_info = runtime_info or {}
        self.auto_model_file = model_file is None
        self.auto_discover_locator = auto_model
        self.model_file = Path(model_file or self.runtime_info.get('model_file') or find_model_file(ROOT))
        self.model_signature = None
        self.locator_file = Path(locator_file or self.runtime_info.get('locator_file') or find_model_file(self.model_file.parent, 'locator'))
        self.locator_signature = self.runtime_info.get('locator_signature')
        self.locator_model = self.runtime_info.get('locator_model')
        self.locator_class_map = self.runtime_info.get('locator_class_map', {})
        self.device = self.runtime_info.get('device', 'cpu')
        self.monitor = ResourceMonitor() if monitor else None
        self.watch_id = None
        self.busy = False
        self.events = queue.Queue()
        # Reuse one inference thread: creating a CUDA worker per frame repeatedly
        # pays thread-local backend setup costs (seconds on the target machine).
        # The busy/latest-frame gate below prevents a backlog in this executor.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inspection')
        self.class_map = dict(enumerate(range(len(CLASSES))))
        self.result = None
        self.report_root = Path(report_root) if report_root else ROOT/'logs'
        preferences_root = Path(board_store).parent if board_store else (Path(report_root).parent if report_root else ROOT)
        self.settings_file = Path(settings_file) if settings_file else preferences_root/'settings.json'
        self.preferences, self.preferences_error = load_preferences(self.settings_file)
        self.preferences_timer = None
        self.init_workflow(preferences_root/'calibrations')
        self.report_path = None
        self.roi = None
        self.drag_start = None
        self.view_transform = None
        self.rotation = 0
        self.board_window = None
        self.active_component = 'ไม่ระบุ'
        self.init_auto_scan(auto_scan, reference_file)
        store = Path(board_store) if board_store else ROOT/'logs'/'boards'
        checklist_error = None
        try:self.checklist = BoardChecklist(store)
        except (OSError,ValueError) as exc:
            checklist_error = str(exc)
            self.checklist = BoardChecklist(store.parent/f'boards_recovery_{time.time_ns()}',load_existing=False)
        self.checklist.lock_pass=True
        self._ui()
        self.threshold.trace_add('write',self.confidence_changed)
        self.locator_threshold.trace_add('write',self.confidence_changed)
        self.ic_component.trace_add('write',self.refresh_pin_details)
        self.board_number.trace_add('write',lambda *_:self.refresh_checklist())
        self.change_scan_mode()
        self.refresh_checklist()
        if checklist_error:
            self.info.set('อ่านเช็กลิสต์เดิมไม่ได้ เริ่มชุดใหม่โดยเก็บไฟล์เดิมไว้: '+checklist_error)
        if self.runtime_info.get('model') is not None:
            self.model = self.runtime_info['model']
            self.class_map = self.runtime_info['class_map']
            self.model_signature = self.runtime_info['signature']
            self.model_path = str(self.model_file)
            self.model_label.set('●  AI พร้อมใช้งาน')
            self.model_detail.set(f'{self.model_file.name} · {self.model.task.upper()} · โหลดแล้ว')
            self.inspect_btn.configure(state='normal')
        if self.monitor:
            self.monitor.start()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.tick_id = root.after(16, self.tick)
        if auto_model:
            self.check_model()
        if auto_camera:
            self.connect(by_name=True)

    def _ui(self):
        build_ui(self)

    def confidence_changed(self, *_):
        try:
            values = dict(inspection_confidence=confidence_value(self.threshold.get()),
                          locator_confidence=confidence_value(self.locator_threshold.get()))
        except (ValueError,TypeError):
            return  # permit an incomplete edit; never persist it
        if values == self.preferences:return
        self.preferences = values
        if self.preferences_timer is not None:self.root.after_cancel(self.preferences_timer)
        self.preferences_timer = self.root.after(350,self.persist_preferences)
        self.cancel_scan()
        self.scan_after = 0
        self.live_preview = None
        self.scan_status.set('ปรับ Confidence แล้ว · กำลังตรวจภาพล่าสุด')

    def persist_preferences(self):
        self.preferences_timer = None
        try:save_preferences(self.settings_file,self.preferences)
        except OSError as exc:self.info.set('บันทึกค่า Confidence ไม่สำเร็จ: '+str(exc))

    def refresh_checklist(self):
        values=self.checklist.session['results'].values()
        passed=sum(v.get('status')=='PASS' for v in values)
        failed=sum(v.get('status')=='FAIL' for v in values)
        inspected=self.inspected_count()
        scanned=self.scanned_count()
        self.board_progress.set(f'สแกนแล้ว {scanned}/{len(COMPONENTS)} · ตรวจครบ {inspected} · ผ่าน {passed} · ไม่ผ่าน {failed}')
        if hasattr(self,'board_save_btn'):
            number=self.board_number.get().strip()
            saved=self.board_signature()==self.last_saved_signature and number==self.checklist.session.get('board_number','')
            self.board_save_btn.configure(state='normal' if number and scanned==len(COMPONENTS) and not self.board_saving and not saved else 'disabled',
                                          text='บันทึกบอร์ดแล้ว' if saved else 'บันทึกผลบอร์ดนี้')
        name = self.component.get()
        if name in COMPONENTS:
            item = self.checklist.session['results'].get(name,{})
            state = {'PASS':'ตรวจครบ · ปกติ','FAIL':'พบข้อบกพร่อง','REVIEW':'รอตรวจเพิ่ม'}.get(item.get('status'),'ยังไม่ตรวจ')
            self.component_status.set(name+' · MISSING · ไม่พบตัวอุปกรณ์' if item.get('component_missing') else
                                      f"{name} · {state} · {item.get('covered',0)}/{COMPONENTS[name]['expected']} จุด")
        else:self.component_status.set('รอจับภาพอุปกรณ์อัตโนมัติ' if self.auto_mode.get() else 'เลือกอุปกรณ์เพื่อบันทึกเช็กลิสต์')
        if hasattr(self,'board_layout'):self.board_layout.refresh()
        self.refresh_pin_details()
        self.refresh_diode_details()
        if self.board_window and self.board_window.window.winfo_exists():
            self.board_window.refresh()

    def select_component_info(self,name):
        self.info_component=name
        self.component.set(name)
        if name in ('U1','U4'):self.ic_component.set(name)
        self.refresh_checklist()
        self.show_evidence(name)

    def show_component_evidence(self):
        name=getattr(self,'info_component',None) or self.component.get()
        if name in COMPONENTS:
            if name not in self.checklist.session['results']:
                self.info.set(name+' · ยังไม่มีภาพหลักฐานที่บันทึกไว้')
                return
            self.show_evidence(name)
        else:self.show_evidence()

    def show_board(self):
        from board_ui import BoardWindow
        if self.board_window and self.board_window.window.winfo_exists():
            self.board_window.window.lift()
        else:
            self.board_window = BoardWindow(self)

    def select_component(self, name=None):
        if self.busy:
            self.component.set(self.active_component)
            return
        if self.auto_mode.get():
            self.auto_mode.set(False)
            self.change_scan_mode()
        name = name or self.component.get()
        self.component.set(name)
        self.active_component = name
        self.reset_roi()
        if name in COMPONENTS:
            data = COMPONENTS[name]
            self.region.set(data['region'])
            total = data['expected']
            # For ICs the operator explicitly chooses the pins in the zoomed view.
            self.joints.set('1-'+str(total) if total <= 3 else '')
            self.expected.set(str(total) if total <= 3 else '0')
            self.info.set(f'{name}: ระบุจุด/ขาที่เห็น แล้วลากกรอบรอบจุดตรวจในภาพซูม')
        else:
            self.joints.set('')
            self.expected.set('0')
        self.clear_result()
        self.refresh_checklist()

    def update_joint_count(self, event=None):
        name = self.component.get()
        if name in COMPONENTS:
            try:self.expected.set(str(len(parse_joints(self.joints.get(), COMPONENTS[name]['expected']))))
            except ValueError:self.expected.set('0')

    def reset_roi(self):
        self.roi = self.drag_start = None
        self.rotation = 0
        self.rotation_label.set('หมุนกรอบ 0°')

    def clear_roi(self):
        if self.busy:return
        self.reset_roi()
        self.clear_result()

    def rotate_roi(self):
        if self.busy:return
        self.rotation = (self.rotation + 90) % 360
        self.rotation_label.set(f'หมุนกรอบ {self.rotation}°')
        self.clear_result()

    def preview_crop(self):
        if self.frame is None:return
        try:
            image,_ = crop_input(self.frame,self.roi,self.rotation)
            photo = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            photo.thumbnail((900,650),Image.Resampling.LANCZOS)
            window = tk.Toplevel(self.root)
            window.title(f'{self.component.get()} · ภาพครอบที่ส่ง AI · {self.rotation}°')
            picture = ImageTk.PhotoImage(photo)
            label = tk.Label(window,image=picture,bg=COLORS['navy'])
            label.image = picture
            label.pack()
        except Exception as exc:self.error(exc)

    def canvas_point(self, event):
        if self.frame is None or self.view_transform is None:return None
        ox,oy,sx,sy = self.view_transform
        h,w = self.frame.shape[:2]
        return (max(0,min(w,round((event.x-ox)/sx))), max(0,min(h,round((event.y-oy)/sy))))

    def roi_start(self, event):
        if self.busy or self.auto_mode.get() or not self.roi_mode.get():return
        point = self.canvas_point(event)
        if point is None:return
        self.frozen = True
        self.source_label.set('ภาพนิ่ง · เลือกกรอบตรวจ')
        self.clear_result()
        self.drag_start = point
        self.roi = None

    def roi_drag(self, event):
        if self.busy or self.drag_start is None:return
        point = self.canvas_point(event)
        if point is None:return
        x,y = self.drag_start
        px,py = point
        self.roi = (min(x,px),min(y,py),max(x,px),max(y,py))
        self.render()

    def roi_end(self, event):
        if self.busy or self.drag_start is None:return
        self.roi_drag(event)
        self.drag_start = None
        if self.roi and (self.roi[2]-self.roi[0]<5 or self.roi[3]-self.roi[1]<5):
            self.roi = None
        self.render()
        self.info.set('กรอบสีฟ้าคือบริเวณที่ส่งให้ AI ตรวจ · ระบุจุด/ขาที่เห็นก่อนกดตรวจภาพ')

    def set_result_status(self, title, detail, color=None):
        self.status.set(title)
        self.result_detail.set(detail)
        self.status_widget.configure(fg=color or COLORS['muted'])

    def toggle_overlay(self):
        if self.auto_mode.get() and not self.frozen:
            self.display_frame=self.frame
            self.render()
            return
        if self.result:
            self.display_frame = self.result[1] if self.overlay.get() else self.result[0]
            self.render()

    def error(self, exc):
        messagebox.showerror('SMD Inspector', str(exc), parent=self.root)

    def clear_result(self):
        self.result = None
        self.report_path = None
        self.table.delete(*self.table.get_children())
        self.set_result_status('READY' if self.frame is not None else 'STANDBY', 'กดตรวจภาพเพื่อเริ่มวิเคราะห์' if self.frame is not None else 'เชื่อมต่อกล้อง หรือเปิดไฟล์ภาพ')
        for value in self.counts:
            value.set('—')
        self.latency.set('—')
        self.save_btn.configure(state='disabled')
        self.confirm_btn.configure(state='disabled')
        self.confirm_btn.pack_forget()
        self.display_frame = self.frame
        self.render()

    def connect(self, by_name=False):
        if self.busy and not by_name:
            return
        try:
            index = None if by_name else int(self.cam_index.get())
            if index is not None and index < 0:
                raise ValueError('หมายเลขกล้องต้องไม่ติดลบ')
            width, height = map(int, self.resolution.get().split('x'))
            if self.camera and self.camera.thread.is_alive():
                self.error('หยุดกล้องก่อน แล้วรอให้กล้องปิดก่อนเชื่อมต่อใหม่')
                return
            if by_name:self.backend.set('DirectShow')
            self.named_camera = by_name
            self.camera = Camera(index, width, height, self.backend.get())
            self.cancel_scan()
            self.camera.start()
            self.frozen = False
            self.frame = self.display_frame = None
            self.reset_roi()
            self.clear_result()
            self.info.set('กำลังเปิดกล้อง…')
            self.camera_label.set('●  กำลังเชื่อมต่อ')
            self.render()
        except Exception as exc:
            self.error(exc)

    def disconnect(self):
        self.cancel_scan()
        if self.camera:
            self.camera.stop()
        self.frozen = True
        self.info.set('หยุดกล้องแล้ว')
        self.camera_label.set('●  ไม่ได้เชื่อมต่อ')
        self.source_label.set('ภาพนิ่ง' if self.frame is not None else 'ไม่มีแหล่งภาพ')

    def freeze(self):
        if self.busy and self.auto_mode.get():
            self.cancel_scan()
            self.frozen=True
            return
        if self.busy:
            return
        if not self.camera or not self.camera.thread.is_alive() or self.camera.stop_event.is_set():
            self.error('กรุณาเชื่อมต่อกล้องก่อน')
            return
        if not self.frozen and self.frame is None:
            return
        self.frozen = not self.frozen
        self.cancel_scan()
        self.reset_roi()
        self.source_label.set('ภาพนิ่งจากกล้อง' if self.frozen else 'LIVE · USB MICROSCOPE')
        self.clear_result()
        self.display_frame = self.frame
        self.render()
        self.info.set('ภาพนิ่ง: พร้อมตรวจด้วย AI หรือบันทึกภาพ' if self.frozen else 'ภาพสดจากกล้อง')

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff')])
        if path:
            try:
                image = read_image(path)
                self.frozen = True
                self.cancel_scan()
                self.frame = self.display_frame = image
                self.reset_roi()
                self.clear_result()
                self.source_label.set('FILE · ' + Path(path).name)
                self.info.set(f'{Path(path).name} | {image.shape[1]} × {image.shape[0]} px')
                self.render()
                if self.auto_mode.get():
                    self.scan_snapshot(image.copy())
            except Exception as exc:
                self.error(exc)

    def job(self, operation, success, on_error=None):
        if self.busy:
            return
        self.busy = True
        self.inspect_btn.configure(state='disabled')
        # Release obsolete Tk objects on the UI thread before model allocations
        # can trigger cyclic garbage collection on a worker thread.
        if not self.auto_mode.get() or not self.model:gc.collect()
        def work():
            try:
                self.events.put((success, operation(), None, on_error))
            except Exception as exc:
                self.events.put((success, None, exc, on_error))
        self.worker.submit(work)

    def check_model(self):
        """Discover added/replaced weights automatically, without a file picker."""
        try:
            if self.busy:
                return
            if self.auto_discover_locator and self.check_locator_model():
                return
            if self.auto_model_file:
                self.model_file = find_model_file(ROOT)
            if not self.model_file.is_file():
                if self.model is not None:
                    self.model = None
                    self.clear_result()
                self.model_signature = None
                self.model_label.set('●  รอไฟล์ best_inspec.pt')
                self.model_detail.set('วาง best_inspec.pt (หรือ best.pt) ข้าง app.py')
                self.inspect_btn.configure(state='disabled')
                return
            stat = self.model_file.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            # Wait for file copying to settle before reading weights.
            if time.time() - stat.st_mtime < 2 or signature == self.model_signature:
                return
            self.model_signature = signature
            self.model = None
            self.clear_result()
            self.model_label.set('●  กำลังโหลดโมเดล')
            self.model_detail.set(self.model_file.name+' · กำลังเตรียม AI')
            def load():
                model, mapping = load_weights(self.model_file)
                if hasattr(model,'to'):model.to('cuda:0' if str(self.device)=='0' else self.device)
                return model,mapping
            def loaded(result):
                self.model, self.class_map = result
                self.model_path = str(self.model_file)
                self.model_label.set('●  AI พร้อมใช้งาน')
                self.model_detail.set(self.model_file.name+' · ตรวจรอยบัดกรี 4 ประเภท')
                self.info.set('โหลด '+self.model_file.name+' อัตโนมัติแล้ว พร้อมตรวจภาพ')
                if self.auto_mode.get():self.retry_scan()
            def failed(exc):
                self.model_label.set('●  โหลดโมเดลไม่สำเร็จ')
                self.model_detail.set(str(exc))
                self.info.set('ตรวจสอบ '+self.model_file.name+' แล้วแทนที่ไฟล์ โปรแกรมจะโหลดใหม่อัตโนมัติ')
            self.job(load, loaded, failed)
        except OSError as exc:
            self.model_signature = None
            self.model_label.set('●  อ่านไฟล์โมเดลไม่ได้')
            self.model_detail.set(str(exc))
        finally:
            self.watch_id = self.root.after(1500, self.check_model)

    def check_locator_model(self):
        """Load a new/replaced locator without blocking Tk or changing any weights."""
        path = find_model_file(self.locator_file.parent, 'locator')
        if not path.is_file():
            if self.locator_model is not None:
                self.cancel_scan()
                self.locator_model = self.locator = None
                self.locator_signature = None
                self.position_detail.set('ไม่มีโมเดลตำแหน่ง · ใช้ภาพอ้างอิง / OCR')
            return False
        stat = path.stat()
        signature = (stat.st_mtime_ns,stat.st_size)
        if signature == self.locator_signature or time.time()-stat.st_mtime < 2:
            return False
        self.locator_file = path
        self.locator_signature = signature
        self.position_detail.set('กำลังโหลด '+path.name)
        def load():
            model,mapping = load_locator_weights(path)
            model.to('cuda:0' if str(self.device)=='0' else self.device)
            return model,mapping
        def done(payload):
            self.cancel_scan()
            self.locator_model,self.locator_class_map = payload
            self.locator = None
            self.scan_attempts.clear()
            self.position_detail.set(path.name+' · พร้อมระบุอุปกรณ์ 62 ตำแหน่ง')
            if self.auto_mode.get() and self.model is not None:self.retry_scan()
        def failed(exc):
            self.locator_model = self.locator = None
            self.position_detail.set('โหลด '+path.name+' ไม่สำเร็จ: '+str(exc))
        self.job(load,done,failed)
        return True

    def inspect(self):
        if self.auto_mode.get():
            if self.frame is not None:
                self.scan_snapshot(self.frame.copy())
            return
        if self.busy or self.model is None or self.frame is None:
            return
        try:
            threshold, expected = float(self.threshold.get()), int(self.expected.get())
            if not .01 <= threshold <= 1 or not 0 <= expected <= 500:
                raise ValueError('Confidence ต้องอยู่ระหว่าง 0.01–1 และจำนวนจุด 0–500')
            component = self.component.get()
            entry=self.checklist.session['results'].get(component,{})
            if entry.get('status')=='PASS' and entry.get('complete'):
                self.info.set(component+' ผ่านและล็อกผลแล้ว · กดรีเซ็ตการตรวจหากต้องการตรวจใหม่')
                return
            joints = parse_joints(self.joints.get(), COMPONENTS[component]['expected']) if component in COMPONENTS else []
            if joints and expected != len(joints):
                raise ValueError('จำนวนจุดตรวจต้องตรงกับหมายเลขจุด/ขาที่เห็นในภาพ')
            source, roi = crop_input(self.frame, self.roi, self.rotation)
        except ValueError as exc:
            self.error(exc)
            return
        self.frozen = True
        raw = self.frame.copy()
        rotation = self.rotation
        self.active_component = component
        self.clear_result()
        metadata = dict(region=self.region.get(), expected=expected, threshold=threshold,
                        model=str(Path(self.model_path).resolve()), scope='visible_region_only',
                        resolution=[raw.shape[1], raw.shape[0]], device=self.device, class_names=list(CLASSES))
        metadata.update(component=component, board_id=self.checklist.session['board_id'],
                        roi=list(roi), rotation=rotation, joints=joints,
                        input_resolution=[source.shape[1],source.shape[0]], component_source='operator_selection')
        self.info.set('กำลังตรวจภาพ…')
        self.source_label.set('ภาพนิ่ง · กำลังวิเคราะห์')
        self.set_result_status('ANALYZING', 'AI กำลังวิเคราะห์รอยบัดกรี', COLORS['accent'])
        started = time.perf_counter()
        diode_reference=self.diode_direction
        generation=self.scan_generation;board_id=self.checklist.session['board_id']
        def run():
            prediction = self.model.predict(source=source, conf=threshold, iou=.45, imgsz=1024, device=self.device, verbose=False,**inference_options(self.device))[0]
            detections=[]
            for d in quality_detections(prediction,self.class_map,threshold,getattr(self.model,'names',None)):
                points=np.array(d.polygon,dtype=float)
                x1,y1,x2,y2=roi;w,h=x2-x1,y2-y1
                u,v=points[:,0].copy(),points[:,1].copy()
                if rotation==90:points[:,0],points[:,1]=v,h-u
                elif rotation==180:points[:,0],points[:,1]=w-u,h-v
                elif rotation==270:points[:,0],points[:,1]=w-v,u
                points[:,0]+=x1;points[:,1]+=y1
                detections.append(detection_from_quad(points,d.class_id,d.confidence,d.native_class_id,d.native_class_name))
            state, reason = decision(detections, expected, threshold)
            annotated = raw.copy()
            for d in detections:
                draw_detection(annotated,d)
            result=raw,annotated,detections,state,reason,metadata
            if component.startswith('D'):
                from frame_registration import FeatureFrame
                from scanner import corners
                from diode_cv import assess_diode,combine_diode_result
                axis=diode_reference.axis(FeatureFrame(raw)) if diode_reference is not None else None
                result=combine_diode_result(result,assess_diode(raw,corners(roi),detections,axis))
            return result
        def done(result):
            if generation!=self.scan_generation or board_id!=self.checklist.session['board_id']:return
            self.result = result
            self.display_frame = result[1] if self.overlay.get() else result[0]
            color = {'PASS': COLORS['good'], 'FAIL': COLORS['red'], 'REVIEW': COLORS['amber']}[result[3]]
            self.set_result_status(result[3], result[4], color)
            for i, value in enumerate(self.counts):
                value.set(str(sum(d.class_id == i for d in result[2])))
            self.latency.set(f'{(time.perf_counter()-started)*1000:,.0f} ms')
            self.source_label.set('ภาพนิ่ง · วิเคราะห์แล้ว')
            self.save_btn.configure(state='normal')
            if component in COMPONENTS and result[3]=='FAIL':
                self.confirm_btn.configure(state='normal')
                self.confirm_btn.pack(fill='x',pady=(6,0),before=self.save_btn)
            for d in result[2]:
                self.table.insert('', 'end', values=(THAI[d.class_id], f'{d.confidence:.0%}'), tags=('good' if d.class_id == 0 else 'defect',))
            self.info.set('ผลเฉพาะบริเวณในภาพ ไม่ใช่การรับรองทั้งบอร์ด | กดบันทึกผลเพื่อเก็บรายงาน')
            if component in COMPONENTS:
                self.persist_component_result()
            self.render()
        self.job(run, done)

    def render(self):
        if hasattr(self,'board_layout'):self.board_layout.refresh()
        # The dark canvas is exactly the displayed image; the surrounding workspace stays white.
        if self.display_frame is not None:
            image_height,image_width=self.display_frame.shape[:2]
        else:image_width,image_height=640,480
        available_width=max(1,self.viewport.winfo_width())
        available_height=max(1,self.viewport.winfo_height())
        scale=min(available_width/image_width,available_height/image_height)
        fitted_width=max(1,round(image_width*scale));fitted_height=max(1,round(image_height*scale))
        self.canvas.place(relx=.5,rely=.5,anchor='center',width=fitted_width,height=fitted_height)
        self.canvas.delete('all')
        self.view_transform = None
        if self.display_frame is None:
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            cx, cy = cw / 2, ch / 2
            for x in range(24, cw, 36):
                for y in range(24, ch, 36):
                    self.canvas.create_oval(x, y, x+1, y+1, fill='#203044', outline='')
            self.canvas.create_rectangle(cx-34, cy-95, cx+34, cy-39, outline='#48c9b0', width=2)
            self.canvas.create_oval(cx-13, cy-81, cx+13, cy-55, outline='#48c9b0', width=2)
            self.canvas.create_text(cx, cy-6, fill='#eff6ff', text='พื้นที่ตรวจสอบรอยบัดกรี', font=(self.ui_font, 19, 'bold'))
            self.canvas.create_text(cx, cy+33, fill='#94a6bc', text='เชื่อมต่อ USB microscope หรือเปิดภาพจากเครื่อง', font=(self.ui_font, 11))
            self.canvas.create_text(cx, cy+67, fill='#5d768e', text='0805  /  0603  /  1206  /  SOP16  /  SOT-23  /  SSOP28', font=(LATIN_FONT, 10))
            self.image_size.set('— × — px')
            return
        height, width = self.display_frame.shape[:2]
        self.image_size.set(f'{width:,} × {height:,} px')
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / width, ch / height)
        dw, dh = max(1, round(width*scale)), max(1, round(height*scale))
        ox, oy = (cw-dw)/2, (ch-dh)/2
        render_frame=self.display_frame
        if self.auto_mode.get() and not self.frozen:
            annotated_live=self.live_overlay.draw(render_frame)
            if self.overlay.get():render_frame=annotated_live
            self.diode_visual_live={item['component']:item for item in self.live_overlay.diode_observations}
            self.refresh_diode_details()
        rgb = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((dw, dh), Image.Resampling.BILINEAR))
        self.canvas.create_image(ox, oy, anchor='nw', image=self.photo)
        self.view_transform = (ox,oy,dw/width,dh/height)
        if self.auto_mode.get() and self.overlay.get() and self.live_preview is None and time.monotonic()-self.auto_targets_at<1.5:
            for hit in self.position_hits:
                points=[(ox+x*dw/width,oy+y*dh/height) for x,y in hit.quad]
                self.canvas.create_polygon(*[p for point in points for p in point],fill='',outline='#6bcfff',width=2)
                x=min(point[0] for point in points);y=min(point[1] for point in points)
                self.canvas.create_text(max(2,min(x,cw-170)),max(2,y-22),anchor='nw',
                                        text=f'{getattr(hit,"source","OCR")} {hit.component} {hit.confidence:.1%}',fill='#6bcfff',font=(self.ui_font,10,'bold'))
            for target in self.auto_targets:
                points=[(ox+x*dw/width,oy+y*dh/height) for x,y in target.quad]
                self.canvas.create_polygon(*[p for point in points for p in point],fill='',outline='#48dfca',width=2)
                x,y=points[0]
                self.canvas.create_text(x+4,y+3,anchor='nw',text=target.component,fill='#48dfca',font=(self.ui_font,11,'bold'))
        if self.roi:
            x1,y1,x2,y2 = self.roi
            self.canvas.create_rectangle(ox+x1*dw/width,oy+y1*dh/height,ox+x2*dw/width,oy+y2*dh/height,
                                         outline='#48dfca',width=2,dash=(6,3))
            self.canvas.create_text(ox+x1*dw/width+5,oy+y1*dh/height+5,anchor='nw',fill='#48dfca',
                                    text=f'{self.component.get()} · ROI · {self.rotation}°',font=(self.ui_font,10,'bold'))

    def save_raw(self):
        if self.frame is None:
            return
        raw = self.frame.copy()
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            try:
                write_image(path, raw)
                self.info.set('บันทึกภาพต้นฉบับแล้ว')
            except Exception as exc:
                self.error(exc)

    def save_result(self):
        if self.result is None:
            self.error('ตรวจภาพด้วย AI ก่อนบันทึกผล')
            return
        try:
            if self.result[5].get('automatic'):
                self.info.set(f'ผลตรวจอัตโนมัติบันทึกไว้แล้ว: {self.report_path}')
                return
            if self.result[5].get('component') in COMPONENTS:
                self.persist_component_result()
                return
            path = self.report_path or save_report(self.report_root, *self.result)
            self.report_path = path
            self.info.set(f'บันทึกผลแล้ว: {path}')
        except Exception as exc:
            self.error(exc)

    def persist_component_result(self):
        """Evidence is saved before updating any checklist item."""
        metadata = self.result[5]
        try:
            if self.report_path is None:
                self.report_path = save_report(self.report_root, *self.result)
            if not (self.report_path/'crop.png').is_file():
                crop,_ = crop_input(self.result[0], metadata['roi'], metadata['rotation'])
                write_image(self.report_path/'crop.png', crop)
            name = metadata['component']
            current = self.checklist.session['results'].get(name,{})
            if current.get('report') != str(self.report_path.resolve()):
                self.checklist.record(name,self.result[3],self.result[4],self.report_path,
                                      metadata['expected'],metadata['roi'],metadata['rotation'],metadata['joints'],
                                      coverage_confirmed=len(self.result[2])==len(metadata['joints']),
                                      diode_polarity=metadata.get('diode_polarity',{}).get('status'),
                                      component_missing=metadata.get('component_missing',False))
            self.refresh_checklist()
            item = self.checklist.session['results'][name]
            self.info.set(f"{name} · บันทึกหลักฐานแล้ว · ตรวจจุด {item['covered']}/{item['total']} · {self.report_path.name}")
        except Exception as exc:
            self.info.set('บันทึกหลักฐาน/เช็กลิสต์ไม่สำเร็จ: '+str(exc))
            self.error(exc)

    def confirm_visible_joints(self):
        """A bridge can span several pads in one box; only the operator can confirm coverage."""
        if self.busy or self.result is None or self.result[3]!='FAIL':return
        metadata = self.result[5]
        name = metadata.get('component')
        if name not in COMPONENTS:return
        if self.report_path is None:
            self.persist_component_result()
            if self.report_path is None:return
        try:
            confirmation=dict(board_id=metadata['board_id'],component=name,joints=metadata['joints'],
                              coverage_confirmed_by='operator',status='FAIL')
            (self.report_path/'coverage_confirmation.json').write_text(json.dumps(confirmation,ensure_ascii=False,indent=2),'utf-8')
            self.checklist.record(name,'FAIL',self.result[4],self.report_path,metadata['expected'],
                                  metadata['roi'],metadata['rotation'],metadata['joints'],
                                  coverage_confirmed=True,confirmed_by='operator',
                                  diode_polarity=metadata.get('diode_polarity',{}).get('status'),
                                  component_missing=metadata.get('component_missing',False))
            self.refresh_checklist()
            self.confirm_btn.configure(state='disabled')
            self.info.set(f'{name}: ผู้ตรวจยืนยันว่าตรวจครบจุดที่ระบุในภาพแล้ว · ยังคงผลพบข้อบกพร่อง')
        except Exception as exc:self.error(exc)

    def tick(self):
        if self.monitor:
            try:
                sample = self.monitor.samples.get_nowait()
                if 'error' in sample:
                    self.hardware_note.set('อ่านทรัพยากรไม่ได้: ' + sample['error'])
                    for key in ('ram','vram'):self.resource_details[key].set('N/A')
                    for name in self.resource_values:
                        self.resource_values[name].set('N/A')
                        self.resource_bars[name]['value'] = 0
                else:
                    for name in ('cpu', 'ram', 'gpu', 'vram'):
                        value = sample[name]
                        self.resource_values[name].set('N/A' if value is None else f'{value:.0f}%')
                        self.resource_bars[name]['value'] = value or 0
                    self.resource_details['gpu'].set(sample['gpu_name'])
                    for key in ('ram', 'vram'):
                        used, total = sample.get(key+'_gb'), sample.get(key+'_total_gb')
                        amount = f'{used:.1f} / {total:.1f} GB' if used is not None and total else 'N/A'
                        self.resource_details[key].set(amount)
                    self.hardware_note.set(f"AI: {'CUDA 0' if self.device == '0' else 'CPU'}  ·  App RAM {sample['app_mb']:.0f} MB  ·  GPU monitor: {sample['gpu_name']}")
            except queue.Empty:
                pass
        try:
            callback, result, error, on_error = self.events.get_nowait()
            self.busy = False
            if error:
                self.info.set('ทำรายการไม่สำเร็จ')
                if on_error:
                    on_error(error)
                else:
                    self.set_result_status('ERROR', str(error), COLORS['red'])
                    self.error(error)
            else:
                callback(result)
            self.inspect_btn.configure(state='normal' if self.model and not self.busy else 'disabled')
        except queue.Empty:
            pass
        if self.camera:
            if self.camera.error:
                self.camera_label.set('●  กล้องขัดข้อง')
                self.info.set(self.camera.error)
                self.camera.error = None
                self.cancel_scan()
                self.frozen = True
            elif not self.frozen:
                sequence,image = self.camera.latest_sample()
                if image is not None:
                    fresh=sequence!=self.video_sequence
                    if fresh:
                        self.video_sequence=sequence;self.preview_frames+=1
                        elapsed=time.monotonic()-self.preview_epoch
                        if elapsed>=1:
                            self.preview_fps=self.preview_frames/elapsed
                            self.preview_frames=0;self.preview_epoch=time.monotonic()
                    if self.named_camera and str(self.camera.index)!=self.cam_index.get():
                        self.cam_index.set(str(self.camera.index))
                    self.camera_label.set('●  '+MICROSCOPE_NAME if self.named_camera else '●  กล้องเชื่อมต่อแล้ว')
                    self.source_label.set('LIVE · USB MICROSCOPE')
                    self.frame = image
                    self.display_frame = image
                    if not self.auto_mode.get():
                        self.info.set(f'ภาพสด {image.shape[1]} × {image.shape[0]} px | ปรับโฟกัสและแสงที่กล้อง แล้วกดจับภาพ')
                    self.maybe_scan(image,sequence)
                    if fresh:self.render()
        self.tick_id = self.root.after(16, self.tick)

    def close(self):
        self.persist_board_number()
        if self.save_board_timer is not None:self.root.after_cancel(self.save_board_timer)
        if self.preferences_timer is not None:
            self.root.after_cancel(self.preferences_timer)
            self.persist_preferences()
        self.cancel_scan()
        self.worker.shutdown(wait=False, cancel_futures=True)
        if self.monitor:
            self.monitor.stop()
        if self.watch_id:
            self.root.after_cancel(self.watch_id)
        if self.tick_id:
            self.root.after_cancel(self.tick_id)
        if self.camera:
            self.camera.stop()
        self.root.destroy()


if __name__ == '__main__':
    from startup import Startup
    root = tk.Tk()
    Startup(root, lambda info: App(root, runtime_info=info, auto_scan=True, auto_camera=True))
    root.mainloop()
