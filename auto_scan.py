"""GUI coordination for automatic inspection; workers never read Tk variables."""
from pathlib import Path
import copy
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from board import COMPONENTS
from inspection import THAI, read_image
from scanner import StableCapture, inspect_target, save_auto_report
from positioning import PositionLocator
from trained_locator import TrainedLocator
from ui import COLORS
from preferences import confidence_value
from live_scan import inspect_live_frame,IC_MISSING_CONFIRM_SECONDS
from inspection import decision
from live_overlay import LiveOverlay


class AutoScanMixin:
    def init_auto_scan(self, enabled, reference_file=None):
        self.auto_mode=tk.BooleanVar(self.root,value=enabled)
        self.scan_status=tk.StringVar(self.root,value='ตรวจต่อเนื่อง · ใช้ภาพล่าสุดจากกล้อง')
        self.position_detail=tk.StringVar(self.root,value='best_locator.pt · พร้อมระบุอุปกรณ์' if self.locator_model is not None else 'ระบุตำแหน่งด้วยภาพอ้างอิง / OCR')
        self.position_hits=[]
        assets=Path(__file__).parent/'assets'
        custom=assets/'board-reference-user.png'
        self.reference_file=Path(reference_file) if reference_file else (custom if custom.is_file() else assets/'smd-board-reference.png')
        self.locator=None
        self.capture_gate=StableCapture()
        self.scan_generation=0
        self.scan_cancel=threading.Event()
        self.scan_attempts={}
        self.scan_after=0
        self.auto_targets=[]
        self.auto_targets_at=0
        self.scan_timer=None
        self.live_preview=None
        self.live_evidence_cache={}
        self.ic_missing_since={}
        self.last_submitted_sequence=None
        self.latest_camera_sequence=None
        self.latest_camera_at=0
        self.live_overlay=LiveOverlay()
        self.video_sequence=None;self.preview_frames=0;self.preview_epoch=time.monotonic();self.preview_fps=0.

    def cancel_scan(self):
        self.ic_missing_since.clear()
        self.scan_cancel.set()
        self.scan_generation+=1
        self.auto_targets=[]
        self.position_hits=[]
        self.capture_gate.reset()
        self.live_preview=None
        self.last_submitted_sequence=None
        self.live_overlay.clear()
        if self.scan_timer is not None:
            self.root.after_cancel(self.scan_timer)
            self.scan_timer=None

    def change_scan_mode(self):
        self.cancel_scan()
        self.workspace_menu.entryconfigure(0,state='normal' if self.auto_mode.get() else 'disabled')
        if self.auto_mode.get():
            self.save_btn.pack_forget()
            self.confirm_btn.pack_forget()
            self.manual_selection.pack_forget();self.manual_tools.pack_forget()
            self.manual_settings.pack_forget()
            self.manual_hint.pack_forget()
            self.auto_settings.pack(fill='x',before=self.threshold_label)
            self.auto_hint.pack(fill='x',before=self.model_card,pady=(10,8))
            self.roi=None
            if self.camera and not self.camera.stop_event.is_set():self.frozen=False
            self.scan_status.set('ตรวจต่อเนื่อง · ไม่ต้องหยุดบอร์ด · แสดงผลภาพล่าสุดที่ประมวลผลเสร็จ')
        else:
            self.save_btn.pack(fill='x',pady=(6,0))
            self.manual_selection.pack(fill='x',before=self.viewport)
            self.manual_tools.pack(fill='x',before=self.viewport)
            self.auto_hint.pack_forget()
            self.auto_settings.pack_forget()
            self.manual_settings.pack(fill='x',before=self.threshold_label)
            self.manual_hint.pack(anchor='w',pady=(4,5),before=self.model_card)
            self.scan_status.set('ตรวจภาพด้วยตนเอง')
        self.refresh_checklist()
        self.render()

    def reset_scan_board(self):
        self.cancel_scan()
        self.scan_attempts.clear()
        self.live_evidence_cache.clear()

    def choose_reference(self):
        if self.busy:return
        from calibration import ReferenceWindow
        ReferenceWindow(self)

    def choose_camera_reference(self):
        if self.busy:return
        from reference_views import MicroscopeReferenceWindow
        MicroscopeReferenceWindow(self)

    def set_scan_reference(self, path):
        self.cancel_scan()
        self.reference_file=Path(path)
        self.locator=None
        self.scan_attempts.clear()
        self.scan_status.set('ตั้งภาพอ้างอิงบอร์ดจริงแล้ว · พร้อมจับตำแหน่งใหม่')
        if self.auto_mode.get() and self.frozen and self.frame is not None:
            generation=self.scan_generation
            def scan_reference_view():
                self.scan_timer=None
                if generation==self.scan_generation and self.auto_mode.get():self.scan_snapshot(self.frame.copy())
            self.scan_timer=self.root.after(100,scan_reference_view)

    def retry_scan(self):
        if self.busy:return
        self.reset_scan_board()
        self.scan_status.set('พร้อมลองตรวจจุดที่ยังไม่ครบใหม่')
        if self.frozen and self.frame is not None:self.scan_snapshot(self.frame.copy())

    def maybe_scan(self, frame, sequence):
        if not self.auto_mode.get() or self.frozen or self.board_saving:return
        if sequence!=self.latest_camera_sequence:
            self.latest_camera_sequence=sequence
            self.latest_camera_at=time.monotonic()
            if self.locator_model is None:self.scan_attempts.clear()
        elif time.monotonic()-self.latest_camera_at>1:
                self.scan_status.set('รอภาพใหม่จากกล้อง')
        # Busy workers never queue frames. Once available, consume the newest
        # camera sample rather than processing intermediate/obsolete frames.
        if sequence!=self.last_submitted_sequence and not self.busy and time.monotonic()>=self.scan_after:
            self.last_submitted_sequence=sequence
            if self.locator_model is None:self.scan_attempts.clear()
            self.scan_snapshot(frame.copy())

    def scan_model_frame(self,frame,position_only=False):
        try:
            threshold=confidence_value(self.threshold.get())
            locator_threshold=confidence_value(self.locator_threshold.get())
        except (ValueError,TypeError) as exc:
            self.scan_status.set(str(exc));return
        self.scan_cancel=threading.Event()
        cancel=self.scan_cancel;generation=self.scan_generation
        board_id=self.checklist.session['board_id']
        stage=self.checklist.frame_snapshot()
        ledger=stage.session['results']
        cache=dict(self.live_evidence_cache)
        missing_since=dict(self.ic_missing_since)
        locator=self.locator
        locator_model=self.locator_model;locator_map=dict(self.locator_class_map)
        locator_file=self.locator_file;device=self.device
        model=None if position_only else self.model
        mapping=dict(self.class_map);model_path=self.model_path;report_root=self.report_root
        pins=dict(self.pin_calibrations);diode=self.diode_direction
        started=time.perf_counter()
        if self.live_preview is None:self.scan_status.set('ภาพสดทำงานต่อ · กำลังเตรียมผล AI ภาพแรก…')
        def run():
            current=locator or TrainedLocator(locator_model,locator_map,locator_file,device,locator_threshold)
            payload=inspect_live_frame(frame,current,model,mapping,threshold,locator_threshold,device,
                                      model_path,board_id,ledger,cache,report_root,cancel,pins,diode,missing_since)
            if payload is not None and not cancel.is_set():
                payload['staged_file']=stage.prepare_frame(payload['reports'])
            return payload
        def failed(exc):
            if generation!=self.scan_generation:return
            self.ic_missing_since.clear()
            self.scan_status.set('ตรวจภาพไม่สำเร็จ: '+str(exc))
            self.scan_after=time.monotonic()+1
        def done(payload):
            if payload is None:return
            staged_file=payload.pop('staged_file',None)
            try:
                if cancel.is_set() or generation!=self.scan_generation:return
                if not self.checklist.commit_frame(stage,staged_file):return
            except OSError as exc:
                self.info.set('บันทึกผลตรวจไม่สำเร็จ: '+str(exc));return
            finally:
                if staged_file is not None:staged_file.cleanup()
            self.locator=payload['locator']
            self.ic_missing_since=payload.get('ic_missing_since',{})
            self.position_hits=list(payload.get('locator_hits',[]))
            self.ic_observations=payload.get('ic_observations',{})
            self.ic_observation_frame=payload['raw']
            if len(self.ic_observations)==1 and self.ic_component.get() not in self.ic_observations:
                self.ic_component.set(next(iter(self.ic_observations)))
            for item in payload.get('diode_overlays',[]):self.diode_live[item['component']]=item
            self.position_detail.set(' · '.join([self.locator.detail,*payload.get('calibration_notes',[])]))
            self.auto_targets=payload['targets'];self.auto_targets_at=time.monotonic()
            self.live_preview=(payload['raw'],payload['annotated'])
            self.live_overlay.update(payload)
            self.scan_after=0
            reports=payload['reports']
            for result,report,key,changed in reports:
                meta=result[5];name=meta['component']
                if changed:
                    self.live_evidence_cache[name]=(key,report)
            if reports:
                self.show_auto_result(*reports[-1][:2])
            detections=payload['detections']
            if reports:
                states=[self.checklist.session['results'].get(r[0][5]['component'],{}).get('status',r[0][3])
                        if r[0][5].get('merge_visible_pins') else r[0][3] for r in reports]
                state='FAIL' if 'FAIL' in states else ('REVIEW' if 'REVIEW' in states else 'PASS')
                detail=' · '.join(r[0][5]['component']+': '+r[0][3] for r in reports)
                if len(reports)==1:
                    latest=reports[0][0];meta=latest[5]
                    detail=meta['component']+': '+(f"ขาที่เห็น {','.join(map(str,meta['joints']))}" if
                        meta.get('merge_visible_pins') and latest[3]!='FAIL' else latest[4])
                self.set_result_status(state,detail,{'PASS':COLORS['good'],'FAIL':COLORS['red'],'REVIEW':COLORS['amber']}[state])
            else:
                state,detail=decision(detections,0,threshold)
                if position_only or model is None:
                    state='LOCATED' if self.position_hits else 'SCANNING'
                    detail='ระบุตำแหน่งเท่านั้น · ยังไม่มีผลตรวจคุณภาพ' if position_only else 'รอ best_inspec.pt'
                else:detail='ผลภาพล่าสุด · ยังไม่ยืนยันอุปกรณ์ในเช็กลิสต์'
                if payload.get('pending_ic_missing'):
                    state='REVIEW'
                    detail=' · '.join(f'{name}: รอยืนยัน missing {elapsed:.2f}/{IC_MISSING_CONFIRM_SECONDS:.2f} วินาที'
                                      for name,elapsed in payload['pending_ic_missing'].items())
                locked=payload.get('locked_components',[])
                if locked and len(locked)==len({h.component for h in self.position_hits}):
                    state='PASS';detail=', '.join(locked)+' · ผ่านและล็อกผลแล้ว'
                self.result=(payload['raw'],payload['annotated'],detections,state,detail,
                    dict(automatic=False,scope='visible_frame_only',threshold=threshold,model=model_path,
                         locator_threshold=locator_threshold,component='ไม่ระบุ',continuous=True,
                         model_class_names=getattr(model,'names',None)))
                self.report_path=None
                self.set_result_status(state,detail,COLORS['good'] if state=='PASS' else (COLORS['red'] if state=='FAIL' else COLORS['amber']))
            self.table.delete(*self.table.get_children())
            for d in detections:
                self.table.insert('','end',values=(THAI[d.class_id],f'{d.confidence:.0%}'),tags=('good' if d.class_id==0 else 'defect',))
            for i,value in enumerate(self.counts):value.set(str(sum(d.class_id==i for d in detections)))
            self.scan_status.set('ตรวจต่อเนื่อง · ผลภาพล่าสุด · '+str(len(self.position_hits))+' กรอบอุปกรณ์ / '+str(len(detections))+' รอยบัดกรี')
            if payload.get('pending_ic_missing'):
                self.scan_status.set(self.scan_status.get()+' · '+', '.join(payload['pending_ic_missing'])+f' รอยืนยัน missing {IC_MISSING_CONFIRM_SECONDS:.2f} วินาที')
            self.latency.set(f'ภาพ {self.preview_fps:.0f} FPS · AI {(time.perf_counter()-started)*1000:,.0f} ms')
            self.display_frame=(self.live_preview[1] if self.overlay.get() else self.live_preview[0]) if self.frozen else self.frame
            if self.display_frame is None:self.display_frame=payload['raw']
            self.refresh_checklist();self.render()
        self.job(run,done,failed)

    def locate_only(self):
        if not self.auto_mode.get():
            self.auto_mode.set(True);self.change_scan_mode()
        if self.frame is not None:self.scan_snapshot(self.frame.copy(),position_only=True)

    def scan_snapshot(self, frame, position_only=False):
        if self.busy:return
        if self.locator_model is not None:
            self.scan_model_frame(frame,position_only)
            return
        try:
            threshold=float(self.threshold.get())
            if not .01<=threshold<=1:raise ValueError('เกณฑ์ความมั่นใจต้องอยู่ระหว่าง 0.01–1')
        except ValueError as exc:
            self.error(exc);return
        self.scan_cancel=threading.Event()
        cancel=self.scan_cancel;generation=self.scan_generation
        board_id=self.checklist.session['board_id']
        ledger=copy.deepcopy(self.checklist.session['results'])
        locator=self.locator;reference_file=self.reference_file
        locator_model=self.locator_model;locator_mapping=self.locator_class_map.copy();locator_file=self.locator_file
        model=None if position_only else self.model
        class_map=self.class_map.copy();device=self.device;model_path=self.model_path
        attempted=set(self.scan_attempts)
        root=self.report_root
        self.scan_status.set('กำลังหาตำแหน่งอุปกรณ์ในภาพซูม…')
        started=time.perf_counter()
        def run():
            current=locator or (TrainedLocator(locator_model,locator_mapping,locator_file,device)
                                if locator_model is not None else PositionLocator(reference_file))
            if cancel.is_set():return current,None,[],[]
            registration=current.locate(frame)
            if registration is None:return current,None,[],[]
            targets=current.targets(frame,registration,ledger)
            reports=[]
            if model is not None:
                eligible=[t for t in targets if (t.component,tuple(t.joints)) not in attempted]
                for target in eligible[:6]:
                    if cancel.is_set():break
                    result,crop=inspect_target(model,class_map,frame,target,registration,threshold,device,model_path,board_id,current)
                    if cancel.is_set():break
                    report=save_auto_report(root,result,crop)
                    reports.append((result,report))
            return current,registration,targets,reports
        def failed(exc):
            if generation!=self.scan_generation:return
            self.scan_status.set('สแกนไม่สำเร็จ: '+str(exc))
            self.scan_after=time.monotonic()+2
        def done(payload):
            if generation!=self.scan_generation or cancel.is_set() or board_id!=self.checklist.session['board_id']:return
            self.locator,registration,targets,reports=payload
            self.position_hits=self.locator.hits
            self.position_detail.set(self.locator.detail)
            self.scan_after=time.monotonic()+(.5 if model is not None else 2)
            if registration is None:
                self.auto_targets=[]
                self.auto_targets_at=time.monotonic()
                self.scan_status.set('ยังไม่ส่งตรวจ AI · ต้องยืนยันตำแหน่งอุปกรณ์ก่อน')
                if position_only or self.result is None:
                    self.set_result_status('พบชื่อ' if self.position_hits else 'รอตำแหน่ง',
                                           'ยังไม่ยืนยันกรอบอุปกรณ์ · ไม่มีผลตรวจคุณภาพ',COLORS['amber'])
                self.render()
                return
            self.auto_targets=targets;self.auto_targets_at=time.monotonic()
            for result,report in reports:
                metadata=result[5];name=metadata['component']
                try:
                    self.checklist.record(name,result[3],result[4],report,metadata['expected'],None,0,
                                          metadata['joints'],confirmed_by=metadata.get('confirmed_by','automatic_pad_mapping'),joint_states=metadata['joint_states'])
                except Exception as exc:
                    self.scan_status.set('บันทึกเช็กลิสต์ไม่สำเร็จ: '+str(exc));return
                self.scan_attempts[(name,tuple(metadata['joints']))]=str(report)
                self.show_auto_result(result,report)
            names=', '.join(t.component for t in targets[:8])
            if position_only:
                self.scan_status.set(f'ทดสอบตำแหน่ง: {names or "ไม่พบจุดตรวจที่ค้างในภาพ"}')
                self.set_result_status('LOCATED','ทดสอบตำแหน่งเท่านั้น · ยังไม่ตรวจคุณภาพรอยบัดกรี',COLORS['accent'])
            elif model is None:
                self.scan_status.set(f'พบ {names or "บริเวณบอร์ด"} · รอ best_inspec.pt (หรือ best.pt) เพื่อเปิดการตรวจ AI')
                self.set_result_status('WAIT AI','จับตำแหน่งได้แล้ว · วาง best_inspec.pt เพื่อเริ่มตรวจรอยบัดกรี',COLORS['amber'])
            elif reports:
                self.scan_status.set('จับภาพและตรวจอัตโนมัติแล้ว: '+', '.join(r[0][5]['component'] for r in reports))
                if self.frozen:
                    def next_batch():
                        self.scan_timer=None
                        if generation==self.scan_generation and self.auto_mode.get() and self.frozen:self.scan_snapshot(frame)
                    self.scan_timer=self.root.after(50,next_batch)
            elif not targets:
                self.scan_status.set('บริเวณนี้ตรวจครบแล้ว · เลื่อนไปยังอุปกรณ์ถัดไป')
            else:
                self.scan_status.set('บริเวณนี้ตรวจแล้ว · จุดที่ยังไม่ครบกด “ลองจุดที่ค้างใหม่” หลังปรับภาพ')
            self.refresh_checklist()
            self.latency.set(f'{(time.perf_counter()-started)*1000:,.0f} ms')
            self.render()
        self.job(run,done,failed)

    def show_auto_result(self,result,report):
        self.result=result;self.report_path=report
        self.component.set(result[5]['component']);self.active_component=self.component.get()
        self.set_result_status(result[3],result[4],{'PASS':COLORS['good'],'FAIL':COLORS['red'],'REVIEW':COLORS['amber']}[result[3]])
        self.table.delete(*self.table.get_children())
        for d in result[2]:
            self.table.insert('','end',values=(THAI[d.class_id],f'{d.confidence:.0%}'),tags=('good' if d.class_id==0 else 'defect',))
        for i,value in enumerate(self.counts):value.set(str(sum(d.class_id==i for d in result[2])))
        self.save_btn.configure(state='normal')
        self.confirm_btn.pack_forget()
        if self.frozen:self.display_frame=result[1] if self.overlay.get() else result[0]

    def show_evidence(self,name=None,archived_entry=None,board_number=None):
        from evidence import result_evidence,read_evidence
        if name is None:
            if self.result is None:return
            title=self.result[5].get('component','ผลตรวจ')
            image=None if title in ('U1','U4') and not self.result[5].get('component_missing') else result_evidence(self.result)
        else:
            entry=archived_entry if archived_entry is not None else self.checklist.session['results'].get(name)
            image=None
            if entry and entry.get('report') and (name not in ('U1','U4') or entry.get('component_missing')):
                try:image=read_evidence(entry['report'])
                except Exception as exc:self.error(exc);return
            title=name
        entry = archived_entry if archived_entry is not None else self.checklist.session['results'].get(title,{})
        window=tk.Toplevel(self.root);window.title((str(board_number)+' · ' if board_number else '')+title+' · ภาพหลักฐานผลตรวจ')
        window.configure(bg='white')
        from component_info import component_text
        details=tk.Frame(window,bg='white',padx=16,pady=16,width=310)
        details.pack(side='right',fill='both')
        tk.Label(details,text=title+' · ข้อมูลผลตรวจ',font=(self.ui_font,14,'bold'),bg='white',fg=COLORS['ink']).pack(anchor='w',pady=(0,10))
        text=tk.Text(details,width=32,height=18,wrap='word',font=(self.ui_font,10),bd=0,bg='white',fg=COLORS['ink'])
        scroll=ttk.Scrollbar(details,command=text.yview);scroll.pack(side='right',fill='y')
        text.configure(yscrollcommand=scroll.set)
        text.pack(fill='both',expand=True)
        text.insert('1.0',component_text(title,entry));text.configure(state='disabled')
        if title in ('U1','U4') and not entry.get('component_missing'):
            from ic_evidence import show_banks
            show_banks(window,title,entry,self.ui_font)
            return
        if image is None:
            tk.Label(window,text='ยังไม่มีภาพหลักฐานของ '+title,width=32,height=10,bg=COLORS['navy'],fg='white').pack(side='left',fill='both',expand=True)
            return
        photo=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        scale=min(6.,1000/photo.width,700/photo.height)
        photo=photo.resize((max(1,round(photo.width*scale)),max(1,round(photo.height*scale))),Image.Resampling.LANCZOS)
        picture=ImageTk.PhotoImage(photo)
        label=tk.Label(window,image=picture,bg=COLORS['navy']);label.image=picture;label.pack(side='left',fill='both',expand=True)
