"""Calibration actions, physical-pin details, and completed-board exports."""
from pathlib import Path
import copy
import json
import tkinter as tk
from PIL import Image,ImageTk
import cv2

from board import COMPONENTS
from board_export import inspected_component,scanned_component,export_board
from calibration_tools import PointCalibrationWindow
from pin_calibration import read_calibrations,save_calibration,refine_pin_points
from inspection import read_image
from calibration_files import load_calibration_profile


class WorkflowMixin:
    def init_workflow(self,folder):
        self.calibration_root=Path(folder)
        self.pin_calibrations,self.diode_direction,self.calibration_issues=read_calibrations(folder)
        self.board_saving=False;self.save_board_timer=None;self.last_saved_board=None;self.last_saved_signature=None
        self.diode_live={}
        self.diode_visual_live={}
        self.ic_observations={};self.ic_observation_frame=None

    def refresh_diode_details(self):
        if not hasattr(self,'diode_summary'):return
        titles={'PASS':'ถูกด้าน','FAIL':'กลับด้าน','REVIEW':'ยังอ่านด้านไม่ได้','NOT_APPLICABLE':'ไม่มีตัว'}
        rows=[];details=[]
        for name in ('D1','D2','D3','D4','D5'):
            entry=self.checklist.session['results'].get(name,{})
            item=self.diode_live.get(name,{})
            locked=entry.get('status')=='PASS' and entry.get('diode_polarity_state')=='PASS'
            state='PASS' if locked else item.get('status',entry.get('diode_polarity_state'))
            rows.append(name+': '+titles.get(state,'รอตรวจ')+(' ✓' if locked else ''))
            reason='ผ่านและล็อกผลแล้ว' if locked else item.get('reason',titles.get(state,'ยังไม่มีผลตรวจทิศทาง'))
            details.append(name+' · '+reason)
        sides={'LEFT':'ซ้าย','RIGHT':'ขวา','UP':'บน','DOWN':'ล่าง','CENTER':'กลาง'}
        observed=getattr(self,'diode_visual_live',{})
        live_lines=[]
        for name,item in observed.items():
            side=sides.get(item.get('band_side_image'))
            line=name+': '+('แถบอยู่'+side if side else 'ยังหาแถบไม่ชัด')
            line+=' · '+titles.get(item.get('status'),'ยังอ่านด้านไม่ได้')
            live_lines.append(line)
        prefix=('สด '+ ' / '.join(live_lines)+'\n') if live_lines else ('ยังไม่ตั้งขั้วบอร์ด + / −\n' if self.diode_direction is None else '')
        self.diode_summary.set(prefix+'   '.join(rows[:3])+'\n'+'   '.join(rows[3:]))
        if hasattr(self,'diode_live_brief'):
            self.diode_live_brief.set('ไดโอด · '+(' / '.join(live_lines) if live_lines else 'รอภาพ · ดูผลในแท็บไดโอด'))
        if self.diode_direction is None:
            self.diode_hint.set('ยังไม่ตั้งขั้วบอร์ด · ตั้งค่าโปรแกรม > ขา IC / ไดโอด\nพบแถบได้แต่ยังตัดสินถูก/กลับด้านไม่ได้\n'+'\n'.join(live_lines))
        elif self.diode_live or any(name.startswith('D') for name in self.checklist.session['results']):
            self.diode_hint.set('ภาพสด: '+(' / '.join(live_lines) or 'รอเห็นไดโอด')+'\nผลบอร์ด (รายการผ่านยังล็อกไว้):\n'+'\n'.join(details))
        else:self.diode_hint.set('พร้อมตรวจแถบขั้วลบด้วย OpenCV · เลื่อนให้เห็นไดโอด')
        detail=self.diode_hint.get()
        if hasattr(self,'diode_details') and detail!=getattr(self,'_diode_detail_text',None):
            self._diode_detail_text=detail
            self.diode_details.configure(state='normal');self.diode_details.delete('1.0','end')
            self.diode_details.insert('1.0',detail);self.diode_details.configure(state='disabled')

    def persist_board_number(self,event=None):
        value=self.board_number.get().strip()
        old=self.checklist.session.get('board_number','')
        if value==old:return True
        self.checklist.session['board_number']=value
        try:self.checklist.save()
        except Exception as exc:
            self.checklist.session['board_number']=old;self.error(exc);return False
        self.refresh_checklist()
        return True

    def reset_board_inspection(self):
        if self.board_saving:return
        if not self.persist_board_number():return
        self.cancel_scan()
        try:
            self.checklist.reset_results();self.reset_scan_board()
            self.ic_observations.clear();self.ic_observation_frame=None
            self.diode_live.clear();self.diode_visual_live.clear();self.last_saved_signature=None
            self.clear_result();self.refresh_checklist()
            self.info.set('รีเซ็ตผลตรวจบอร์ดเดิมแล้ว · รายการที่ผ่านสามารถตรวจใหม่ได้')
        except Exception as exc:self.error(exc)

    def calibration_summary(self):
        return ' · '.join([name+(' พร้อม' if name in self.pin_calibrations else ' ยังไม่ตั้ง') for name in ('U1','U4')]+[
                          'ไดโอด'+(' พร้อม' if self.diode_direction is not None else ' ยังไม่ตั้งขั้ว')])

    def calibrate_ic(self):
        PointCalibrationWindow(self,self.ic_component.get())

    def calibrate_diode(self):
        PointCalibrationWindow(self,'diode')

    def install_calibration(self,kind,image,points,joint_centers=None):
        extra=None
        if kind in ('U1','U4') and joint_centers is not None:
            original=copy.deepcopy(points)
            points,matches=refine_pin_points(kind,points,joint_centers)
            extra=dict(operator_points=original,pitch_refinement_matches=matches)
        save_calibration(self.calibration_root,kind,image,points,extra)
        self.cancel_scan()
        self.pin_calibrations,self.diode_direction,self.calibration_issues=read_calibrations(self.calibration_root)
        previous=[]
        if kind in ('U1','U4'):
            old=self.checklist.session['results'].pop(kind,None)
            if old:previous.append(dict(component=kind,result=old))
        else:
            self.diode_live.clear();self.diode_visual_live.clear()
            for name,item in self.checklist.session['results'].items():
                if not name.startswith('D'):continue
                if item.get('diode_polarity_state')=='NOT_APPLICABLE':continue
                previous.append(dict(component=name,result=item))
                item=item.copy()
                self.checklist.session['results'][name]=item
                item['diode_polarity_state']='REVIEW'
                if item['status']=='PASS':item['status']='REVIEW'
        if previous:
            self.checklist.session.setdefault('calibration_resets',[]).append(dict(kind=kind,previous_results=previous))
            self.checklist.save()
        self.live_evidence_cache.clear();self.scan_after=0
        self.calibration_status.set(self.calibration_summary())
        self.info.set('บันทึกจุดอ้างอิง '+kind+' แล้ว · พร้อมตรวจต่อ')
        self.refresh_checklist()

    def refresh_pin_details(self,*_):
        if not hasattr(self,'pin_table'):return
        if hasattr(self,'ic_overview'):
            overview=[]
            for component in ('U1','U4'):
                coverage=self.checklist.session['results'].get(component,{}).get('coverage',{})
                pins=[coverage.get(str(pin),{}) for pin in range(1,COMPONENTS[component]['expected']+1)]
                confirmed=[p for p in pins if p.get('confirmed') and p.get('confirmed_by') not in
                           ('latest_frame_component_count','full_component_detection_count')]
                passed=sum(p.get('status')=='PASS' for p in confirmed)
                failed=sum(p.get('status')=='FAIL' for p in confirmed)
                overview.append(f"{component} · {len(confirmed)}/{COMPONENTS[component]['expected']} ขา · ผ่าน {passed} · ไม่ผ่าน {failed}")
            self.ic_overview.set('\n'.join(overview))
        name=self.ic_component.get();total=COMPONENTS[name]['expected']
        entry=self.checklist.session['results'].get(name,{})
        if entry.get('component_missing'):
            self.pin_progress.set(name+' · MISSING · พบ missing จากฝั่งที่เห็นแล้ว')
            self.pin_table.delete(*self.pin_table.get_children())
            self._pin_signature=None
            return
        coverage=entry.get('coverage',{});rows=[]
        for pin in range(1,total+1):
            item=coverage.get(str(pin),{})
            physical=item.get('confirmed_by') not in ('latest_frame_component_count','full_component_detection_count')
            state=item.get('status','PENDING') if physical else 'REVIEW'
            rows.append((pin,state,bool(item.get('confirmed',False) and physical)))
        calibration=self.pin_calibrations.get(name)
        located=calibration is not None and (getattr(calibration,'last_registration',None) is not None or calibration.tracker.last is not None)
        candidates=self.ic_observations.get(name,[]) if not located else []
        candidate_signature=tuple((d['class_id'],round(d['confidence'],2)) for d in candidates)
        hint=('ตั้งขา 1 ในตั้งค่าโปรแกรม > ขา IC / ไดโอด' if calibration is None else calibration.detail)
        progress=f'{name} · ตรวจขาจริง {sum(row[2] for row in rows)}/{total}\n'+hint
        if entry.get('status')=='FAIL':
            progress=name+' · FAIL\n'+entry.get('reason','พบข้อบกพร่อง')+'\n'+progress
        self.pin_progress.set(progress)
        signature=(name,tuple(rows),candidate_signature,located)
        if getattr(self,'_pin_signature',None)==signature:return
        self._pin_signature=signature
        self.pin_table.delete(*self.pin_table.get_children())
        from inspection import CLASSES
        for index,item in enumerate(candidates):
            self.pin_table.insert('','end',iid=f'candidate:{index}',values=('ยังไม่ผูกเลขขา',f"{CLASSES[item['class_id']]} {item['confidence']:.0%}"),tags=('REVIEW',))
        titles={'PASS':'ผ่าน','FAIL':'ไม่ผ่าน','REVIEW':'รอตรวจเพิ่ม','PENDING':'ยังไม่ตรวจ'}
        for pin,state,confirmed in rows:
            self.pin_table.insert('','end',iid=str(pin),values=(f'{name} ขา {pin}',titles[state]),tags=(state,))

    def show_pin_evidence(self,event=None):
        from evidence import read_evidence,crop_evidence
        selected=self.pin_table.selection()
        if not selected:return
        name=self.ic_component.get();pin=selected[0]
        title=f'{name} ขา {pin}'
        try:
            if pin.startswith('candidate:'):
                candidate=self.ic_observations.get(name,[])[int(pin.split(':')[1])]
                image=crop_evidence(self.ic_observation_frame,candidate['quad'],name+' / unassigned')
                title=name+' รอยต่อที่พบ · ยังไม่ผูกเลขขา'
            else:
                item=self.checklist.session['results'].get(name,{}).get('coverage',{}).get(pin)
                if not item:return
                image=read_evidence(item['report'],pin)
        except Exception as exc:self.error(exc);return
        photo=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        scale=min(6.,1000/photo.width,700/photo.height)
        photo=photo.resize((max(1,round(photo.width*scale)),max(1,round(photo.height*scale))),Image.Resampling.LANCZOS)
        window=tk.Toplevel(self.root);window.title(title+' · ภาพหลักฐาน')
        picture=ImageTk.PhotoImage(photo)
        label=tk.Label(window,image=picture);label.image=picture;label.pack()

    def inspected_count(self):
        return sum(inspected_component(name,self.checklist.session['results'].get(name)) for name in COMPONENTS)

    def scanned_count(self):
        return sum(scanned_component(name,self.checklist.session['results'].get(name)) for name in COMPONENTS)

    def board_signature(self):
        records=self.checklist.session['results']
        return (self.checklist.session['board_id'],self.checklist.session.get('board_number',''),tuple((name,records.get(name,{}).get('status'),
                records.get(name,{}).get('report'),records.get(name,{}).get('covered'),
                records.get(name,{}).get('diode_polarity_state')) for name in COMPONENTS))

    def save_complete_board(self):
        if self.board_saving:return
        from board_export import validate_board_number
        try:validate_board_number(self.board_number.get())
        except ValueError as exc:
            self.info.set(str(exc));self.board_number_entry.focus_set();return
        if not self.persist_board_number():return
        if self.board_signature()==self.last_saved_signature:
            self.info.set('บอร์ดนี้บันทึกแล้ว: '+str(self.last_saved_board));return
        if self.scanned_count()!=len(COMPONENTS):
            self.info.set('ต้องสแกนให้มีผลตรวจครบทุกอุปกรณ์ก่อนบันทึก');return
        snapshot=copy.deepcopy(self.checklist.session)
        signature=self.board_signature()
        profiles={}
        try:
            for kind in ('U1','U4','diode'):
                path=self.calibration_root/kind/'active.json'
                if path.is_file():profiles[kind]=load_calibration_profile(path)
        except (ValueError,OSError,KeyError,TypeError) as exc:
            self.info.set('บันทึกบอร์ดไม่สำเร็จ: '+str(exc));return
        snapshot['calibration_profiles']=profiles
        snapshot['confidence_settings']=dict(self.preferences)
        self.board_saving=True;self.cancel_scan();self.refresh_checklist()
        self.info.set('กำลังบันทึกผลบอร์ดและสำเนาภาพหลักฐาน…')
        def start():
            self.save_board_timer=None
            if self.busy:
                self.save_board_timer=self.root.after(25,start);return
            def done(folder):
                self.board_saving=False;self.last_saved_board=folder
                self.last_saved_signature=signature
                self.board_history.invalidate()
                if self.begin_new_board():
                    self.info.set('บันทึกบอร์ด '+snapshot['board_number']+' แล้ว · พร้อมเริ่มบอร์ดใหม่')
                    self.board_number_entry.focus_set()
                else:
                    self.info.set('บันทึกบอร์ดแล้ว แต่เริ่มบอร์ดใหม่ไม่สำเร็จ · '+str(folder))
            def failed(exc):
                self.board_saving=False
                self.info.set('บันทึกบอร์ดไม่สำเร็จ: '+str(exc));self.refresh_checklist()
            self.job(lambda:export_board(snapshot,self.report_root/'completed_boards'),done,failed)
        start()

    def begin_new_board(self):
        if self.board_saving:return False
        if not self.persist_board_number():return False
        self.cancel_scan()
        try:
            self.checklist.new_board();self.reset_scan_board()
            self.board_number.set('');self.diode_live.clear();self.diode_visual_live.clear();self.last_saved_signature=None
            self.ic_observations.clear();self.ic_observation_frame=None
            self.pin_calibrations,self.diode_direction,self.calibration_issues=read_calibrations(self.calibration_root)
            self.info_component=None
            self.component.set('ไม่ระบุ');self.clear_result();self.refresh_checklist()
            self.info.set('เริ่มตรวจบอร์ดชุดใหม่แล้ว')
            return True
        except Exception as exc:
            self.error(exc)
            return False
