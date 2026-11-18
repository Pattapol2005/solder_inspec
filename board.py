"""Reference board selection, microscope ROI geometry, and per-board checklist."""
import csv
import json
import copy
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def components():
    items = {}
    ys = [91, 128, 165, 202, 239, 276, 313, 350, 387, 424]
    def add(name, box, expected, group):
        items[name] = dict(box=box, expected=expected, region=group)
    for i,y in enumerate(ys):
        add(f'R{i+1}', (65,y-12,108,y+12),2,'0805 R1-R20')
        add(f'R{i+11}', (16,y-12,61,y+12),2,'0805 R1-R20')
        add(f'R{40-i}', (115,y-12,153,y+12),2,'0603 R21-R40')
        add(f'R{i+21}', (155,y-12,193,y+12),2,'0603 R21-R40')
    for i,y in enumerate([96,136,176,216,256,295,336,376,417]):
        add(f'R{i+41}', (193,y-14,251,y+14),2,'1206 R41-R49 / D1-D5')
    for i,y in enumerate([96,136,176,216,256]):
        add(f'D{i+1}', (260,y-15,317,y+15),2,'1206 R41-R49 / D1-D5')
    for i,box in enumerate([(332,234,360,267),(369,234,397,267),(255,293,286,324),
                            (292,293,323,324),(332,293,363,324),(369,293,400,324)]):
        add(f'Q{i+1}',box,3,'SOT-23 Q1-Q6')
    add('U1',(331,71,404,184),14,'SOP14 U1')
    add('U4',(263,337,348,434),28,'SSOP28 U4')
    return {name:items[name] for name in sorted(items,key=lambda n:({'R':0,'D':1,'Q':2,'U':3}[n[0]],int(n[1:])))}


COMPONENTS = components()


def parse_joints(text, total):
    """Operator identifies the pads visible in this microscope capture."""
    joints=set()
    try:
        for part in text.replace(' ', '').split(','):
            if '-' in part:
                start,end=map(int,part.split('-'))
                if not 1<=start<=end<=total:raise ValueError()
                joints.update(range(start,end+1))
            else:joints.add(int(part))
        if not joints or min(joints)<1 or max(joints)>total:raise ValueError()
    except ValueError:
        raise ValueError(f'ระบุจุดที่เห็นเป็น 1-2 หรือ 1,3 โดยใช้หมายเลข 1 ถึง {total}') from None
    return sorted(joints)


def crop_input(frame, roi=None, rotation=0):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=map(int,roi or (0,0,w,h))
    if not (0<=x1<x2<=w and 0<=y1<y2<=h):
        raise ValueError('กรอบตรวจต้องอยู่ในภาพและมีขนาดมากกว่า 0')
    if rotation not in (0,90,180,270):
        raise ValueError('Invalid crop rotation')
    crop=frame[y1:y2,x1:x2].copy()
    if rotation:
        crop=cv2.rotate(crop,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
    return crop,(x1,y1,x2,y2)


def restore_box(box, roi, rotation):
    """Return detector coordinates in the original microscope frame."""
    x1,y1,x2,y2=map(float,box)
    ox,oy,ex,ey=roi
    w,h=ex-ox,ey-oy
    points=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=float)
    u,v=points[:,0].copy(),points[:,1].copy()
    if rotation==90:points[:,0],points[:,1]=v,h-u
    elif rotation==180:points[:,0],points[:,1]=w-u,h-v
    elif rotation==270:points[:,0],points[:,1]=w-v,u
    points[:,0]=np.clip(points[:,0],0,w)+ox
    points[:,1]=np.clip(points[:,1],0,h)+oy
    return tuple(map(float,(*points.min(axis=0),*points.max(axis=0))))


class BoardChecklist:
    def __init__(self,folder,load_existing=True):
        self.folder=Path(folder)
        self.revision=0
        self.path=self.folder/'current.json'
        self.session=json.loads(self.path.read_text('utf-8')) if load_existing and self.path.exists() else self.fresh()
        if not isinstance(self.session,dict):raise ValueError('ไฟล์เช็กลิสต์ไม่ถูกต้อง')
        if not isinstance(self.session.get('results'),dict) or not self.session.get('board_id'):
            raise ValueError('ไฟล์เช็กลิสต์ไม่ถูกต้อง')
        for name,item in self.session['results'].items():
            if name not in COMPONENTS or not isinstance(item,dict) or item.get('status') not in ('PASS','FAIL','REVIEW'):
                raise ValueError('ไฟล์เช็กลิสต์มีผลตรวจไม่ถูกต้อง')
            if name.startswith('D') and not item.get('diode_polarity_state'):
                if 'legacy_solder_result' not in item:item['legacy_solder_result']=copy.deepcopy({k:v for k,v in item.items() if k!='history'})
                item['complete']=False
                if item['status']=='PASS':item['status']='REVIEW'
            if name in ('U1','U4'):
                if item.get('total',COMPONENTS[name]['expected'])!=COMPONENTS[name]['expected']:
                    item['legacy_pin_count_result']=copy.deepcopy({k:v for k,v in item.items() if k!='history'})
                    item.update(status='REVIEW',coverage={},covered=0,complete=False,total=COMPONENTS[name]['expected'])
                coverage=item.get('coverage',{})
                invalid=[pin for pin,value in coverage.items() if value.get('confirmed_by') in ('latest_frame_component_count','full_component_detection_count')]
                if invalid:
                    item['legacy_unmapped_result']=copy.deepcopy({k:v for k,v in item.items() if k!='history'})
                    for pin in invalid:coverage.pop(pin)
                    item.update(covered=sum(bool(v.get('confirmed')) for v in coverage.values()),complete=False)
                    if item['status']=='PASS':item['status']='REVIEW'

    @staticmethod
    def fresh():
        return dict(board_id=datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid4().hex[:6],results={})

    def save(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.session,ensure_ascii=False,indent=2),'utf-8')
        temp.replace(self.path)
        self.revision += 1

    def frame_snapshot(self):
        """Keep a stable set of records while inference and file preparation run."""
        stage = copy.copy(self)
        stage.session = dict(self.session, results=dict(self.session['results']))
        for key in ('inspection_resets','calibration_resets'):
            if key in stage.session:
                stage.session[key] = list(stage.session[key])
        return stage

    def prepare_frame(self, reports):
        changed = [row for row in reports if row[3]]
        if not changed:
            return None
        for result, report, _, _ in changed:
            meta = result[5]
            pins = meta.get('merge_visible_pins',False)
            keep_ic_coverage=meta['component'] in ('U1','U4') and bool(meta.get('defect_regions'))
            self.record(meta['component'],result[3],result[4],report,meta['expected'],None,0,
                        meta['joints'],confirmed_by='calibrated_ic_pin_reference' if pins else 'latest_frame_component_count',
                        joint_states=meta['joint_states'],replace_previous=not pins and not keep_ic_coverage,replace_joint_states=pins,
                        diode_polarity=meta.get('diode_polarity',{}).get('status'),persist=False,
                        component_missing=meta.get('component_missing',False))
        self.folder.mkdir(parents=True,exist_ok=True)
        staged = tempfile.TemporaryDirectory(dir=self.folder,prefix='.frame-')
        try:
            (Path(staged.name)/'current.json').write_text(json.dumps(self.session,ensure_ascii=False,separators=(',',':')),'utf-8')
        except Exception:
            staged.cleanup()
            raise
        return staged

    def commit_frame(self, stage, staged_file):
        if self.revision != stage.revision or self.session['board_id'] != stage.session['board_id']:
            return False
        if staged_file is not None:
            (Path(staged_file.name)/'current.json').replace(self.path)
            self.session['results'] = stage.session['results']
            self.revision += 1
        return True

    def record(self,name,state,reason,report,expected,roi,rotation,joints=None,coverage_confirmed=True,confirmed_by='detection_count',joint_states=None,replace_previous=False,replace_joint_states=False,diode_polarity=None,persist=True,component_missing=False):
        if name not in COMPONENTS or state not in ('PASS','FAIL','REVIEW'):
            raise ValueError('Invalid component inspection')
        previous=self.session['results'].get(name)
        lock_pass=getattr(self,'lock_pass',False)
        if lock_pass and previous and previous.get('status')=='PASS' and previous.get('complete'):
            if not name.startswith('D') or previous.get('diode_polarity_state')=='PASS':return
        report=Path(report)
        if not (report/'result.json').is_file():
            raise ValueError('ต้องบันทึกภาพหลักฐานก่อนอัปเดตเช็กลิสต์')
        total=COMPONENTS[name]['expected']
        joints=list(joints or range(1,total+1))
        if len(set(joints))!=len(joints) or not all(isinstance(j,int) and 1<=j<=total for j in joints):
            raise ValueError('Invalid joint coverage')
        if expected!=len(joints):raise ValueError('จำนวนจุดไม่ตรงกับขาที่ระบุ')
        if joint_states is not None:
            joint_states={int(j):s for j,s in joint_states.items()}
            if set(joint_states)!=set(joints) or any(s not in ('PASS','FAIL','REVIEW') for s in joint_states.values()):
                raise ValueError('Invalid per-joint results')
        entry=dict(status=state,reason=reason,report=str(report.resolve()),expected=expected,
                   roi=list(roi) if roi is not None else None,rotation=rotation,joints=joints,confirmed_by=confirmed_by,time=datetime.now().astimezone().isoformat())
        if component_missing:
            if state!='FAIL':raise ValueError('ผล missing ต้องเป็น FAIL')
            entry.update(component_missing=True,confirmed_by='component_missing_detection')
        if diode_polarity is not None:
            if diode_polarity not in ('PASS','FAIL','REVIEW','NOT_APPLICABLE'):raise ValueError('Invalid diode polarity result')
            entry['diode_polarity_state']=diode_polarity
        previous=self.session['results'].get(name)
        entry['history']=(previous.get('history',[])+[{k:v for k,v in previous.items() if k!='history'}]) if previous else []
        if replace_previous or replace_joint_states:entry['history']=entry['history'][-50:]
        coverage=copy.deepcopy(previous.get('coverage',{})) if previous and not replace_previous and not component_missing else {}
        if name in ('U1','U4') and replace_joint_states:
            coverage={pin:value for pin,value in coverage.items() if value.get('confirmed_by') not in ('latest_frame_component_count','full_component_detection_count')}
        for joint in ([] if component_missing else joints):
            joint_state=joint_states[joint] if joint_states is not None else state
            if lock_pass and coverage.get(str(joint),{}).get('status')=='PASS' and coverage[str(joint)].get('confirmed'):
                continue
            if joint_states is not None and coverage.get(str(joint),{}).get('confirmed',False):
                if not replace_joint_states or joint_state=='REVIEW':continue
            coverage[str(joint)]=dict(status=joint_state,report=str(report.resolve()),confirmed=coverage_confirmed and joint_state!='REVIEW',confirmed_by=confirmed_by)
        covered=sum(v.get('confirmed',False) and v['status'] in ('PASS','FAIL') for v in coverage.values())
        entry.update(coverage=coverage,covered=covered,total=total,complete=covered==total)
        entry['status']='FAIL' if state=='FAIL' or any(v['status']=='FAIL' for v in coverage.values()) else ('PASS' if covered==total else 'REVIEW')
        if replace_previous:entry['status']=state
        self.session['results'][name]=entry
        if not persist:return
        try:self.save()
        except Exception:
            if previous is None:self.session['results'].pop(name)
            else:self.session['results'][name]=previous
            raise

    def new_board(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        (self.folder/(self.session['board_id']+'.json')).write_text(json.dumps(self.session,ensure_ascii=False,indent=2),'utf-8')
        old=self.session
        self.session=self.fresh()
        try:self.save()
        except Exception:
            self.session=old
            raise

    def reset_results(self):
        old=copy.deepcopy(self.session)
        self.session.setdefault('inspection_resets',[]).append(dict(
            time=datetime.now().astimezone().isoformat(),results=self.session['results']))
        self.session['results']={}
        try:self.save()
        except Exception:
            self.session=old
            raise

    def completed(self):
        return sum(bool(entry.get('complete') or entry.get('component_missing')) for entry in self.session['results'].values())

    def export_csv(self,path):
        with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
            fields=['board_id','component','status','time','report','reason','covered','total','complete']
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for name in COMPONENTS:
                item=self.session['results'].get(name,{})
                writer.writerow(dict(board_id=self.session['board_id'],component=name,
                                     **{k:item.get(k,{'status':'PENDING','covered':0,'total':COMPONENTS[name]['expected'],'complete':False}.get(k,'')) for k in fields[2:]}))
