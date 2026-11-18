"""Self-contained board snapshots with the evidence used by each physical IC pin."""
from datetime import datetime
from pathlib import Path
import copy
import csv
import json
import shutil
import re
from uuid import uuid4

from board import COMPONENTS
from component_info import failure_reason
from evidence import save_board_images


def validate_board_number(value):
    value=str(value or '').strip()
    if not value or len(value)>80 or any(ord(c)<32 for c in value):
        raise ValueError('กรุณาใส่เลขประจำบอร์ด 1–80 ตัวอักษรก่อนบันทึก')
    return value


def inspected_component(name,entry):
    if not entry:return False
    if entry.get('status') not in ('PASS','FAIL'):return False
    if entry.get('status')=='FAIL' and entry.get('component_missing'):return True
    if name.startswith('D'):
        polarity=entry.get('diode_polarity_state')
        if polarity not in ('PASS','FAIL') and not (polarity=='NOT_APPLICABLE' and entry.get('status')=='FAIL'):return False
    if name in ('U1','U4'):
        total=COMPONENTS[name]['expected'];coverage=entry.get('coverage',{})
        return all(coverage.get(str(pin),{}).get('confirmed',False) and coverage[str(pin)].get('status') in ('PASS','FAIL') and
                   coverage[str(pin)].get('confirmed_by') not in ('latest_frame_component_count','full_component_detection_count')
                   for pin in range(1,total+1))
    return entry.get('complete',False) or (entry.get('status')=='FAIL' and
             entry.get('confirmed_by') in ('latest_frame_component_count','full_component_detection_count'))


def scanned_component(name,entry):
    return (name in COMPONENTS and bool(entry) and
            entry.get('status') in ('PASS','FAIL','REVIEW') and bool(entry.get('report')))


def export_board(session,root):
    number=validate_board_number(session.get('board_number'))
    records=session['results']
    pending=[name for name in COMPONENTS if not scanned_component(name,records.get(name))]
    if pending:raise ValueError('ยังสแกนไม่ครบทุกอุปกรณ์: '+', '.join(pending))
    # Validate references before copying the board's potentially large evidence.
    for kind,profile in session.get('calibration_profiles',{}).items():
        if kind not in ('U1','U4','diode'):raise ValueError('ไม่รู้จักชนิดอ้างอิงในรายงานบอร์ด')
        source=Path(profile['image'])
        if not source.is_file():raise ValueError(f'ไม่พบภาพอ้างอิง {kind}: {source}')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    slug=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',number).strip(' .')[:48] or 'unnumbered'
    destination=root/('board_'+slug+'_'+stamp+'_'+uuid4().hex[:6])
    staging=root/('.partial_'+destination.name);staging.mkdir(exist_ok=False)
    snapshot=copy.deepcopy(session);snapshot['board_number']=number;sources={}
    for entry in snapshot['results'].values():
        entry['history_count']=len(entry.pop('history',[]))
        if entry.get('status')=='FAIL':entry['failure_reason']=failure_reason(entry)
        for item in [entry,*entry.get('coverage',{}).values()]:
            if not item.get('report'):continue
            source=Path(item['report']).resolve()
            if not (source/'result.json').is_file():raise ValueError('ไม่พบหลักฐาน '+str(source))
            if source not in sources:
                target=Path('evidence')/f'{len(sources)+1:03d}_{source.name}'
                sources[source]=target
                shutil.copytree(source,staging/target)
            item['report']=sources[source].as_posix()
    for kind,profile in snapshot.get('calibration_profiles',{}).items():
        if kind not in ('U1','U4','diode'):raise ValueError('ไม่รู้จักชนิดอ้างอิงในรายงานบอร์ด')
        source=Path(profile['image'])
        target=Path('calibrations')/kind/'reference.png'
        (staging/target).parent.mkdir(parents=True,exist_ok=True)
        try:shutil.copy2(source,staging/target)
        except OSError as exc:
            raise OSError(f'คัดลอกภาพอ้างอิง {kind} ไม่สำเร็จ: {source} -> {staging/target}: {exc}') from exc
        profile['image']=target.as_posix()
        (staging/target.parent/'calibration.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    overall='FAIL' if any(e['status']=='FAIL' for e in records.values()) else (
        'PASS' if all(inspected_component(n,records[n]) for n in COMPONENTS) else 'REVIEW')
    snapshot.update(schema_version=1,saved_at=datetime.now().astimezone().isoformat(),
                    inspected_components=len(COMPONENTS),overall_status=overall,
                    evidence_reports=len(sources))
    (staging/'board.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    with (staging/'summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['board_id','board_number','component','status','diode_polarity','covered','total','inspection_complete','pin_coverage_complete','report'])
        writer.writeheader()
        for name in COMPONENTS:
            entry=snapshot['results'][name]
            writer.writerow(dict(board_id=snapshot['board_id'],board_number=number,component=name,status=entry['status'],diode_polarity=entry.get('diode_polarity_state',''),
                covered=entry.get('covered',0),total=COMPONENTS[name]['expected'],inspection_complete=inspected_component(name,entry),
                pin_coverage_complete=entry.get('complete',False),report=entry['report']))
    with (staging/'ic_pins.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['board_number','component','pin','status','confirmed','report'])
        writer.writeheader()
        for name in ('U1','U4'):
            for pin in range(1,COMPONENTS[name]['expected']+1):
                record=snapshot['results'][name].get('coverage',{}).get(str(pin),{})
                writer.writerow(dict(board_number=number,component=name,pin=pin,status=record.get('status','PENDING'),
                                     confirmed=record.get('confirmed',False),report=record.get('report','')))
    save_board_images(snapshot,staging)
    staging.replace(destination)
    return destination
