"""Read-only component details from saved board results."""
import json
from pathlib import Path
from board import COMPONENTS


def failure_reason(entry):
    if entry.get('failure_reason'):return entry['failure_reason']
    reasons=[entry['reason']] if entry.get('reason') else []
    meta={}
    if entry.get('report'):
        try:meta=json.loads((Path(entry['report'])/'result.json').read_text('utf-8'))
        except (OSError,ValueError):pass
    if not isinstance(meta,dict):meta={}
    defects={1:'ตะกั่วเชื่อมขา (bridge)',2:'ตะกั่วมากเกิน (excess)',3:'ไม่พบอุปกรณ์/รอยบัดกรี (missing)'}
    kinds={d.get('class_id') for d in meta.get('detections',[]) if isinstance(d,dict)}
    reasons.extend(label for key,label in defects.items() if key in kinds)
    polarity=meta.get('diode_polarity',{})
    if polarity.get('status') in ('FAIL','NOT_APPLICABLE') and polarity.get('reason'):
        reasons.append(polarity['reason'])
    elif entry.get('diode_polarity_state')=='FAIL':
        reasons.append('ไดโอดกลับด้าน')
    pins=[pin for pin,item in entry.get('coverage',{}).items() if item.get('status')=='FAIL']
    if pins:reasons.append('จุด/ขาที่ไม่ผ่าน '+', '.join(sorted(pins,key=int)))
    return '\n'.join(dict.fromkeys(reasons)) or 'ผลตรวจไม่ผ่าน · ไม่มีรายละเอียดสาเหตุในหลักฐานเดิม'


def component_text(name,entry):
    if name not in COMPONENTS:return 'คลิกอุปกรณ์บนผังบอร์ดเพื่อดูผลตรวจของตัวนั้น'
    data=COMPONENTS[name]
    titles={'PASS':'ผ่าน','FAIL':'ไม่ผ่าน','REVIEW':'รอตรวจเพิ่ม','PENDING':'ยังไม่ตรวจ','NOT_APPLICABLE':'ไม่มีตัวอุปกรณ์'}
    state=entry.get('status','PENDING')
    if entry.get('component_missing'):
        return '\n'.join([name+' · MISSING (ไม่ผ่าน)',data['region'],
                          entry.get('reason','ตรวจพบ missing · ไม่พบตัวอุปกรณ์')])
    lines=[name+' · '+titles.get(state,state),data['region'],
           f"ตรวจยืนยันแล้ว {entry.get('covered',0)}/{data['expected']} จุด",
           'ตรวจครบแล้ว' if entry.get('complete') else 'ยังตรวจไม่ครบทุกจุด','',
           entry.get('reason','ยังไม่มีผลตรวจที่บันทึกไว้')]
    if name.startswith('D'):
        polarity={'PASS':'ถูกด้าน','FAIL':'กลับด้าน','REVIEW':'ยังอ่านด้านไม่ได้','NOT_APPLICABLE':'ไม่มีตัวอุปกรณ์'}
        lines+=['','ทิศทางไดโอด: '+polarity.get(entry.get('diode_polarity_state'),'ยังไม่ตรวจ')]
    lines+=['','ผลแต่ละขา' if name.startswith('U') else 'ผลแต่ละจุด']
    for pin in range(1,data['expected']+1):
        item=entry.get('coverage',{}).get(str(pin),{})
        confirmed=item.get('confirmed',False)
        lines.append(f"{pin}: "+titles.get(item.get('status','PENDING'),'รอตรวจเพิ่ม')+(' · ยืนยันแล้ว' if confirmed else ' · ยังไม่ยืนยัน'))
    if entry.get('time'):lines+=['','เวลาตรวจ: '+entry['time']]
    if state=='PASS' and entry.get('complete'):lines+=['','ล็อกผลผ่านไว้จนกว่าจะรีเซ็ตหรือเริ่มบอร์ดใหม่']
    return '\n'.join(lines)
