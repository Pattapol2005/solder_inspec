"""Inspection rules and portable image/report storage; no trained weights bundled."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

CLASSES = ('good', 'bridge', 'excess', 'missing')
THAI = ('ปกติ', 'ตะกั่วเชื่อมกัน', 'ตะกั่วมาก', 'อุปกรณ์/รอยบัดกรีขาดหาย')
REGIONS = ('0805 R1-R20', '0603 R21-R40', '1206 R41-R49 / D1-D5',
           'SOP14 U1', 'SOT-23 Q1-Q6', 'SSOP28 U4', 'Other')


@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    polygon: tuple | None = None
    native_class_id: int | None = None
    native_class_name: str | None = None


def prediction_quads(prediction):
    """Read actual OBB corners; retain compatibility with straight-box models."""
    obb = getattr(prediction, 'obb', None)
    boxes = getattr(prediction, 'boxes', None)
    if obb is not None:
        polygons = obb.xyxyxyxy.cpu().tolist()
        classes, scores = obb.cls.cpu().tolist(), obb.conf.cpu().tolist()
    elif boxes is not None:
        polygons = [[[a,b],[c,b],[c,d],[a,d]] for a,b,c,d in boxes.xyxy.cpu().tolist()]
        classes, scores = boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
    else:
        raise ValueError('ผลโมเดลไม่มีกรอบ OBB หรือ Bounding Box ที่รองรับ')
    for points, class_id, score in zip(polygons, classes, scores):
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(points).all() or not np.isfinite(score):
            continue
        yield points, int(class_id), float(score)


def detection_from_quad(points, class_id, confidence, native_class_id=None, native_class_name=None):
    points = np.asarray(points, dtype=float).reshape(4, 2)
    return Detection(class_id, confidence, tuple(map(float, (*points.min(axis=0), *points.max(axis=0)))),
                     tuple(tuple(map(float, p)) for p in points),native_class_id,native_class_name)


def detection_polygon(detection):
    if detection.polygon is not None:return np.float32(detection.polygon)
    x1,y1,x2,y2=detection.box
    return np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


def hidden_by_missing(points,missing_quads):
    polygon=np.float32(points)
    area=cv2.contourArea(polygon)
    if area<1:return False
    for missing in missing_quads:
        intersection,_=cv2.intersectConvexConvex(polygon,np.float32(missing))
        if intersection>=area*.90:return True
    return False


def visible_detections(detections):
    """Missing takes display precedence over proposals contained inside it."""
    missing=[(i,d) for i,d in enumerate(detections) if d.class_id==3]
    if not missing:return list(detections)
    missing.sort(key=lambda item:(-cv2.contourArea(detection_polygon(item[1])),-item[1].confidence))
    masks=[];kept_missing=set()
    for index,detection in missing:
        polygon=detection_polygon(detection)
        if hidden_by_missing(polygon,masks):continue
        masks.append(polygon);kept_missing.add(index)
    visible=[]
    for index,detection in enumerate(detections):
        if detection.class_id==3:
            if index in kept_missing:visible.append(detection)
        elif not hidden_by_missing(detection_polygon(detection),masks):
            visible.append(detection)
    return visible


def quality_detections(prediction, class_map, threshold, model_names=None):
    """The result/checkpoint names are authoritative; never guess native ID order."""
    names = getattr(prediction,'names',None) or model_names
    mapping = validate_names(names) if names is not None else class_map
    detections=[]
    for points,native_id,score in prediction_quads(prediction):
        if score < threshold:continue
        if native_id not in mapping:raise ValueError(f'ไม่รู้จัก class ID {native_id} ในโมเดลตรวจ')
        canonical_id=mapping[native_id]
        detections.append(detection_from_quad(points,canonical_id,score,native_id,CLASSES[canonical_id]))
    return visible_detections(detections)


def draw_detection(image, detection):
    color = (80, 210, 80) if detection.class_id == 0 else (40, 70, 240)
    x1,y1,x2,y2 = map(int, detection.box)
    if detection.polygon is not None:
        cv2.polylines(image, [np.int32(detection.polygon)], True, color, 2)
    else:
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
    cv2.putText(image, f'{CLASSES[detection.class_id]} {detection.confidence:.2f}',
                (x1,max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('อ่านภาพไม่ได้ กรุณาเลือกไฟล์ภาพที่ถูกต้อง')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('บันทึกภาพไม่สำเร็จ')
    encoded.tofile(str(path))


def validate_names(names):
    """Map external model IDs to inspection IDs without assuming training order."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    normalized = {int(i): str(name).strip().lower() for i, name in items}
    if len(normalized) != len(CLASSES) or set(normalized.values()) != set(CLASSES):
        raise ValueError('โมเดลต้องมี 4 คลาส: good, bridge, excess, missing (เรียงลำดับใดก็ได้)')
    return {i: CLASSES.index(name) for i, name in normalized.items()}


def decision(detections, expected, threshold):
    """PASS is only a region-level candidate with explicitly confirmed coverage."""
    if not 0 < threshold <= 1 or expected < 0:
        raise ValueError('Invalid inspection settings')
    accepted = [d for d in detections if d.confidence >= threshold]
    if any(d.class_id not in range(4) for d in accepted):
        return 'REVIEW', 'พบคลาสที่ไม่รองรับ'
    if any(d.class_id != 0 for d in accepted):
        return 'FAIL', 'พบรอยบัดกรีที่สงสัยว่าบกพร่อง'
    if not accepted:
        return 'REVIEW', 'ไม่พบจุดตรวจที่มั่นใจเพียงพอ'
    if expected == 0:
        return 'REVIEW', 'ยังไม่ได้ระบุจำนวนจุดตรวจที่คาดหวังในภาพ'
    if len(accepted) != expected:
        return 'REVIEW', f'พบ {len(accepted)} จุด แต่คาดหวัง {expected} จุด'
    return 'PASS', 'จุดที่ตรวจพบในภาพนี้เป็นปกติ — ควรตรวจยืนยันด้วยคน'


def draw_pin_number(image,component,pin,quad,color):
    # On the dense SSOP bank only the number fits inside a single lead.
    x,y=np.rint(np.mean(quad,axis=0)).astype(int)
    text=f'{component}/{pin}'
    font=cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text,font,.4,1)[0][0]+4>np.ptp(quad,axis=0)[0]:
        text=str(pin)
    (w,h),baseline=cv2.getTextSize(text,font,.4,1)
    left=max(0,min(x-w//2,image.shape[1]-w-2))
    bottom=max(h+1,min(y+h//2,image.shape[0]-baseline-1))
    cv2.rectangle(image,(left-1,bottom-h-2),(left+w+1,bottom+baseline),(25,34,40),-1)
    cv2.putText(image,text,(left,bottom),font,.4,color,1,cv2.LINE_AA)


def save_report(root, raw, annotated, detections, status, reason, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8]
    folder = root / report_id
    folder.mkdir()
    write_image(folder / 'raw.png', raw)
    write_image(folder / 'result.png', annotated)
    record = dict(id=report_id, timestamp=datetime.now().astimezone().isoformat(),
                  status=status, reason=reason, **metadata,
                  detections=[asdict(d) for d in detections])
    (folder / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), 'utf-8')
    csv_path = root / 'summary.csv'
    fields = ['id', 'timestamp', 'status', 'reason', 'region', 'expected', 'threshold', 'model', 'count']
    exists = csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({**{key: record.get(key, '') for key in fields}, 'count': len(detections)})
    return folder
