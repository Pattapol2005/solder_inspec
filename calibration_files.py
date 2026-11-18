"""Resolve reference images in the current checkout, including legacy profiles."""
import hashlib
import json
from pathlib import Path, PureWindowsPath


def load_calibration_profile(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    kind = data['kind']
    if kind not in ('U1', 'U4', 'diode'):
        raise ValueError('ไม่รู้จักชนิดอ้างอิงใน '+str(path))
    stored = str(data['image'])
    parts = PureWindowsPath(stored).parts
    absolute = Path(stored).is_absolute() or PureWindowsPath(stored).is_absolute()
    if absolute and len(parts) >= 3 and parts[-3].casefold() == kind.casefold():
        # Rebase the exact saved revision. Never pick a newer reference merely
        # because it has the same filename or fall back to another checkout.
        local_kind = path.parent if path.parent.name.casefold() == kind.casefold() else path.parent.parent
        image = local_kind / parts[-2] / parts[-1]
    elif absolute:
        image = Path(stored)
    else:
        image = path.parent / Path(stored.replace('\\', '/'))
    if not image.is_file():
        raise ValueError(f'ไม่พบภาพอ้างอิง {kind}: {image} · กรุณาคัดลอกภาพอ้างอิงรุ่นนี้ให้ครบ หรือตั้งอ้างอิงใหม่')
    expected = data.get('image_sha256')
    if expected and hashlib.sha256(image.read_bytes()).hexdigest() != expected:
        raise ValueError(f'ภาพอ้างอิง {kind} ไม่ตรงกับจุดที่บันทึกไว้: {image}')
    return dict(data, image=str(image.resolve()))
