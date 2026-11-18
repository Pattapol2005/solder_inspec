"""Small, atomic local preferences; independent of camera and model state."""
import json
import math
from pathlib import Path

DEFAULTS = {'inspection_confidence': .25, 'locator_confidence': .60}


def confidence_value(value):
    value = float(value)
    if not math.isfinite(value) or not .01 <= value <= 1:
        raise ValueError('Confidence ต้องอยู่ระหว่าง 0.01–1.00')
    return value


def load_preferences(path):
    result = dict(DEFAULTS)
    path = Path(path)
    if not path.is_file():
        return result, None
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            raise ValueError('รูปแบบไฟล์ตั้งค่าไม่ถูกต้อง')
        for key in DEFAULTS:
            if key in document:
                result[key] = confidence_value(document[key])
        return result, None
    except (ValueError, OSError, TypeError) as exc:
        return dict(DEFAULTS), str(exc)


def save_preferences(path, values):
    path = Path(path)
    document = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(previous, dict):document.update(previous)
        except (ValueError, OSError):
            pass
    document.update({key:confidence_value(values[key]) for key in DEFAULTS})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)
