"""Reference markers are not board components or solder quality classes."""
import cv2
import numpy as np
from board import COMPONENTS

ALIASES={'plus':'plus','polarity_plus':'plus','minus':'minus','polarity_minus':'minus',
         'ic_head':'ic_head','ic_pin1_dot':'ic_head'}


def normalize_name(name):
    name=str(name).strip()
    return ALIASES.get(name.lower(),name.upper())


def validate_locator_names(names):
    items=names.items() if isinstance(names,dict) else enumerate(names)
    mapping={int(i):normalize_name(name) for i,name in items}
    values=list(mapping.values());allowed=set(COMPONENTS)|set(ALIASES.values())
    if not set(COMPONENTS)<=set(values) or set(values)-allowed or len(values)!=len(set(values)):
        raise ValueError('โมเดล locator ต้องมีอุปกรณ์ 62 คลาส และเพิ่ม plus, minus, ic_head ได้ โดยชื่อไม่ซ้ำ')
    return mapping


def assigned_markers(hits,markers,name):
    assigned=[]
    for marker in markers:
        center=tuple(map(float,np.float32(marker.quad).mean(axis=0)))
        owners=[h.component for h in hits if cv2.pointPolygonTest(np.float32(h.quad),center,False)>=0]
        if owners==[name]:assigned.append(marker)
    return assigned


def diode_marker_reference(hits,markers,name):
    assigned=assigned_markers(hits,markers,name)
    plus=[h for h in assigned if h.component=='plus'];minus=[h for h in assigned if h.component=='minus']
    if len(plus)!=1 or len(minus)!=1:return None
    p=np.float32(plus[0].quad).mean(axis=0);m=np.float32(minus[0].quad).mean(axis=0)
    axis=m-p;distance=float(np.linalg.norm(axis))
    owner=next(h for h in hits if h.component==name)
    extent=float(np.max(np.ptp(np.float32(owner.quad),axis=0)))
    if distance<max(12.,extent*.25):return None
    return dict(axis=(axis/distance).tolist(),source='locator_plus_minus',plus_center=p.tolist(),minus_center=m.tolist(),
                plus_score=float(plus[0].confidence),minus_score=float(minus[0].confidence))
