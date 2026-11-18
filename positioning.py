"""Positioning without user training: reference features and pretrained, local OCR."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re

import cv2
import numpy as np

from board import COMPONENTS
from scanner import BoardLocator, Registration, transform


@dataclass
class LabelHit:
    component: str
    confidence: float
    quad: np.ndarray
    source: str = 'PP-OCRv4'


def parse_labels(rows, minimum=.90):
    """Accept explicit designators only; never turn 010, 224, RI, or R? into an ID."""
    found={};duplicates=set()
    for row in rows or []:
        quad,text,score=row[:3]
        if len(row)>=6 and len(row[3])==len(row[4]):
            chars=[(str(char).upper(),np.float32(box),float(conf)) for char,box,conf in zip(row[4],row[3],row[5])]
        else:
            points=np.float32(quad);chars=[]
            for i,char in enumerate(str(text)):
                start,end=i/len(text),(i+1)/len(text)
                box=np.array([points[0]+(points[1]-points[0])*start,points[0]+(points[1]-points[0])*end,
                              points[3]+(points[2]-points[3])*end,points[3]+(points[2]-points[3])*start])
                chars.append((char.upper(),box,float(score)))
        chars=[item for item in chars if item[0] not in ' +-\t\n']
        value=''.join(item[0] for item in chars)
        if not re.fullmatch(r'(?:[RDQU][1-9][0-9]*)+',value):continue
        for match in re.finditer(r'[RDQU][1-9][0-9]*',value):
            chunk=chars[match.start():match.end()]
            confidence=float(np.mean([item[2] for item in chunk]))
            if confidence<minimum:continue
            name=match.group()
            box=np.float32([chunk[0][1][0],chunk[-1][1][1],chunk[-1][1][2],chunk[0][1][3]])
            if name in found:
                duplicates.add(name)
            else:found[name]=LabelHit(name,confidence,box)
    return [hit for name,hit in found.items() if name not in duplicates]


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1,
                    det_use_cuda=False,rec_use_cuda=False,cls_use_cuda=False)


@lru_cache(maxsize=1)
def modern_ocr_engine():
    from rapidocr import RapidOCR
    return RapidOCR(params={'Global.log_level':'error',
                           'EngineConfig.onnxruntime.intra_op_num_threads':2,
                           'EngineConfig.onnxruntime.inter_op_num_threads':1})


def modern_rows(output):
    rows=[]
    for index,(box,text,score) in enumerate(zip(output.boxes if output.boxes is not None else [],
                                              output.txts or [],output.scores or [])):
        row=[np.float32(box).tolist(),text,float(score)]
        words=output.word_results[index] if output.word_results and index<len(output.word_results) else None
        if words and all(item[2] is not None for item in words):
            row.extend([[item[2] for item in words],[item[0] for item in words],[float(item[1]) for item in words]])
        rows.append(row)
    return rows


def combine_hits(hits):
    """Fuse the same label from two engines, without accepting two different locations."""
    grouped={}
    for hit in hits:grouped.setdefault(hit.component,[]).append(hit)
    result=[]
    for candidates in grouped.values():
        best=max(candidates,key=lambda hit:hit.confidence)
        height=max(1,float(np.linalg.norm(best.quad[3]-best.quad[0])))
        if any(np.linalg.norm(hit.quad.mean(axis=0)-best.quad.mean(axis=0))>height for hit in candidates):continue
        result.append(best)
    return result


class LabelReader:
    def __init__(self,engine=None):
        self.engine=engine
        self.detail='PP-OCRv4 + PP-OCRv6'

    def read(self,frame,turn=0):
        rotated=np.ascontiguousarray(np.rot90(frame,turn))
        h,w=rotated.shape[:2]
        scale=min(2. if max(h,w)<600 else 1.,1400/max(h,w))
        image=cv2.resize(rotated,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_CUBIC)
        rows,_=(self.engine or ocr_engine())(image,return_word_box=True,text_score=.65)
        hits=parse_labels(rows)
        if self.engine is None:
            try:
                output=modern_ocr_engine()(image,return_word_box=True,return_single_char_box=True,text_score=.65)
                modern=parse_labels(modern_rows(output))
                for hit in modern:hit.source='PP-OCRv6'
                hits=combine_hits(hits+modern)
            except Exception as exc:
                self.detail='PP-OCRv4 · ตัวอ่าน v6 ยังใช้ไม่ได้: '+str(exc)
        original_height,original_width=frame.shape[:2]
        for hit in hits:
            points=hit.quad/np.array([image.shape[1]/w,image.shape[0]/h],np.float32)
            u,v=points[:,0].copy(),points[:,1].copy()
            if turn==1:points[:,0],points[:,1]=original_width-1-v,u
            elif turn==2:points[:,0],points[:,1]=original_width-1-u,original_height-1-v
            elif turn==3:points[:,0],points[:,1]=v,original_height-1-u
            hit.quad=points
        return hits


def nominal_label_center(name):
    x1,y1,x2,y2=COMPONENTS[name]['box']
    if name=='U4':return np.float32([376,343])
    if name=='U1':return np.float32([368,74])
    return np.float32([(x1+x2)/2,y1-7])


def fit_label_anchors(reference,hits):
    """Require three agreeing names, relative spacing, text direction and scale."""
    common=[hit for hit in hits if hit.component in reference and hit.component in COMPONENTS]
    if len(common)<3:return None
    source=np.float32([reference[hit.component].quad.mean(axis=0) for hit in common])
    target=np.float32([hit.quad.mean(axis=0) for hit in common])
    heights=[np.linalg.norm(hit.quad[3]-hit.quad[0]) for hit in common]
    tolerance=max(3.,float(np.median(heights))*.12)
    matrix,mask=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,
                                          ransacReprojThreshold=tolerance,maxIters=3000,confidence=.999)
    if matrix is None or mask is None:return None
    # Four names spanning rows and columns also support mild camera skew.
    if len(common)>=4 and cv2.contourArea(cv2.convexHull(source))>100:
        affine=np.linalg.lstsq(np.column_stack((source,np.ones(len(source)))),target,rcond=None)[0].T
        residual=np.linalg.norm(source@affine[:,:2].T+affine[:,2]-target,axis=1)
        singular=np.linalg.svd(affine[:,:2],compute_uv=False)
        if residual.max()<=tolerance*1.6 and singular.min()>0 and singular.max()/singular.min()<1.8:
            matrix=affine;mask=np.ones((len(common),1),np.uint8)
    good=mask.ravel().astype(bool)
    if good.sum()<3 or good.mean()<.75:return None
    matrix=np.vstack((matrix,[0.,0.,1.]))
    if np.linalg.det(matrix[:2,:2])<=0:return None
    source_span=float(np.max(np.linalg.norm(source[good]-source[good].mean(axis=0),axis=1)))
    if source_span<15:return None
    evidence=[]
    for hit,valid in zip(common,good):
        if not valid:continue
        ref=reference[hit.component]
        projected=transform(ref.quad,matrix)
        a=projected[1]-projected[0];b=hit.quad[1]-hit.quad[0]
        cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6))
        ratio=np.linalg.norm(hit.quad[3]-hit.quad[0])/max(1e-6,np.linalg.norm(projected[3]-projected[0]))
        # Different detectors include different margins around a glyph. Four agreeing
        # positions constrain the transform more strongly than a single text-box height.
        minimum,maximum=(.40,2.2) if good.sum()>=4 else (.60,1.65)
        if cosine<.96 or not minimum<ratio<maximum:return None
        evidence.append(dict(component=hit.component,confidence=hit.confidence,quad=hit.quad.tolist(),source=hit.source))
    errors=np.linalg.norm(transform(source[good],matrix)-target[good],axis=1)
    return Registration(matrix,int(good.sum()),float(good.mean()),float(np.median(errors)),
                        method='ocr_reference_anchors',anchors=evidence)


class PositionLocator(BoardLocator):
    def __init__(self,path,reader=None):
        super().__init__(path)
        self.reader=reader or LabelReader()
        self.reference_labels=None
        self.operator_anchors={}
        manifest_path=self.path.with_suffix('.json')
        if manifest_path.is_file():
            manifest=json.loads(manifest_path.read_text('utf-8'))
            if manifest.get('image_sha256')==self.fingerprint:
                self.operator_anchors=manifest.get('anchors',{})
        self.hits=[]
        self.detail='พร้อมระบุตำแหน่ง · ไม่ต้องเทรนโมเดลตำแหน่ง'

    def locate(self,frame):
        self.hits=[]
        registration=super().locate(frame)
        if registration is not None:
            self.detail=f'เทียบภาพอ้างอิง · ตรงกัน {registration.inliers} จุด · ไม่ใช้ best.pt ระบุตำแหน่ง'
            return registration
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray,cv2.CV_64F).var()<5:
            self.detail='ยังไม่มีรายละเอียดภาพเพียงพอให้อ่านตำแหน่ง';return None
        try:
            observed={}
            for turn in range(4):
                hits=self.reader.read(frame,turn)
                for hit in hits:
                    previous=observed.get(hit.component)
                    if previous is None or hit.confidence>previous.confidence:observed[hit.component]=hit
                self.hits=list(observed.values())
                if len(hits)<3:continue
                if self.reference_labels is None:
                    self.reference_labels={}
                    for name,center in self.operator_anchors.items():
                        if name not in COMPONENTS:continue
                        x,y=map(float,center);half=4*len(name)
                        quad=np.float32([[x-half,y-7],[x+half,y-7],[x+half,y+7],[x-half,y+7]])
                        self.reference_labels[name]=LabelHit(name,1.,quad,source='operator_reference')
                    for hit in self.reader.read(self.reference):
                        if hit.component not in COMPONENTS:continue
                        scale=np.array([431/self.reference.shape[1],447/self.reference.shape[0]])
                        hit.quad=hit.quad*scale
                        if np.linalg.norm(hit.quad.mean(axis=0)-nominal_label_center(hit.component))<15:
                            self.reference_labels[hit.component]=hit
                registration=fit_label_anchors(self.reference_labels,hits)
                if registration:
                    self.hits=hits
                    names=', '.join(item['component'] for item in registration.anchors)
                    self.detail='OCR ยืนยันตำแหน่งร่วมกับภาพอ้างอิง: '+names
                    return registration
            names=', '.join(hit.component for hit in self.hits)
            unknown=[hit.component for hit in self.hits if hit.component not in COMPONENTS]
            if unknown:self.detail='OCR อ่านได้ '+names+' · ชื่ออยู่นอกผัง: '+', '.join(unknown)+' · ตรวจรุ่นบอร์ด/ภาพอ้างอิง'
            elif names:self.detail='OCR อ่านได้ '+names+' · '+('ระยะ/มุมยังไม่ตรงอ้างอิง · ตั้งอ้างอิงจากกล้องบริเวณนี้' if len(self.hits)>=3 else 'ต้องมีชื่อที่ตรงอ้างอิงอย่างน้อย 3 จุด')
            else:self.detail='ยังอ่านชื่อไม่ได้ · ให้เห็นตัวหนังสือชัด หรือตั้งภาพอ้างอิงบอร์ดจริง'
        except Exception as exc:
            self.detail='OCR ยังใช้ไม่ได้: '+str(exc)+' · เทียบภาพอ้างอิงยังใช้งานได้'
        return None

    def targets(self,frame,registration,results):
        targets=super().targets(frame,registration,results)
        if registration.method=='ocr_reference_anchors':
            verified={item['component'] for item in registration.anchors}
            targets=[target for target in targets if target.component in verified]
        return targets
