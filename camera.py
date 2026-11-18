"""Capture runs off the GUI thread. Each worker exclusively owns its device."""
import os
import threading
import time

import cv2

MICROSCOPE_NAME = 'USB 2.0 Camera'


def microscope_index():
    from cv2_enumerate_cameras import enumerate_cameras
    matches = [device.index for device in enumerate_cameras(cv2.CAP_DSHOW)
               if device.name.strip().casefold() == MICROSCOPE_NAME.casefold()]
    if not matches:
        raise RuntimeError('ไม่พบ USB 2.0 Camera · ตรวจสาย USB หรือเลือกกล้องในการตั้งค่า')
    if len(matches) > 1:
        raise RuntimeError('พบ USB 2.0 Camera หลายตัว · เลือกหมายเลขกล้องในการตั้งค่า')
    return matches[0]


class Camera:
    def __init__(self, index, width, height, backend='Auto'):
        self.index, self.width, self.height = index, width, height
        self.backend = backend
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def latest_sample(self):
        with self.lock:
            return self.sequence, None if self.frame is None else self.frame.copy()

    def _run(self):
        cap = None
        try:
            backends = {'Auto': cv2.CAP_ANY, 'DirectShow': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF}
            backend = backends[self.backend] if os.name == 'nt' else cv2.CAP_ANY
            if self.index is None:
                self.index = microscope_index()
                backend = cv2.CAP_DSHOW
            if self.stop_event.is_set():return
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                raise RuntimeError('เปิดกล้องไม่ได้ ลองเปลี่ยนหมายเลขกล้องหรือ backend และปิดแอปที่ใช้กล้องอยู่')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:pass  # some USB backends do not expose this property
            failed = 0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    failed += 1
                    if failed >= 15:
                        raise RuntimeError('ไม่ได้รับภาพจากกล้อง ตรวจสาย USB แล้วเชื่อมต่อใหม่')
                    time.sleep(.05)
                    continue
                failed = 0
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
