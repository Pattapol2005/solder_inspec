"""Startup diagnostics and background resource sampling (no Tk calls)."""
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import os
import sys
from functools import lru_cache


@lru_cache(maxsize=4)
def inference_options(device):
    gpu=str(device).lower() not in ('cpu','mps','none')
    if gpu:
        import torch
        import cv2
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.set_num_threads(min(8,max(1,(os.cpu_count() or 4)//2)))
        cv2.setNumThreads(4)
    # A loaded YOLO already imports its configuration. Do not import the whole
    # framework here just to inspect capabilities (e.g. before a model is ready).
    config=getattr(sys.modules.get('ultralytics.cfg'),'DEFAULT_CFG_DICT',{})
    if 'quantize' in config:return {'quantize':16 if gpu else 32}
    return {'half':gpu}  # compatible with earlier Ultralytics installations


def model_file(folder, role='inspection'):
    names = ('best_inspec.pt', 'best.pt') if role == 'inspection' else ('best_locator.pt', 'locator.pt')
    folder = Path(folder)
    return next((folder/name for name in names if (folder/name).is_file()), folder/names[0])


def load_weights(path):
    from ultralytics import YOLO
    from inspection import validate_names
    model = YOLO(str(path))
    if model.task not in ('detect', 'obb'):
        raise ValueError('ต้องใช้โมเดล YOLO Detection หรือ OBB')
    return model, validate_names(model.names)


def load_locator_weights(path):
    from ultralytics import YOLO
    from board import COMPONENTS
    model = YOLO(str(path))
    if model.task not in ('detect', 'obb'):
        raise ValueError('โมเดลตำแหน่งต้องเป็น YOLO Detection หรือ OBB')
    from locator_markers import validate_locator_names
    mapping=validate_locator_names(model.names)
    return model, mapping


def probe_system(folder, emit):
    folder = Path(folder)
    result = dict(device='cpu', device_name='CPU', model=None, signature=None)
    def check(name, fn):
        emit(name, 'checking', 'กำลังตรวจสอบ…')
        try:
            state, text = fn()
        except Exception as exc:
            state, text = 'warning', str(exc)
        emit(name, state, text)
    def cpu():
        import psutil
        name = platform.processor() or platform.machine()
        if platform.system() == 'Windows':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                name = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        result['cpu_name'] = name
        return 'ok', f'{name} · {psutil.cpu_count()} threads'
    def ram():
        import psutil
        m = psutil.virtual_memory()
        return 'ok', f'{m.total/2**30:.1f} GB · available {m.available/2**30:.1f} GB'
    def gpu():
        names = []
        if platform.system() == 'Windows':
            proc = subprocess.run(['powershell', '-NoProfile', '-Command',
                '(Get-CimInstance Win32_VideoController).Name'], capture_output=True,
                text=True, timeout=12, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            names = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
        result['gpu_names'] = names
        return ('ok', ' / '.join(names)) if names else ('warning', 'ไม่พบข้อมูล GPU · ยังใช้งาน CPU ได้')
    def cuda():
        import torch
        result['torch_version'] = torch.__version__
        if torch.version.cuda is None:
            return 'warning', f'PyTorch {torch.__version__} เป็นรุ่น CPU · เปิด install_cuda.bat เพื่อใช้ NVIDIA GPU'
        if not torch.cuda.is_available():
            return 'warning', f'PyTorch CUDA {torch.version.cuda} · เข้าถึง GPU ไม่ได้ ตรวจไดรเวอร์ NVIDIA · ใช้ CPU'
        # Verify execution, not merely driver discovery.
        torch.ones(1, device='cuda:0').add_(1)
        torch.cuda.synchronize(0)
        result.update(device='0', device_name=torch.cuda.get_device_name(0))
        return 'ok', f'CUDA {torch.version.cuda} · {result["device_name"]}'
    def libraries():
        from importlib.metadata import version, PackageNotFoundError
        import cv2
        import PIL
        import psutil
        description=f'Python {platform.python_version()} · OpenCV {cv2.__version__} · Pillow {PIL.__version__} · psutil {psutil.__version__}'
        try:ocr_version=version('rapidocr-onnxruntime')
        except PackageNotFoundError:return 'warning',description+' · ยังไม่มี OCR: เปิด setup.bat'
        try:modern_version=version('rapidocr')
        except PackageNotFoundError:return 'warning',description+f' · OCR v4 พร้อม · เปิด setup.bat เพื่อเพิ่ม v6'
        return 'ok',description+f' · OCR v4/v6 พร้อม ({ocr_version}/{modern_version})'
    def storage():
        with tempfile.TemporaryFile(dir=folder) as handle:
            handle.write(b'write check')
        return 'ok', f'เขียนไฟล์ได้ · พื้นที่ว่าง {shutil.disk_usage(folder).free/2**30:.1f} GB'
    def weights():
        path = model_file(folder)
        if not path.is_file():
            return 'warning', 'ยังไม่มี best_inspec.pt (หรือ best.pt) · เข้าโปรแกรมเพื่อใช้กล้องได้'
        stat = path.stat()
        model, mapping = load_weights(path)
        model.to('cuda:0' if result['device'] == '0' else 'cpu')
        result.update(model=model, class_map=mapping, model_file=str(path), signature=(stat.st_mtime_ns, stat.st_size))
        return 'ok', f'{path.name} · {model.task.upper()} · {stat.st_size/2**20:.1f} MB · 4 classes'
    def locator_weights():
        path = model_file(folder, 'locator')
        if not path.is_file():
            return 'warning', 'ยังไม่มี best_locator.pt · ยังใช้ภาพอ้างอิง/OCR ได้'
        stat = path.stat()
        model, mapping = load_locator_weights(path)
        model.to('cuda:0' if result['device'] == '0' else 'cpu')
        result.update(locator_model=model, locator_class_map=mapping, locator_file=str(path),
                      locator_signature=(stat.st_mtime_ns, stat.st_size))
        return 'ok', f'{path.name} · {model.task.upper()} · {stat.st_size/2**20:.1f} MB · {len(mapping)} classes'
    for name, fn in [('CPU', cpu), ('RAM', ram), ('GPU', gpu), ('CUDA / PyTorch', cuda),
                     ('Libraries', libraries), ('Storage', storage), ('best_inspec.pt', weights),
                     ('best_locator.pt', locator_weights)]:
        check(name, fn)
    return result


class ResourceMonitor:
    """System-wide usage. GPU metrics refer to NVIDIA physical GPU 0 via NVML."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def publish(self, sample):
        try:
            self.samples.get_nowait()
        except queue.Empty:
            pass
        self.samples.put_nowait(sample)

    def _run(self):
        nvml = None
        handle = None
        try:
            import psutil
            process = psutil.Process()
            psutil.cpu_percent(None)  # prime; discard first sample
            try:
                import pynvml as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                handle = None
            while not self.stop_event.wait(1):
                mem = psutil.virtual_memory()
                sample = dict(cpu=psutil.cpu_percent(None), ram=mem.percent,
                              ram_gb=mem.used/2**30, ram_total_gb=mem.total/2**30, app_mb=process.memory_info().rss/2**20,
                              gpu=None, vram=None, gpu_name='NVIDIA GPU 0: N/A')
                if handle is not None:
                    try:
                        name = nvml.nvmlDeviceGetName(handle)
                        sample['gpu_name'] = name.decode() if isinstance(name, bytes) else name
                        try:
                            sample['gpu'] = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        except Exception:
                            pass
                        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                        sample['vram_gb'] = memory.used / 2**30
                        sample['vram_total_gb'] = memory.total / 2**30
                        sample['vram'] = 100 * memory.used / memory.total
                    except Exception:
                        pass
                self.publish(sample)
        except Exception as exc:
            self.publish(dict(error=str(exc)))
        finally:
            if nvml is not None:
                try:
                    nvml.nvmlShutdown()
                except Exception:
                    pass
