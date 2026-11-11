"""Build portable source bundles; never modify the original application/environment."""
from pathlib import Path
import html
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "camera_packages"
SOURCE = (ROOT / "test.py").read_text(encoding="utf-8")

GUIDE = r'''# คู่มือเริ่มจากศูนย์: โปรแกรมตรวจชิ้นงานด้วยกล้อง

## 1. โปรแกรมนี้ทำอะไร
เปิดกล้อง รับภาพทีละเฟรม แล้วใช้โมเดล best_inspec.pt วาดผลทำนายบนภาพสด
กรอบอาจเป็นกรอบหมุนตามโมเดล OBB; ชื่อคลาสและค่าความมั่นใจแสดงตามผลโมเดล
ค่าความมั่นใจไม่ใช่หลักประกันว่าผลถูกต้อง ต้องทดสอบกับชิ้นงานจริงก่อนนำไปใช้ตัดสินงาน
กด Q หรือ Esc ขณะเลือกหน้าต่างภาพเพื่อหยุด หรือกดปุ่มปิดหน้าต่าง
โหมดกล้องไม่บันทึกภาพหรือวิดีโออัตโนมัติ

## 2. ต้องเตรียมอะไร
- คอมพิวเตอร์ Windows 10/11 แบบ Intel/AMD 64-bit หรือ Mac Intel/Apple Silicon ที่ใช้ macOS 13 ขึ้นไปสำหรับชุดนี้
- กล้องในตัวหรือกล้อง USB ที่ระบบปฏิบัติการมองเห็น
- อินเทอร์เน็ตตอนติดตั้ง และพื้นที่ว่างเผื่อแพ็กเกจหลาย GB
- ติดตั้ง Python ซึ่งเป็นตัวอ่านและรันโค้ด; ตัวติดตั้งด้านล่างจะสร้าง .venv ซึ่งเป็นที่เก็บแพ็กเกจเฉพาะโปรแกรมนี้
- ไม่ต้องมี VS Code, Git, Anaconda หรือ CUDA เพื่อเริ่มใช้งาน ตัวเริ่มต้นใช้ CPU ได้
- ไม่ต้องคัดลอก .venv จากเครื่องอื่นและไม่ต้องใช้ environment-before.txt ของเครื่องเดิม

ภายในโฟลเดอร์ต้องมีไฟล์โค้ด ไฟล์ติดตั้ง ไฟล์เปิดโปรแกรม และ best_inspec.pt อยู่ด้วยกัน
best_locator.pt ไม่ได้ถูกเรียกใช้ในโค้ดชุดนี้ จึงไม่ได้ใส่มาในชุดแจก
ห้ามเปลี่ยนชื่อ best_inspec.pt โดยไม่ปรับ MODEL_PATH ในโค้ด

## 3. ติดตั้งบน Windows ทีละขั้น
1. แตกไฟล์ YOLO_Camera_Windows.zip: คลิกขวา > Extract All / แยกทั้งหมด
   ย้ายโฟลเดอร์ YOLO_Camera_Windows ไปไว้ที่ Desktop ห้ามรันจากหน้าดูไฟล์ ZIP
2. เปิด https://www.python.org/downloads/release/python-31210/
   เลื่อนลงหัวข้อ Files แล้วเลือก Windows installer (64-bit)
   ใช้ Python 3.12.10 สำหรับชุดนี้ เป็นเวอร์ชันที่มี installer ตรงกับชุดแพ็กเกจ ไม่ได้หมายความว่าเป็น Python รุ่นใหม่ที่สุด
   อย่าเลือก embeddable package, source archive หรือ ARM64
3. เปิดไฟล์ติดตั้ง Python ติ๊ก Add python.exe to PATH และเปิดตัวเลือกติดตั้ง launcher หากมี
   กด Install Now รอจนเสร็จ แล้วปิดหน้าต่างติดตั้ง
4. เปิดโฟลเดอร์ YOLO_Camera_Windows ดับเบิลคลิก INSTALL_WINDOWS.cmd
   สคริปต์ตรวจ Python 3.12 สร้าง .venv และดาวน์โหลด PyTorch, OpenCV, NumPy และ Ultralytics
   ครั้งแรกอาจใช้เวลาหลายนาที อย่าปิดหน้าต่างระหว่างดาวน์โหลด
   ต้องเห็นข้อความ INSTALL COMPLETE ถ้าไม่เห็นให้ดู error ก่อน ไม่ใช่ติดตั้งสำเร็จ
5. ดับเบิลคลิก RUN_WINDOWS.cmd จะเปิดกล้อง 1 ตามค่าเดิมของงาน
   หากไม่ขึ้นหรือเป็นกล้องคนละตัว ให้ปิดโปรแกรมแล้วดับเบิลคลิก CAMERA_0_WINDOWS.cmd
6. ต้องเห็นหน้าต่าง YOLO - Live Camera ภาพขยับได้และมีผลทำนายเมื่อโมเดลตรวจพบ
   กด Q หรือ Esc เพื่อจบ ครั้งถัดไปเปิด RUN_WINDOWS.cmd ได้เลย ไม่ต้องติดตั้งซ้ำ

หากต้องการพิมพ์คำสั่งเอง: เปิดโฟลเดอร์ใน File Explorer คลิกแถบ path ด้านบน พิมพ์ cmd แล้ว Enter
ตัวอย่างในหน้าต่าง Command Prompt ที่เปิดขึ้น:

```bat
RUN_WINDOWS.cmd --source 0
RUN_WINDOWS.cmd --source 1 --imgsz 640
RUN_WINDOWS.cmd --check
```

สำหรับ PowerShell ให้เติม .\ เช่น .\RUN_WINDOWS.cmd --source 0
Windows ชุดนี้ติดตั้ง PyTorch แบบ CPU เพื่อให้เริ่มต้นได้โดยไม่ต้องลง CUDA
แม้มีการ์ด NVIDIA ก็จะยังใช้ CPU กับชุดติดตั้งเริ่มต้นนี้ การติดตั้ง CUDA เพิ่มเป็นอีกขั้นตอนหนึ่ง
ถ้ารันใน environment อื่นที่รองรับ CUDA โค้ด Windows เลือก CUDA ได้เมื่อใช้ --device auto

## 4. ติดตั้งบน Mac ทีละขั้น
1. คลิกเมนู Apple > About This Mac / เกี่ยวกับ Mac เครื่องนี้
   ถ้า Chip แสดง Apple M1/M2/M3/M4 หรือรุ่นอื่นในตระกูล Apple ให้ใช้ Terminal แบบปกติ
   หากเป็น Intel ตัวติดตั้งจะเลือก PyTorch รุ่นสำหรับ Intel ให้
   บน Apple Silicon ไม่ควรเปิด Terminal แบบ Open using Rosetta สำหรับชุดนี้
2. ดับเบิลคลิก YOLO_Camera_Mac.zip เพื่อแตกไฟล์ แล้วลากโฟลเดอร์ YOLO_Camera_Mac ไปไว้ Desktop
3. เปิด https://www.python.org/downloads/release/python-31210/
   เลื่อนลง Files เลือก macOS 64-bit universal2 installer (.pkg)
   ดับเบิลคลิก .pkg กด Continue ตามขั้นตอนจนติดตั้งสำเร็จ
   ใช้ Python 3.12.10 ตามชุดนี้ ไม่ใช่ Python ที่อาจมากับระบบ
4. เปิด Finder > Applications > Python 3.12 แล้วดับเบิลคลิก Install Certificates.command หากมี
   รอจนเสร็จ ใช้แก้การตรวจใบรับรอง HTTPS ของ Python โดยไม่ปิด SSL verification
5. เปิด Terminal: กด Command + Space พิมพ์ Terminal แล้ว Enter
   Terminal คือหน้าต่างรับคำสั่ง พิมพ์หรือวางทีละบรรทัดแล้วกด Enter
   เมื่อวางโฟลเดอร์บน Desktop ตามข้อ 2 ให้รัน:

```bash
cd ~/Desktop/YOLO_Camera_Mac
bash INSTALL_MAC.sh
```

   หากวางโฟลเดอร์ไว้ที่อื่น: พิมพ์ cd ตามด้วยเว้นวรรค ลากโฟลเดอร์จาก Finder มาวางใน Terminal แล้ว Enter
   สคริปต์จะสร้าง .venv ตรวจชิป และดาวน์โหลดแพ็กเกจ ต้องเห็น INSTALL COMPLETE
   ไม่ต้องใช้ sudo และไม่ต้องติดตั้ง Homebrew
6. เปิดกล้องตามค่าเดิม:

```bash
bash RUN_MAC.sh
```

   หากเป็นกล้องในตัวหรือไม่พบกล้อง 1 ให้ใช้:

```bash
bash RUN_MAC.sh --source 0
```

7. หากมีหน้าต่างถามสิทธิ์กล้องให้กด Allow / อนุญาต
   ต้องเห็นภาพสด กด Q หรือ Esc เพื่อหยุด
8. ครั้งถัดไปเปิด Terminal แล้วรันเพียง:

```bash
cd ~/Desktop/YOLO_Camera_Mac
bash RUN_MAC.sh --source 0
```

ไม่ใช้ไฟล์ .cmd บน Mac และไม่ต้อง chmod เพราะเราเรียกไฟล์ผ่าน bash โดยตรง
Mac ใช้ CPU เป็นค่าเริ่มต้นทั้ง Intel และ Apple Silicon เพื่อลดปัญหา operator ของโมเดล OBB
Apple Silicon อาจทดลอง GPU ด้วย bash RUN_MAC.sh --source 0 --device mps
โหมด MPS เป็นตัวเลือกที่ยังไม่ได้ทดสอบกับโมเดลนี้บนเครื่อง Mac จริง หาก error ให้กลับมาใช้ --device cpu
ตัวเปิด Mac ตั้ง PYTORCH_ENABLE_MPS_FALLBACK=1 ให้ แต่ไม่ได้รับประกันว่าทุก operator จะทำงานได้

## 5. เลือกกล้อง ปรับความเร็ว และทดสอบไฟล์
- --source 0 คือกล้องหมายเลข 0; --source 1 คือหมายเลข 1 ลำดับจริงขึ้นกับเครื่องและอุปกรณ์ที่เสียบ
- ชุดนี้คงค่าเดิมคือกล้อง 1, imgsz 1024, conf 0.25 และ iou 0.45
- ถ้าภาพช้า ทดลอง --imgsz 640 ซึ่งอาจเปลี่ยนความสามารถในการตรวจรายละเอียดเล็ก ๆ
- --conf 0.4 จะกรองผลที่ความมั่นใจต่ำกว่า 0.4 ออก ไม่ได้ทำให้โมเดลแม่นขึ้นโดยตัวมันเอง
- ใช้ --check เพื่อทดสอบโหลดโมเดลและประมวลผลภาพจำลอง โดยไม่เปิดกล้อง ไม่ใช่การตรวจความแม่นยำ

ตัวอย่าง Windows:
```bat
RUN_WINDOWS.cmd --source 0 --imgsz 640 --conf 0.25
RUN_WINDOWS.cmd --source "C:\Users\YourName\Pictures\sample.jpg"
```

ตัวอย่าง Mac:
```bash
bash RUN_MAC.sh --source 0 --imgsz 640 --conf 0.25
bash RUN_MAC.sh --source "$HOME/Pictures/sample.jpg"
bash RUN_MAC.sh --check
```

โหมดไฟล์จะบันทึกผลใน runs/test/prediction ภายใต้โฟลเดอร์โปรแกรม และอาจเขียนทับผลชื่อเดิม
โหมดกล้องไม่บันทึกไฟล์ หากไม่พบวัตถุ ภาพสดอาจไม่มีกรอบ ซึ่งไม่ได้แปลว่ากล้องเสีย
ห้ามย้าย/เปลี่ยนชื่อโฟลเดอร์หลังติดตั้ง .venv เพราะ path ภายในอาจอ้างอิงที่เดิม
ถ้าต้องย้าย ให้แตก ZIP ที่ตำแหน่งใหม่แล้วรันตัวติดตั้งใหม่

## 6. แก้ปัญหาที่พบบ่อย
### Python 3.12 not found / python3.12: command not found
ติดตั้ง Python ตามข้อ 3 หรือ 4 แล้วปิดและเปิด Terminal/Command Prompt ใหม่
Windows ตรวจด้วย py -3.12 --version; Mac ตรวจด้วย python3.12 --version
หาก Windows ไม่มี py ให้ลอง python --version ต้องเป็น 3.12.x; ตัวติดตั้งรองรับทั้งสองแบบ
หาก Python เปิด Microsoft Store ให้ติดตั้งจาก python.org ตามลิงก์ก่อน

### Cannot open camera / Cannot read a frame
ปิด Zoom, Teams, Camera, Photo Booth หรือโปรแกรมอื่นที่ใช้กล้อง แล้วลอง --source 0 และ --source 1
กล้อง USB ให้ตรวจสายและพอร์ต ลองเปิดด้วยแอปกล้องของระบบก่อน แล้วปิดแอปนั้นก่อนรันโปรแกรม
Windows: Settings > Privacy & security > Camera เปิด Camera access และสิทธิ์สำหรับ desktop apps ตามที่ระบบแสดง
Windows 10 อาจอยู่ Settings > Privacy > Camera
Mac: System Settings > Privacy & Security > Camera เปิดสิทธิ์ให้ Terminal หรือแอปที่ใช้เรียก Pythonตามที่ปรากฏ
หลังเปลี่ยนสิทธิ์ให้ปิดและเปิดโปรแกรม/Terminal ใหม่

### No module named ultralytics / numpy หรือ error _ARRAY_API
ให้รันผ่าน RUN_WINDOWS.cmd หรือ bash RUN_MAC.sh ซึ่งเรียก Python ใน .venv ของชุดนี้
อย่าเรียก Python หลักที่ติดตั้งแพ็กเกจชุดอื่น หากติดตั้งไม่ครบให้รัน INSTALL อีกครั้ง
แพ็กเกจชุดนี้กำหนด NumPy 1.26.4 คู่กับ OpenCV 4.10 และ PyTorch ที่ระบุไว้

### FileNotFoundError: best_inspec.pt
ตรวจว่า best_inspec.pt อยู่ข้างไฟล์ camera_windows.py หรือ camera_mac.py
หาก ZIP แตกมาเป็นหลายชั้น ให้เปิดชั้นที่มีไฟล์โค้ดจริง ห้ามนำเฉพาะโค้ดไปไว้ที่อื่น

### หน้าต่าง OpenCV เปิดไม่ได้ / function is not implemented
ใช้ environment ใหม่ของชุดนี้ ไม่ติดตั้ง opencv-python-headless เพราะชุดนั้นไม่มีหน้าต่าง GUI
หากรัน --check ผ่าน ก็ยังไม่ได้ยืนยันว่าหน้าต่าง GUI และสิทธิ์กล้องทำงาน ต้องเปิดกล้องจริงอีกครั้ง

### Windows: DLL load failed / WinError 126
ติดตั้ง Microsoft Visual C++ Redistributable v14 รุ่น X64 จากเว็บ Microsoft ด้านล่าง แล้วเปิดโปรแกรมใหม่
หากมีข้อความต้อง restart ให้ restart ตามที่ตัวติดตั้งแจ้ง ไม่ต้องติดตั้ง Visual Studio ทั้งชุด

### Mac: No matching distribution found for torch
ตรวจ Python ให้เป็น 3.12 และตรวจ uname -m ใน Terminal
Apple Silicon ควรเป็น arm64; Intel เป็น x86_64 ตัวติดตั้งเลือก requirements ตามค่านี้
หากใช้ Apple Silicon แต่เห็น x86_64 ให้ปิดการเปิด Terminal ผ่าน Rosetta แล้วสร้าง .venv ใหม่จากชุด ZIP ใหม่
Mac Intel ต้องใช้ torch 2.2.2 ตามไฟล์ requirements-mac-intel.txt ไม่อัปเกรดเป็นรุ่นล่าสุดเอง

### ดาวน์โหลดไม่ได้ / CERTIFICATE_VERIFY_FAILED
ตรวจอินเทอร์เน็ต วันเวลาเครื่อง และบน Mac รัน Install Certificates.command ตามข้อ 4
หากเป็นเครือข่ายองค์กรให้ตรวจ proxy/ใบรับรองกับผู้ดูแล ไม่แก้ด้วยการปิดการตรวจ HTTPS

### ภาพช้า / โปรแกรมดูเหมือนค้าง
CPU อาจช้าโดยเฉพาะ imgsz 1024 ช่วงแรกมีเวลาโหลดโมเดล รอสักครู่แล้วทดลอง --imgsz 640
ถ้าต้องหยุดใน Terminal ใช้ Ctrl+C โปรแกรมจะคืนกล้อง

## 7. สิ่งที่รวมมาและขอบเขตการตรวจ
- แพ็กเกจแยก Windows และ Mac มี best_inspec.pt รวมอยู่แล้ว พร้อมโค้ด ติดตั้ง และตัวเปิด
- ต้นฉบับ test.py และ .venv เดิมไม่ได้ถูกแก้ไขเพื่อสร้างชุดแจกนี้
- ไม่ต้องมีโปรแกรมแก้โค้ดเพื่อเริ่มใช้ หากต้องการแก้ในภายหลังใช้ VS Code ได้เป็นตัวเลือก
- อ่าน VALIDATION.txt เพื่อดูการตรวจที่ทำจริง อย่าสับสนการทดสอบภาพจำลองกับความแม่นยำหรือการทดสอบ Mac
- เวอร์ชันที่ pin เป็นชุดความเข้ากันได้เฉพาะงาน ไม่ได้หมายความว่าเป็นแพ็กเกจรุ่นล่าสุด

## 8. แหล่งดาวน์โหลดและอ้างอิงทางการ
- Python installer: https://www.python.org/downloads/release/python-31210/
- Ultralytics: https://docs.ultralytics.com/quickstart/
- PyTorch รุ่นก่อนหน้า: https://pytorch.org/get-started/previous-versions/
- PyTorch สิ้นสุด binary macOS Intel ที่สาย 2.2.x: https://pytorch.org/blog/pytorch2-2/
- สิทธิ์กล้อง Mac: https://support.apple.com/guide/mac-help/control-access-to-your-camera-mchlf6d108da/mac
- สิทธิ์กล้อง Windows: https://support.microsoft.com/en-us/windows/privacy/windows-camera-microphone-and-privacy
- Visual C++ Runtime X64: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n")
    if path.suffix == ".cmd":
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def guide_html(markdown):
    parts = []
    paragraph = []
    code = None
    def flush():
        if paragraph:
            value = html.escape("\n".join(paragraph))
            value = re.sub(r"https://[^\s<>]+", lambda m: '<a href="' + m[0] + '">' + m[0] + '</a>', value)
            parts.append('<p>' + value + '</p>')
            paragraph.clear()
    for line in markdown.splitlines():
        if line.startswith('```'):
            flush()
            if code is None:
                code = []
            else:
                parts.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
                code = None
        elif code is not None:
            code.append(line)
        elif line.startswith('#'):
            flush()
            level = len(line) - len(line.lstrip('#'))
            parts.append(f'<h{level}>' + html.escape(line[level:].strip()) + f'</h{level}>')
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
    flush()
    return '<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>คู่มือติดตั้งกล้อง Windows และ Mac</title><style>body{max-width:960px;margin:32px auto;padding:0 24px;background:#f6f8fc;color:#15243a;font:18px/1.85 Tahoma,Arial,sans-serif}h1{font-size:30px}h2{color:#1261ab;border-top:1px solid #dce3ee;padding-top:26px;margin-top:40px}h3{font-size:20px}p{white-space:pre-line;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#13263f;color:#fff;border-radius:10px;padding:22px;font:16px/1.7 monospace}a{color:#1261ab}@media print{body{background:white;margin:0;font-size:12pt}pre{background:#eee;color:black}h2{break-after:avoid}}</style><body>' + ''.join(parts) + '</body></html>'


COMMON = "numpy==1.26.4\nopencv-python==4.10.0.84\nultralytics==8.4.60\n"

WIN_INSTALL = r'''@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_CMD=py -3.12"
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if not errorlevel 1 goto python_ok
set "PYTHON_CMD=python"
python -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if not errorlevel 1 goto python_ok
echo Python 3.12 not found. Install Python 3.12 from python.org first.
goto failed
:python_ok
%PYTHON_CMD% -c "import platform; assert platform.machine().upper() in ('AMD64','X86_64'), 'This bundle requires Intel/AMD 64-bit Windows'"
if errorlevel 1 goto failed
if exist ".venv\Scripts\python.exe" goto env_ready
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto failed
:env_ready
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12), 'Use a fresh folder with Python 3.12'"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-windows.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" camera_windows.py --check
if errorlevel 1 goto failed
echo.
echo INSTALL COMPLETE. Open RUN_WINDOWS.cmd or CAMERA_0_WINDOWS.cmd.
pause
exit /b 0
:failed
echo.
echo INSTALL FAILED. Read the error above. See START_HERE.html for help.
pause
exit /b 1
'''

MAC_INSTALL = r'''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo "INSTALL FAILED. Read the error above; see START_HERE.html." >&2' ERR
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 not found. Install the macOS universal2 .pkg from python.org."
  exit 1
fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3,12)'
arch="$(python3.12 -c 'import platform; print(platform.machine())')"
case "$arch" in
  arm64) req="requirements-mac-apple.txt" ;;
  x86_64) req="requirements-mac-intel.txt" ;;
  *) echo "Unsupported architecture: $arch"; exit 1 ;;
esac
echo "Python architecture: $arch / $req"
if [ ! -x .venv/bin/python ]; then python3.12 -m venv .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use a fresh folder with Python 3.12"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "$req"
.venv/bin/python -m pip check
.venv/bin/python camera_mac.py --check
echo "INSTALL COMPLETE. Run: bash RUN_MAC.sh"
'''

for system in ("Windows", "Mac"):
    folder = OUT / f"YOLO_Camera_{system}"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"camera_{system.lower()}.py"
    code = SOURCE.replace(
        'parser = argparse.ArgumentParser(description="Run best.pt on a webcam (default: 0), image, folder, or video.")',
        'parser = argparse.ArgumentParser(description="Live camera inspection using best_inspec.pt (default camera: 1).")'
    )
    default_device = "auto" if system == "Windows" else "cpu"
    code = code.replace('    args = parser.parse_args()', f'''    parser.add_argument("--device", default="{default_device}", help="auto, cpu, mps, or CUDA index such as 0")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--check", action="store_true", help="Synthetic inference check; does not open a camera")
    args = parser.parse_args()
    if args.imgsz <= 0 or not 0 <= args.conf <= 1:
        parser.error("--imgsz must be positive; --conf must be between 0 and 1")''')
    code = code.replace('    if is_camera:\n        source = int(args.source)', '    if args.check:\n        source = "synthetic check"\n    elif is_camera:\n        source = int(args.source)')
    code = code.replace('    device = 0 if torch.cuda.is_available() else "cpu"', '''    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    elif args.device.isdecimal():
        device = int(args.device)
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")''')
    code = code.replace('def run_camera(model, camera_index, device):', 'def run_camera(model, camera_index, device, imgsz, conf):')
    code = code.replace('source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU,', 'source=frame, imgsz=imgsz, conf=conf, iou=IOU,')
    code = code.replace('run_camera(model, source, device)', 'run_camera(model, source, device, args.imgsz, args.conf)')
    code = code.replace('        imgsz=IMGSZ,', '        imgsz=args.imgsz,').replace('        conf=CONF,', '        conf=args.conf,')
    code = code.replace('    if is_camera:\n        try:', '''    if args.check:
        import numpy as np
        import cv2
        sample = np.zeros((640, 640, 3), dtype=np.uint8)
        result = model.predict(sample, imgsz=args.imgsz, conf=args.conf,
                               iou=IOU, device=device, save=False, verbose=False)[0]
        annotated = result.plot()
        assert annotated.shape == sample.shape
        print("CHECK PASSED: model inference + annotation; task:", model.task)
        print("NumPy:", np.__version__, "Torch:", torch.__version__, "OpenCV:", cv2.__version__)
        print("Camera and GUI are not tested by --check.")
        return

    if is_camera:
        try:''')
    code = code.replace('check Windows camera permissions, or try --source 1.', 'check camera privacy permissions, or try --source 0 / --source 1.')
    code = code.replace('Specify an existing image, image folder, or video: run.cmd --source', 'Specify an existing image, image folder, or video with --source')
    write(folder / name, code)
    shutil.copy2(ROOT / ".gitignore", folder / ".gitignore")
    shutil.copy2(ROOT / "best_inspec.pt", folder / "best_inspec.pt")
    write(folder / "คู่มือเริ่มต้น.md", GUIDE)
    # A self-contained browser-readable guide: no renderer, network, or fonts required.
    page = guide_html(GUIDE)
    write(folder / "START_HERE.html", page)
    if system == "Windows":
        write(folder / "requirements-windows.txt", COMMON + "torch==2.5.1\ntorchvision==0.20.1\n")
        write(folder / "INSTALL_WINDOWS.cmd", WIN_INSTALL)
        write(folder / "RUN_WINDOWS.cmd", r'''@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_WINDOWS.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" camera_windows.py %*
set "APP_EXIT=%errorlevel%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%
''')
        write(folder / "CAMERA_0_WINDOWS.cmd", '@echo off\r\ncall "%~dp0RUN_WINDOWS.cmd" --source 0 %*\r\n')
    else:
        write(folder / "requirements-mac-apple.txt", COMMON + "torch==2.5.1\ntorchvision==0.20.1\n")
        write(folder / "requirements-mac-intel.txt", COMMON + "torch==2.2.2\ntorchvision==0.17.2\n")
        write(folder / "INSTALL_MAC.sh", MAC_INSTALL)
        write(folder / "RUN_MAC.sh", r'''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run: bash INSTALL_MAC.sh"
  exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python camera_mac.py "$@"
''')

print("Created:", OUT)
