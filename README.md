# ทดสอบ best.pt

## กล้องเว็บแคม (ค่าเริ่มต้น)

```powershell
.\run.cmd
```

แสดงกรอบตรวจจับและความมั่นใจแบบสด กด `Q`, `Esc` หรือปิดหน้าต่างเพื่อหยุด
โหมดกล้องไม่บันทึกภาพหรือวิดีโออัตโนมัติ หากต้องการกล้องตัวที่สองใช้:

```powershell
.\run.cmd --source 1
```

หากเปิดกล้องไม่ได้ ให้ปิดโปรแกรมอื่นที่ใช้กล้อง และตรวจสิทธิ์กล้องของ Windows

## ภาพหรือวิดีโอจากไฟล์

เปิด PowerShell ใน `D:\Download\test` แล้วระบุไฟล์ภาพ โฟลเดอร์ภาพ หรือวิดีโอที่มีอยู่จริง:

```powershell
.\run.cmd --source "D:\path\to\images"
```

เปลี่ยน path ตัวอย่างเป็น path ของคุณ ผลลัพธ์บันทึกใน `runs\test\prediction`
path แบบ relative จะอ้างอิงจากโฟลเดอร์ของ `test.py`

หากรันจาก VS Code ให้เลือก interpreter เป็น `D:\Download\test\.venv\Scripts\python.exe`
แล้วระบุ `--source` หรือแก้ `TEST_SOURCE` ใน `test.py` ให้ตรงกับภาพจริง
Python หลักตัวเดิมยังใช้ NumPy 2.2.6 และอาจพบ error เดิมหากเรียกตรง

## การแก้ไขและผลตรวจ

- `.venv` สร้างจาก Python 3.10 แบบ `--system-site-packages` ใช้ PyTorch/CUDA เดิม และติดตั้ง NumPy 1.26.4 เฉพาะใน `.venv`
- สำรองโค้ดเดิมใน `test.py.bak` และรายการแพ็กเกจเดิมใน `environment-before.txt`
- ตรวจ input path ก่อน import YOLO พร้อมรองรับ `--source` และกำหนด model/output path จากตำแหน่งสคริปต์
- ทดสอบ NumPy to Torch และ inference ของ `best.pt` ด้วยภาพสีดำจำลอง 640x640 บน CUDA ผ่านแล้ว ไม่ได้ใช้วัดความแม่นยำ
- ยังไม่ได้ทดสอบภาพจริง เพราะยังไม่ได้ระบุแหล่งภาพ
- pip รายงาน dependency conflicts ของแพ็กเกจอื่นที่สืบทอดมา เช่น TensorFlow, MediaPipe และ inference-sdk; `.venv` นี้ตรวจเฉพาะการทำงาน YOLO ข้างต้น

NumPy อธิบายปัญหา binary compatibility และวิธีใช้ `numpy<2` ที่:
https://numpy.org/doc/2.0/user/troubleshooting-importerror.html
"# solder_inspec" 
