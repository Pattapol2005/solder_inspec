from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import ast
from html.parser import HTMLParser

root = Path(__file__).resolve().parent
out = root / "camera_packages"
validation = """Validation date: 2026-11-11 (Asia/Bangkok)

PASS: Python syntax compilation and command-line help.
PASS: Invalid confidence values rejected before model import.
PASS: Both source variants load the included best_inspec.pt and run synthetic
      inference + result.plot() in the existing Windows Python 3.10 environment
      with NumPy 1.26.4, torch 1.13.1+cu116, OpenCV 4.10.0, ultralytics 8.4.60.
      Windows variant used CUDA; Mac variant's CPU path was exercised on Windows.
PASS: Mac shell scripts pass bash -n using Git Bash on Windows.
PASS: PyPI lists Python 3.12-compatible wheels for all explicitly pinned primary
      packages on Windows x64, Mac arm64 and Mac Intel; see wheel_availability.json.
PASS: Bundled model SHA256 matches source; ZIP CRC and model hash verified.

NOT VERIFIED: installation in fresh Python 3.12 environments on either OS.
NOT VERIFIED: execution, camera permissions, GUI or MPS on a physical Mac.
NOT VERIFIED: packaged GUI loop on Windows; synthetic checks do not open a camera.
NOT VERIFIED: model accuracy on a labelled test set.
Dependency resolver dry-runs were stopped during large-wheel downloads; binary
availability checks are not a complete dependency installation test.

INSTALL scripts perform pip check and real model --check on the target machine.
INSTALL COMPLETE is printed only after both checks pass.
"""

model_hash = hashlib.sha256((root / "best_inspec.pt").read_bytes()).hexdigest()
for os_name in ("Windows", "Mac"):
    folder = out / f"YOLO_Camera_{os_name}"
    code = folder / f"camera_{os_name.lower()}.py"
    ast.parse(code.read_text(encoding="utf-8"))
    html = (folder / "START_HERE.html").read_text(encoding="utf-8")
    HTMLParser().feed(html)
    assert "คู่มือเริ่มจากศูนย์" in html and "<pre><code>" in html
    assert "https://www.python.org/downloads/release/python-31210/" in html
    assert hashlib.sha256((folder / "best_inspec.pt").read_bytes()).hexdigest() == model_hash
    (folder / "VALIDATION.txt").write_text(validation, encoding="utf-8")
    shutil.copy2(out / "wheel_availability.json", folder / "wheel_availability.json")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.name != "SHA256.json")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (folder / "SHA256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = out / f"YOLO_Camera_{os_name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                z.write(path, f"{folder.name}/{path.name}")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        packed_model = z.read(f"{folder.name}/best_inspec.pt")
        assert hashlib.sha256(packed_model).hexdigest() == model_hash
        assert not any(".venv" in n or "__pycache__" in n for n in z.namelist())
    print(archive, round(archive.stat().st_size / 1024**2, 1), "MB; verified")
print("Model SHA256:", model_hash)
