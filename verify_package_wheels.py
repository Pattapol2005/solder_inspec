"""Check published binary availability without downloading large wheels."""
import concurrent.futures
import json
import urllib.request
from pathlib import Path

checks = {
    ("numpy", "1.26.4"): ["cp312-cp312-win_amd64", "cp312-cp312-macosx_11_0_arm64", "cp312-cp312-macosx_10_9_x86_64"],
    ("opencv-python", "4.10.0.84"): ["abi3-win_amd64", "abi3-macosx_11_0_arm64", "abi3-macosx_12_0_x86_64"],
    ("torch", "2.5.1"): ["cp312-cp312-win_amd64", "cp312-none-macosx_11_0_arm64"],
    ("torchvision", "0.20.1"): ["cp312-cp312-win_amd64", "cp312-cp312-macosx_11_0_arm64"],
    ("torch", "2.2.2"): ["cp312-none-macosx_10_9_x86_64"],
    ("torchvision", "0.17.2"): ["cp312-cp312-macosx_10_13_x86_64"],
    ("ultralytics", "8.4.60"): ["py3-none-any"],
}


def check(item):
    (name, version), tags = item
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=45) as response:
        metadata = json.load(response)
    files = [f["filename"] for f in metadata["urls"]]
    found = {tag: [f for f in files if tag in f] for tag in tags}
    return {"package": name, "version": version, "matches": found,
            "passed": all(found.values()), "source": url}


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(check, checks.items()))
for result in results:
    print(json.dumps(result))
Path("camera_packages/wheel_availability.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
assert all(r["passed"] for r in results), "Inspect wheel tags in report"
