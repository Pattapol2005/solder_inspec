
import argparse
from pathlib import Path

# ========== CONFIG ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_inspec.pt"

# เลขกล้อง: 0 = กล้องตัวแรก, 1 = กล้องตัวที่สอง หรือใส่ path ภาพ/วิดีโอ
TEST_SOURCE = "1"

IMGSZ = 1024
CONF = 0.25
IOU = 0.45

OUTPUT_DIR = BASE_DIR / "runs/test"
RUN_NAME = "prediction"


def run_camera(model, camera_index, device):
    import cv2

    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_index}. Close other camera apps, "
                "check Windows camera permissions, or try --source 1."
            )
        print("Live camera: press Q or Esc, or close the window, to exit.")
        window = "YOLO - Live Camera"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read a frame from camera {camera_index}.")
            result = model.predict(
                source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU,
                device=device, save=False, verbose=False,
            )[0]
            cv2.imshow(window, result.plot())
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ========== LOAD MODEL ==========
def main():
    parser = argparse.ArgumentParser(description="Run best.pt on a webcam (default: 0), image, folder, or video.")
    parser.add_argument("--source", default=TEST_SOURCE, help="Camera index such as 0 or 1, or an input file/folder path")
    args = parser.parse_args()
    is_camera = args.source.isdecimal()
    if is_camera:
        source = int(args.source)
    else:
        source = Path(args.source).expanduser()
        if not source.is_absolute():
            source = BASE_DIR / source
        if not source.exists():
            parser.error(
                f"Input path does not exist: {source}\n"
                'Specify an existing image, image folder, or video: run.cmd --source "D:\\path\\to\\images"'
            )
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(MODEL_PATH)

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)

    print("Model:", MODEL_PATH)
    print("Source:", source)
    print("Device:", device)
    print("Classes:", model.names)

    if is_camera:
        try:
            run_camera(model, source, device)
        except KeyboardInterrupt:
            print("\nCamera stopped.")
        return

    # ========== PREDICT ==========
    results = model.predict(
        source=str(source),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        stream=True,
        verbose=True
    )

    # ========== RESULTS ==========
    for result in results:
        print(f"\nImage: {result.path}")

        boxes = result.obb if result.obb is not None else result.boxes
        if boxes is not None:
            print("Detections:", len(boxes))

            for box in boxes:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())

                print(
                    f"Class: {model.names[cls_id]} | "
                    f"Confidence: {confidence:.3f}"
                )

    print("\nTesting completed!")
    print("Results saved to:", f"{OUTPUT_DIR}/{RUN_NAME}")


if __name__ == "__main__":
    main()
