
"""
ScanGuard — YOLO Model Comparison
================================
Compare YOLOv8 Nano and YOLOv8 Small.

Controls:
    1 -> YOLOv8 Nano
    2 -> YOLOv8 Small
    3 -> Show/hide other object classes
    Q -> Quit
"""

import cv2
import time
from ultralytics import YOLO


# ---------- CONFIG ----------

CLASSES_OF_INTEREST = {
    "bottle", "cup", "banana", "apple", "orange",
    "book", "cell phone", "remote", "mouse", "scissors"
}

CONF = 0.25
IMAGE_SIZE = 640

GREEN = (90, 220, 140)
WHITE = (255, 255, 255)

# Load both models
models = {
    1: YOLO("yolov8n.pt"),
    2: YOLO("yolov8s.pt")
}


# ---------- MAIN ----------

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not found.")
        return

    mode = 1
    show_all = False

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            model = models[mode]

            start = time.monotonic()

            results = model.predict(
                frame,
                conf=CONF,
                imgsz=IMAGE_SIZE,
                verbose=False
            )[0]

            elapsed = time.monotonic() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            detected = 0

            # ---------- DETECTIONS ----------

            for box in results.boxes:

                name = model.names[int(box.cls.item())]

                if not show_all and name not in CLASSES_OF_INTEREST:
                    continue

                detected += 1

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf.item())

                label = f"{name} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    GREEN,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                    2
                )

            # ---------- CENTER LINE ----------

            h, w = frame.shape[:2]

            zone_x = w // 2

            cv2.line(
                frame,
                (zone_x, 90),
                (zone_x, h),
                GREEN,
                2
            )

            # ---------- INFORMATION ----------

            cv2.rectangle(
                frame,
                (0, 0),
                (w, 90),
                (25, 25, 25),
                -1
            )

            model_name = (
                "YOLOv8 Nano"
                if mode == 1
                else "YOLOv8 Small"
            )

            cv2.putText(
                frame,
                f"MODEL: {model_name}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                GREEN,
                2
            )

            cv2.putText(
                frame,
                f"Detected: {detected} | FPS: {fps:.1f}",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1
            )

            cv2.putText(
                frame,
                "[1] Nano [2] Small [3] All classes [Q] Quit",
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                WHITE,
                1
            )

            cv2.imshow(
                "ScanGuard - Model Comparison",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                mode = 1

            elif key == ord("2"):
                mode = 2

            elif key == ord("3"):
                show_all = not show_all

            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
