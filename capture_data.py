
"""
ScanGuard — Dataset Image Capture
=================================
Collect webcam images for training a custom YOLO model.

Controls:
    1 -> Select can
    2 -> Select bottle
    3 -> Select carton
    S -> Save image
    Q -> Quit
"""

import cv2
from pathlib import Path
from datetime import datetime


# ---------- CONFIGURATION ----------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "raw"

CLASSES = {
    ord("1"): "can",
    ord("2"): "bottle",
    ord("3"): "carton"
}

for name in CLASSES.values():
    (DATASET_DIR / name).mkdir(parents=True, exist_ok=True)


# ---------- MAIN ----------

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not found.")
        return

    selected_class = "can"

    print("ScanGuard Dataset Capture")
    print("1: Can | 2: Bottle | 3: Carton")
    print("S: Save image | Q: Quit")

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # Keep an untouched image for training
            clean_frame = frame.copy()

            # Display selected class
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 75),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"Selected: {selected_class}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (90, 220, 140),
                2
            )

            cv2.putText(
                frame,
                "[1] Can [2] Bottle [3] Carton [S] Save [Q] Quit",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            cv2.imshow("ScanGuard - Dataset Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in CLASSES:
                selected_class = CLASSES[key]
                print(f"Selected: {selected_class}")

            elif key == ord("s"):

                filename = (
                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    ) + ".jpg"
                )

                path = DATASET_DIR / selected_class / filename

                if cv2.imwrite(str(path), clean_frame):
                    print(f"Saved: {path}")
                else:
                    print("ERROR: Image could not be saved.")

            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
