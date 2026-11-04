
"""
ScanGuard — Detection Quality Test
==================================
Test YOLO independently from ByteTrack.

Controls:
    1 -> Original settings
    2 -> Improved sensitivity
    3 -> Show all YOLO classes
    Q -> Quit
"""

import cv2
from ultralytics import YOLO


# ---------- CONFIGURATION ----------

MODEL = "yolov8n.pt"

CLASSES_OF_INTEREST = {
    "bottle", "cup", "banana", "apple", "orange",
    "book", "cell phone", "remote", "mouse", "scissors"
}

GREEN = (90, 220, 140)
WHITE = (255, 255, 255)

model = YOLO(MODEL)


# ---------- MAIN ----------

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not found")
        return

    mode = 1
    show_all = False

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Original detection settings
        if mode == 1:
            conf = 0.4
            image_size = 640

        # More sensitive detection settings
        else:
            conf = 0.20
            image_size = 960

        # YOLO detection only — no ByteTrack
        results = model.predict(
            frame,
            conf=conf,
            imgsz=image_size,
            verbose=False
        )[0]

        detected = 0

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
                0.6,
                GREEN,
                2
            )

        # Information panel
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 100),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            f"MODE: {mode} | CONF: {conf} | SIZE: {image_size}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            2
        )

        cv2.putText(
            frame,
            f"DETECTED: {detected} | ALL CLASSES: {show_all}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2
        )

        cv2.putText(
            frame,
            "[1] Original  [2] Sensitive  [3] All classes  [Q] Quit",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1
        )

        cv2.imshow("ScanGuard - Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            mode = 1

        elif key == ord("2"):
            mode = 2

        elif key == ord("3"):
            show_all = not show_all

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
