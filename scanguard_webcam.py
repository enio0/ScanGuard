
"""
ScanGuard — AI-Powered Smart Checkout Monitoring
================================================
Detects and tracks objects crossing a virtual line using YOLOv8
and ByteTrack, comparing them against simulated checkout scans.

Controls:
    S     -> Scan an item
    P     -> Finish payment and check for discrepancies
    R     -> Reset transaction
    Q/Esc -> Quit
"""

import cv2
import time
from ultralytics import YOLO


# ---------- CONFIGURATION ----------

CLASSES_OF_INTEREST = {
    "bottle", "cup", "banana", "apple", "orange",
    "book", "cell phone", "remote", "mouse", "scissors"
}

ZONE_X_RATIO = 0.55
ZONE_MARGIN = 25
CONF = 0.4

GREEN = (90, 220, 140)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
RED = (50, 50, 230)

model = YOLO("yolov8n.pt")


# ---------- GLOBAL VARIABLES ----------

camera_count = 0
scanned_count = 0
verdict = None

# track_id -> {"armed": bool, "counted": bool}
tracks = {}


# ---------- OBJECT TRACKING ----------

def update_tracks(results, zone_x):
    global camera_count

    if results.boxes.id is None:
        return

    left_limit = zone_x - ZONE_MARGIN
    right_limit = zone_x + ZONE_MARGIN

    for box in results.boxes:

        name = model.names[int(box.cls)]

        if name not in CLASSES_OF_INTEREST:
            continue

        track_id = int(box.id.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2

        # First time detecting this object
        if track_id not in tracks:
            tracks[track_id] = {
                "armed": False,
                "counted": False
            }

        current = tracks[track_id]

        # Object must first reach the left zone
        if center_x <= left_limit:
            current["armed"] = True

        # Count when it reaches the right zone
        if (
            current["armed"]
            and not current["counted"]
            and center_x >= right_limit
        ):
            camera_count += 1
            current["counted"] = True

            print(
                f"Product counted! "
                f"ID: {track_id} | "
                f"Class: {name} | "
                f"Total: {camera_count}"
            )


# ---------- MAIN APPLICATION ----------

def main():
    global camera_count, scanned_count, verdict

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("No webcam found (index 0).")

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                print("Unable to read webcam frame.")
                break

            h, w = frame.shape[:2]
            zone_x = int(w * ZONE_X_RATIO)

            left_limit = zone_x - ZONE_MARGIN
            right_limit = zone_x + ZONE_MARGIN

            # ---------- YOLO + BYTETRACK ----------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONF,
                verbose=False
            )[0]

            # ---------- UPDATE OBJECT COUNT ----------

            update_tracks(results, zone_x)

            # ---------- DRAW DETECTIONS ----------

            for box in results.boxes:

                name = model.names[int(box.cls)]

                if name not in CLASSES_OF_INTEREST:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                track_id = (
                    int(box.id.item())
                    if box.id is not None
                    else -1
                )

                label = f"{name} ID:{track_id}"

                # Use orange for an object already counted
                if (
                    track_id in tracks
                    and tracks[track_id]["counted"]
                ):
                    color = ORANGE
                else:
                    color = GREEN

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # ---------- DETECTION ZONE ----------

            # Left boundary
            cv2.line(
                frame,
                (left_limit, 0),
                (left_limit, h - 70),
                ORANGE,
                1
            )

            # Central detection line
            cv2.line(
                frame,
                (zone_x, 0),
                (zone_x, h - 70),
                GREEN,
                2
            )

            # Right boundary
            cv2.line(
                frame,
                (right_limit, 0),
                (right_limit, h - 70),
                ORANGE,
                1
            )

            cv2.putText(
                frame,
                "DETECTION ZONE (left -> right)",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                GREEN,
                2
            )

            # ---------- BOTTOM DASHBOARD ----------

            cv2.rectangle(
                frame,
                (0, h - 80),
                (w, h),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"CAMERA: {camera_count}",
                (12, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2
            )

            cv2.putText(
                frame,
                f"SCANNED: {scanned_count}",
                (12, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                frame,
                "[s] scan  [p] pay  [r] reset  [q] quit",
                (max(w - 340, 0), h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1
            )

            # ---------- PAYMENT VERDICT ----------

            if verdict and time.time() - verdict[2] < 5:

                color = (
                    GREEN
                    if verdict[0] == "ok"
                    else RED
                )

                cv2.rectangle(
                    frame,
                    (0, 35),
                    (w, 80),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    verdict[1],
                    (12, 63),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    WHITE,
                    2
                )

            # ---------- SHOW WEBCAM ----------

            cv2.imshow("ScanGuard - ByteTrack v1.1", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key in (ord("q"), 27):
                break

            # Simulate product scan
            elif key == ord("s"):
                scanned_count += 1

                print(
                    f"Product scanned! "
                    f"Total scanned: {scanned_count}"
                )

            # Reset transaction
            elif key == ord("r"):

                camera_count = 0
                scanned_count = 0
                tracks.clear()
                verdict = None

                # Reset ByteTrack internal state
                if model.predictor is not None:

                    trackers = getattr(
                        model.predictor,
                        "trackers",
                        []
                    )

                    for tracker in trackers:
                        tracker.reset()

                print("Transaction reset.")

            # Finish payment
            elif key == ord("p"):

                if scanned_count >= camera_count:

                    verdict = (
                        "ok",
                        f"CLEARED: {camera_count} detected / "
                        f"{scanned_count} scanned",
                        time.time()
                    )

                    print("Transaction cleared.")

                else:

                    difference = camera_count - scanned_count

                    verdict = (
                        "alarm",
                        f"ALERT: {difference} potentially unscanned "
                        f"item(s)",
                        time.time()
                    )

                    print(
                        f"ALERT: {difference} potentially "
                        f"unscanned item(s)."
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
