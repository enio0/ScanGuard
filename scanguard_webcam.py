
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
CONF = 0.4

GREEN = (90, 220, 140)
WHITE = (255, 255, 255)

model = YOLO("yolov8n.pt")


# ---------- GLOBAL VARIABLES ----------

camera_count = 0
scanned_count = 0
verdict = None

# track_id -> {"previous_x": int, "counted": bool}
tracks = {}


# ---------- OBJECT TRACKING ----------

def update_tracks(results, zone_x):
    global camera_count

    if results.boxes.id is None:
        return

    for box in results.boxes:

        name = model.names[int(box.cls)]

        if name not in CLASSES_OF_INTEREST:
            continue

        track_id = int(box.id.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2

        # First time seeing this object
        if track_id not in tracks:
            tracks[track_id] = {
                "previous_x": center_x,
                "counted": False
            }
            continue

        previous_x = tracks[track_id]["previous_x"]

        # Count object crossing from left to right
        if (
            not tracks[track_id]["counted"]
            and previous_x < zone_x <= center_x
        ):
            camera_count += 1
            tracks[track_id]["counted"] = True

            print(f"Product counted! ID: {track_id}")

        # Update object's position
        tracks[track_id]["previous_x"] = center_x


# ---------- MAIN APPLICATION ----------

def main():
    global camera_count, scanned_count, verdict, tracks

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise SystemExit("No webcam found (index 0).")

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            h, w = frame.shape[:2]
            zone_x = int(w * ZONE_X_RATIO)

            # ---------- YOLO + BYTETRACK ----------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONF,
                verbose=False
            )[0]

            # ---------- DRAW DETECTIONS ----------

            for box in results.boxes:

                name = model.names[int(box.cls)]

                if name not in CLASSES_OF_INTEREST:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get ByteTrack ID
                track_id = (
                    int(box.id.item())
                    if box.id is not None
                    else -1
                )

                label = f"{name} ID:{track_id}"

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    GREEN,
                    2
                )

                # Object name + tracking ID
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    GREEN,
                    2
                )

            # ---------- UPDATE OBJECT COUNT ----------

            update_tracks(results, zone_x)

            # ---------- DETECTION LINE ----------

            cv2.line(
                frame,
                (zone_x, 0),
                (zone_x, h),
                GREEN,
                2
            )

            cv2.putText(
                frame,
                "DETECTION LINE (left -> right)",
                (min(zone_x + 8, max(w - 300, 0)), 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GREEN,
                1
            )

            # ---------- BOTTOM DASHBOARD ----------

            cv2.rectangle(
                frame,
                (0, h - 70),
                (w, h),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"CAMERA: {camera_count}",
                (12, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                GREEN,
                2
            )

            cv2.putText(
                frame,
                f"SCANNED: {scanned_count}",
                (12, h - 18),
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
                (180, 180, 180),
                1
            )

            # ---------- PAYMENT VERDICT ----------

            if verdict and time.time() - verdict[2] < 5:

                color = (
                    (60, 200, 90)
                    if verdict[0] == "ok"
                    else (50, 50, 230)
                )

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (w, 46),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    verdict[1],
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    WHITE,
                    2
                )

            # ---------- SHOW WEBCAM ----------

            cv2.imshow("ScanGuard — ByteTrack", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("s"):
                scanned_count += 1

            elif key == ord("r"):
                camera_count = 0
                scanned_count = 0
                tracks.clear()
                verdict = None

                # Clear ByteTrack's internal tracking state
                if model.predictor is not None:
                    for tracker in getattr(model.predictor, "trackers", []):
                        tracker.reset()

                print("Transaction reset.")

            elif key == ord("p"):

                if scanned_count >= camera_count:

                    verdict = (
                        "ok",
                        f"CLEARED: {camera_count} detected / "
                        f"{scanned_count} scanned",
                        time.time()
                    )

                else:

                    diff = camera_count - scanned_count

                    verdict = (
                        "alarm",
                        f"ALERT: {diff} potential unscanned item(s) "
                        f"({camera_count} detected / "
                        f"{scanned_count} scanned)",
                        time.time()
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
