
"""
ScanGuard — AI-Powered Smart Checkout Monitoring
================================================
Version 1.3.1 — Improved Detection

Features:
- YOLOv8 object detection
- ByteTrack object tracking
- Improved detection of smaller objects
- Tracking IDs displayed on webcam
- Detection zone with hysteresis
- Temporary tracking ID recovery
- Simulated product scanning
- Payment discrepancy alerts

Controls:
    S     -> Scan an item
    P     -> Finish payment
    R     -> Reset transaction
    Q/Esc -> Quit
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO


# ---------- CONFIGURATION ----------

CLASSES_OF_INTEREST = {
    "bottle", "cup", "banana", "apple", "orange",
    "book", "cell phone", "remote", "mouse", "scissors"
}

# Detection
CONF = 0.20
IMAGE_SIZE = 960

# Webcam requested resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Counting zone
ZONE_X_RATIO = 0.55
ZONE_MARGIN = 25

# Temporary object recovery
RECONNECT_DISTANCE = 70
RECONNECT_TIME = 1.5

# Colors (BGR)
GREEN = (90, 220, 140)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
RED = (50, 50, 230)
DARK = (25, 25, 25)

# Files
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"
TRACKER_PATH = BASE_DIR / "custom_bytetrack.yaml"

if not TRACKER_PATH.is_file():
    raise SystemExit(
        "Missing custom_bytetrack.yaml"
    )

model = YOLO(str(MODEL_PATH))


# ---------- GLOBAL VARIABLES ----------

camera_count = 0
scanned_count = 0
verdict = None

# Each tracked object stores:
# center, name, last_seen, armed, counted
tracks = {}


# ---------- OBJECT TRACKING ----------

def update_tracks(results, zone_x):

    global camera_count

    now = time.monotonic()

    left_limit = zone_x - ZONE_MARGIN
    right_limit = zone_x + ZONE_MARGIN

    active_ids = set()

    # Collect visible tracking IDs
    for box in results.boxes:

        name = model.names[int(box.cls.item())]

        if name not in CLASSES_OF_INTEREST:
            continue

        if box.id is not None:
            active_ids.add(int(box.id.item()))

    # Process tracked detections
    for box in results.boxes:

        name = model.names[int(box.cls.item())]

        if name not in CLASSES_OF_INTEREST:
            continue

        # Some YOLO detections do not have a tracking ID
        if box.id is None:
            continue

        track_id = int(box.id.item())

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # ---------- NEW TRACK ----------

        if track_id not in tracks:

            previous_id = None
            best_distance = RECONNECT_DISTANCE

            # Try to reconnect a recently lost track
            for old_id, old in tracks.items():

                if old_id in active_ids:
                    continue

                if not old["counted"]:
                    continue

                if old["name"] != name:
                    continue

                age = now - old["last_seen"]

                if age > RECONNECT_TIME:
                    continue

                old_x, old_y = old["center"]

                dx = center_x - old_x
                dy = center_y - old_y

                distance = (dx * dx + dy * dy) ** 0.5

                if distance < best_distance:

                    best_distance = distance
                    previous_id = old_id

            # Recover possible previous identity
            if previous_id is not None:

                old_track = tracks.pop(previous_id)

                tracks[track_id] = {
                    "center": (center_x, center_y),
                    "name": name,
                    "last_seen": now,
                    "armed": old_track["armed"],
                    "counted": True
                }

                print(
                    f"Possible ID recovery: "
                    f"{previous_id} -> {track_id}"
                )

            else:

                tracks[track_id] = {
                    "center": (center_x, center_y),
                    "name": name,
                    "last_seen": now,
                    "armed": False,
                    "counted": False
                }

        # ---------- UPDATE TRACK ----------

        current = tracks[track_id]

        # Arm object on the left
        if center_x <= left_limit:
            current["armed"] = True

        # Count when reaching the right side
        if (
            current["armed"]
            and not current["counted"]
            and center_x >= right_limit
        ):

            camera_count += 1
            current["counted"] = True

            print(
                f"COUNTED: {name} | "
                f"ID: {track_id} | "
                f"Total: {camera_count}"
            )

        # Save latest information
        current["center"] = (center_x, center_y)
        current["name"] = name
        current["last_seen"] = now

    # ---------- REMOVE OLD RECORDS ----------

    for old_id in list(tracks):

        if old_id in active_ids:
            continue

        age = now - tracks[old_id]["last_seen"]

        if age > RECONNECT_TIME:
            del tracks[old_id]


# ---------- RESET TRANSACTION ----------

def reset_transaction():

    global camera_count, scanned_count, verdict

    camera_count = 0
    scanned_count = 0
    verdict = None

    tracks.clear()

    # Reset ByteTrack
    if model.predictor is not None:

        trackers = getattr(
            model.predictor,
            "trackers",
            []
        )

        for tracker in trackers:
            tracker.reset()

    print("Transaction reset.")


# ---------- PAYMENT CHECK ----------

def check_payment():

    global verdict

    if scanned_count >= camera_count:

        verdict = (
            "ok",
            f"CLEARED: {camera_count} detected / "
            f"{scanned_count} scanned",
            time.monotonic()
        )

        print("Transaction cleared.")

    else:

        difference = camera_count - scanned_count

        verdict = (
            "alarm",
            f"ALERT: {difference} potentially unscanned item(s)",
            time.monotonic()
        )

        print(
            f"ALERT: {difference} potentially unscanned item(s)."
        )


# ---------- DRAW INTERFACE ----------

def draw_interface(frame, results, zone_x, fps):

    h, w = frame.shape[:2]

    left_limit = zone_x - ZONE_MARGIN
    right_limit = zone_x + ZONE_MARGIN

    # ---------- DETECTED OBJECTS ----------

    detections = 0
    tracked = 0

    for box in results.boxes:

        name = model.names[int(box.cls.item())]

        if name not in CLASSES_OF_INTEREST:
            continue

        detections += 1

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        confidence = float(box.conf.item())

        if box.id is not None:

            track_id = int(box.id.item())
            tracked += 1

            counted = (
                track_id in tracks
                and tracks[track_id]["counted"]
            )

            if counted:
                color = ORANGE
                status = "COUNTED"
            else:
                color = GREEN
                status = ""

            label = (
                f"{name} ID:{track_id} "
                f"{confidence:.2f} {status}"
            )

        else:

            # Still display detections without tracking IDs
            color = ORANGE

            label = (
                f"{name} {confidence:.2f} "
                f"UNTRACKED"
            )

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
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # ---------- DETECTION ZONE ----------

    for position, color, thickness in [
        (left_limit, ORANGE, 1),
        (zone_x, GREEN, 2),
        (right_limit, ORANGE, 1)
    ]:

        cv2.line(
            frame,
            (position, 35),
            (position, h - 90),
            color,
            thickness
        )

    # ---------- TOP INFORMATION ----------

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 75),
        DARK,
        -1
    )

    cv2.putText(
        frame,
        "ScanGuard v1.3.1 - Improved Detection",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        GREEN,
        2
    )

    cv2.putText(
        frame,
        f"Detected: {detections} | "
        f"Tracked: {tracked} | "
        f"FPS: {fps:.1f}",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    # ---------- BOTTOM DASHBOARD ----------

    cv2.rectangle(
        frame,
        (0, h - 90),
        (w, h),
        DARK,
        -1
    )

    cv2.putText(
        frame,
        f"CAMERA: {camera_count}",
        (12, h - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN,
        2
    )

    cv2.putText(
        frame,
        f"SCANNED: {scanned_count}",
        (12, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2
    )

    cv2.putText(
        frame,
        "[S] Scan  [P] Pay  [R] Reset  [Q] Quit",
        (12, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        WHITE,
        1
    )

    # ---------- PAYMENT ALERT ----------

    if verdict and time.monotonic() - verdict[2] < 5:

        color = (
            GREEN
            if verdict[0] == "ok"
            else RED
        )

        cv2.rectangle(
            frame,
            (0, 80),
            (w, 125),
            color,
            -1
        )

        cv2.putText(
            frame,
            verdict[1],
            (12, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            2
        )


# ---------- MAIN APPLICATION ----------

def main():

    global scanned_count

    cap = cv2.VideoCapture(0)

    # Request higher webcam resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():
        raise SystemExit("No webcam found.")

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(
        f"Camera resolution: "
        f"{actual_width:.0f}x{actual_height:.0f}"
    )

    previous_time = time.monotonic()

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print("Unable to read webcam frame.")
                break

            h, w = frame.shape[:2]

            zone_x = int(w * ZONE_X_RATIO)

            # ---------- YOLO + BYTETRACK ----------

            results = model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_PATH),
                conf=CONF,
                imgsz=IMAGE_SIZE,
                verbose=False
            )[0]

            # ---------- UPDATE OBJECT COUNT ----------

            update_tracks(results, zone_x)

            # ---------- FPS ----------

            current_time = time.monotonic()
            elapsed = current_time - previous_time

            if elapsed > 0:
                fps = 1.0 / elapsed
            else:
                fps = 0.0

            previous_time = current_time

            # ---------- DRAW INTERFACE ----------

            draw_interface(
                frame,
                results,
                zone_x,
                fps
            )

            # ---------- DISPLAY WEBCAM ----------

            cv2.imshow(
                "ScanGuard - Improved Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key in (ord("q"), 27):
                break

            # Simulated scan
            elif key == ord("s"):

                scanned_count += 1

                print(
                    f"Product scanned! "
                    f"Total: {scanned_count}"
                )

            # Reset
            elif key == ord("r"):

                reset_transaction()

            # Payment
            elif key == ord("p"):

                check_payment()

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
