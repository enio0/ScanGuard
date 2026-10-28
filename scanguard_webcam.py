
"""
ScanGuard — AI-Powered Smart Checkout Monitoring
================================================
Version 1.3 — Duplicate Count Prevention

Features:
- YOLOv8 object detection
- ByteTrack object tracking
- Left-to-right crossing detection
- Temporary tracking ID recovery
- Duplicate count reduction
- Simulated checkout and payment verification

Controls:
    S     -> Scan a product
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

ZONE_X_RATIO = 0.55
ZONE_MARGIN = 25
CONF = 0.4

# Temporary tracking ID recovery
RECONNECT_DISTANCE = 70
RECONNECT_TIME = 1.5

# Colors (OpenCV uses BGR)
GREEN = (90, 220, 140)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
RED = (50, 50, 230)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"
TRACKER_PATH = BASE_DIR / "custom_bytetrack.yaml"

if not TRACKER_PATH.is_file():
    raise SystemExit(
        "Missing custom_bytetrack.yaml."
    )

model = YOLO(str(MODEL_PATH))


# ---------- GLOBAL VARIABLES ----------

camera_count = 0
scanned_count = 0
verdict = None

# Each tracked object stores:
# center: last known position
# name: detected object class
# last_seen: last detection time
# armed: object observed left of the zone
# counted: object already counted
tracks = {}


# ---------- OBJECT TRACKING ----------

def update_tracks(results, zone_x):

    global camera_count

    now = time.monotonic()

    left_limit = zone_x - ZONE_MARGIN
    right_limit = zone_x + ZONE_MARGIN

    active_ids = set()

    # Find all currently tracked IDs
    if results.boxes.id is not None:

        for box in results.boxes:

            name = model.names[int(box.cls)]

            if name not in CLASSES_OF_INTEREST:
                continue

            active_ids.add(int(box.id.item()))

        # Process tracked objects
        for box in results.boxes:

            name = model.names[int(box.cls)]

            if name not in CLASSES_OF_INTEREST:
                continue

            track_id = int(box.id.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ---------- NEW TRACKING ID ----------

            if track_id not in tracks:

                previous_id = None
                best_distance = RECONNECT_DISTANCE

                # Search recently disappeared objects
                for old_id, old in tracks.items():

                    # Never match an object still visible
                    if old_id in active_ids:
                        continue

                    # Only recover previously counted objects
                    if not old["counted"]:
                        continue

                    # Require matching object category
                    if old["name"] != name:
                        continue

                    # Require recent disappearance
                    if now - old["last_seen"] > RECONNECT_TIME:
                        continue

                    old_x, old_y = old["center"]

                    dx = center_x - old_x
                    dy = center_y - old_y

                    distance = (dx * dx + dy * dy) ** 0.5

                    if distance < best_distance:

                        best_distance = distance
                        previous_id = old_id

                # ---------- RECOVER OLD OBJECT ----------

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

                # ---------- CREATE NEW OBJECT ----------

                else:

                    tracks[track_id] = {
                        "center": (center_x, center_y),
                        "name": name,
                        "last_seen": now,
                        "armed": False,
                        "counted": False
                    }

            # ---------- UPDATE OBJECT STATE ----------

            current = tracks[track_id]

            # Arm object when it reaches the left zone
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

            # Save latest position and timestamp
            current["center"] = (center_x, center_y)
            current["name"] = name
            current["last_seen"] = now

    # ---------- CLEAN OLD TRACKS ----------

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

    # Reset ByteTrack's internal state
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


# ---------- MAIN APPLICATION ----------

def main():

    global scanned_count

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
                tracker=str(TRACKER_PATH),
                conf=CONF,
                verbose=False
            )[0]

            # ---------- UPDATE TRACKS ----------

            update_tracks(results, zone_x)

            # ---------- DRAW DETECTED OBJECTS ----------

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

                # Check whether object was counted
                counted = (
                    track_id in tracks
                    and tracks[track_id]["counted"]
                )

                if counted:
                    color = ORANGE
                    label = f"{name} ID:{track_id} COUNTED"

                else:
                    color = GREEN
                    label = f"{name} ID:{track_id}"

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Object label
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

            cv2.line(
                frame,
                (left_limit, 0),
                (left_limit, h - 80),
                ORANGE,
                1
            )

            cv2.line(
                frame,
                (zone_x, 0),
                (zone_x, h - 80),
                GREEN,
                2
            )

            cv2.line(
                frame,
                (right_limit, 0),
                (right_limit, h - 80),
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

            if (
                verdict
                and time.monotonic() - verdict[2] < 5
            ):

                if verdict[0] == "ok":
                    color = GREEN
                else:
                    color = RED

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

            # ---------- DISPLAY WEBCAM ----------

            cv2.imshow("ScanGuard - ByteTrack v1.3", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key in (ord("q"), 27):
                break

            # Simulate scan
            elif key == ord("s"):

                scanned_count += 1

                print(
                    f"Product scanned! "
                    f"Total scanned: {scanned_count}"
                )

            # Reset transaction
            elif key == ord("r"):

                reset_transaction()

            # Check payment
            elif key == ord("p"):

                check_payment()

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
