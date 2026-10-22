"""
ScanGuard — real-webcam prototype
=================================
Counts objects that cross a virtual detection zone in your webcam feed
and compares that count against items "scanned" at the till.

Setup (Python 3.9+):
    pip install ultralytics opencv-python

Run:
    python scanguard_webcam.py

Controls (with the video window focused):
    s      -> simulate scanning one item at the POS
    p      -> "finish & pay": reconcile camera count vs scanned count
    r      -> reset counts
    q/Esc  -> quit

How it works:
  * YOLOv8-nano detects everyday objects (bottle, cup, book, banana, etc.)
  * A simple centroid tracker follows each object between frames
  * When an object's centroid crosses the vertical zone line left->right,
    the camera count increments once for that object
  * On "pay", if camera_count > scanned_count an ALARM banner is shown
"""

import cv2
import time
from ultralytics import YOLO

# ---------- config ----------
CLASSES_OF_INTEREST = {"bottle", "cup", "banana", "apple", "orange",
                       "book", "cell phone", "remote", "mouse", "scissors"}
ZONE_X_RATIO = 0.55        # vertical line position (fraction of frame width)
MATCH_DIST = 80            # px distance to consider it the same object
TRACK_TTL = 1.0            # seconds before a lost track is dropped
CONF = 0.4

model = YOLO("yolov8n.pt")   # downloads automatically on first run

camera_count = 0
scanned_count = 0
verdict = None               # None | ("ok"|"alarm", message, timestamp)
tracks = {}                  # id -> {"pos":(x,y), "last":t, "counted":bool}
next_id = 0


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

        if track_id not in tracks:
            tracks[track_id] = {
                "previous_x": center_x,
                "counted": False
            }
            continue

        previous_x = tracks[track_id]["previous_x"]

        if (
            not tracks[track_id]["counted"]
            and previous_x < zone_x <= center_x
        ):
            camera_count += 1
            tracks[track_id]["counted"] = True
            print(f"Product counted! ID: {track_id}")

        tracks[track_id]["previous_x"] = center_x


def main():
    global scanned_count, camera_count, verdict, tracks
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No webcam found (index 0).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        zone_x = int(w * ZONE_X_RATIO)

        # ---- detect ----
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF,
            verbose=False
            )[0]
        print("Detected objects:", len(results.boxes))
        print("Tracking IDs:", results.boxes.id)        
        dets = []
        for box in results.boxes:
            name = model.names[int(box.cls)]
            if name not in CLASSES_OF_INTEREST:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dets.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 220, 140), 2)
            cv2.putText(frame, name, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 140), 1)

        update_tracks(results, zone_x)

        # ---- overlay ----
        cv2.line(frame, (zone_x, 0), (zone_x, h), (90, 220, 140), 2)
        cv2.putText(frame, "DETECTION LINE  (move items left -> right)",
                    (zone_x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90, 220, 140), 1)
        cv2.rectangle(frame, (0, h - 58), (w, h), (25, 25, 25), -1)
        cv2.putText(frame, f"CAMERA: {camera_count}", (12, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 220, 140), 2)
        cv2.putText(frame, f"SCANNED: {scanned_count}", (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[s] scan  [p] pay  [r] reset  [q] quit",
                    (w - 340, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1)

        if verdict and time.time() - verdict[2] < 5:
            color = (60, 200, 90) if verdict[0] == "ok" else (50, 50, 230)
            cv2.rectangle(frame, (0, 0), (w, 46), color, -1)
            cv2.putText(frame, verdict[1], (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("ScanGuard prototype", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            scanned_count += 1
        elif key == ord("r"):
            camera_count = scanned_count = 0
            tracks = {}
            verdict = None
        elif key == ord("p"):
            if scanned_count >= camera_count:
                verdict = ("ok", f"CLEARED: {camera_count} detected / "
                                 f"{scanned_count} scanned", time.time())
            else:
                diff = camera_count - scanned_count
                verdict = ("alarm", f"ALARM: {diff} item(s) not scanned "
                                    f"({camera_count} detected / "
                                    f"{scanned_count} paid)", time.time())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()