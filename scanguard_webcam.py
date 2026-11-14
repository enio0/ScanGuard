
import time

import cv2
from ultralytics import YOLO

MODEL = "yolov8n.pt"
CONFIDENCE = 0.35
LINE_POSITION = 0.5
MATCH_DISTANCE = 100
TRACK_TIMEOUT = 1.0

# These are the everyday objects recognized by the pretrained model.
ITEMS = {"bottle", "cup", "banana", "apple", "orange", "book", "cell phone", "remote", "mouse", "scissors"}

model = YOLO(MODEL)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    camera_count = 0
    scanned_count = 0
    tracks = {}  # id: {x, y, last_seen, counted}
    next_id = 0
    message = ""
    message_time = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            line_x = int(width * LINE_POSITION)
            now = time.monotonic()
            result = model.predict(frame, conf=CONFIDENCE, verbose=False)[0]
            detections = []

            for box in result.boxes:
                name = model.names[int(box.cls.item())]
                if name not in ITEMS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, name, (x1, y1, x2, y2)))

            # Match each detection to a nearby object from the previous frames.
            matches = set()
            for cx, cy, name, bounds in detections:
                chosen_id = None
                best_distance = MATCH_DISTANCE
                for track_id, track in tracks.items():
                    if track_id in matches or track["name"] != name:
                        continue
                    if now - track["last_seen"] > TRACK_TIMEOUT:
                        continue
                    distance = ((cx - track["x"]) ** 2 + (cy - track["y"]) ** 2) ** 0.5
                    if distance < best_distance:
                        chosen_id = track_id
                        best_distance = distance

                if chosen_id is None:
                    chosen_id = next_id
                    next_id += 1
                    tracks[chosen_id] = {
                        "x": cx, "y": cy, "name": name,
                        "last_seen": now, "counted": False
                    }
                else:
                    track = tracks[chosen_id]
                    if not track["counted"] and track["x"] < line_x <= cx:
                        camera_count += 1
                        track["counted"] = True
                        print(f"Counted: {name} (camera total: {camera_count})")
                    track["x"] = cx
                    track["y"] = cy
                    track["last_seen"] = now

                matches.add(chosen_id)
                x1, y1, x2, y2 = bounds
                color = (0, 165, 255) if tracks[chosen_id]["counted"] else (80, 220, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Forget objects that have left the scene.
            tracks = {
                track_id: track for track_id, track in tracks.items()
                if now - track["last_seen"] <= TRACK_TIMEOUT
            }

            cv2.line(frame, (line_x, 0), (line_x, height), (80, 220, 120), 2)
            cv2.rectangle(frame, (0, height - 85), (width, height), (30, 30, 30), -1)
            cv2.putText(frame, f"Camera: {camera_count}   Scanned: {scanned_count}",
                        (12, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, "S: scan   P: pay   R: reset   Q: quit",
                        (12, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            if message and now - message_time < 4:
                cv2.rectangle(frame, (0, 0), (width, 45), (30, 30, 30), -1)
                cv2.putText(frame, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2)

            cv2.imshow("ScanGuard", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                scanned_count += 1
                print(f"Scanned: {scanned_count}")
            elif key == ord("p"):
                if camera_count > scanned_count:
                    message = f"Check: {camera_count - scanned_count} possible unscanned item(s)"
                else:
                    message = "Counts match (or more items were scanned)"
                message_time = time.monotonic()
                print(message)
            elif key == ord("r"):
                camera_count = 0
                scanned_count = 0
                tracks.clear()
                message = ""
                print("Reset")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
