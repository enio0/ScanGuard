# ScanGuard

ScanGuard is a small Python project that uses a webcam to count everyday objects moving past a line. I built it as an experiment with computer vision and a simple checkout simulation.

It uses YOLOv8 to detect objects and a basic position tracker to avoid counting the same detection in every frame. The **Camera** counter goes up when an object crosses from left to right. The **Scanned** counter is manual, so you can pretend to scan items at a checkout.

## Run it

Use Python 3.10+ and a working webcam.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scanguard_webcam.py
```

The first run may download `yolov8n.pt`.

## Controls

- `S` — simulate a scan
- `P` — compare camera and scanned counts
- `R` — reset the transaction
- `Q` or `Esc` — quit

Click the webcam window first so the keys work.

## Limitations

This is a prototype, not a real theft-detection system. It only recognizes some of the classes in the pretrained YOLO model. For example, a drinks can might be mistaken for a bottle or a phone. Tracking can also lose objects or count something twice, especially if objects overlap or move quickly. A count mismatch is only a reason to check the transaction, not proof of theft.

I kept the project simple on purpose. Future improvements could include a proper barcode scanner, better product recognition, and testing with recorded checkout videos.
