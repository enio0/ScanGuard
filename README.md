# ScanGuard

ScanGuard is a small computer vision project I built using Python, OpenCV, and YOLOv8. The idea was to experiment with object detection and see whether a webcam could help identify items that pass through a self-checkout area without being scanned.

The program detects everyday objects, tracks their movement, and counts them when they cross a virtual line on the screen. It then compares the camera count with the number of items manually scanned in a simple checkout simulation.

This is an experimental prototype rather than a complete checkout or security system.

## Technologies Used

- **Python** — Main programming language
- **OpenCV** — Webcam input, object tracking, and visual interface
- **YOLOv8** — Real-time object detection
- **Git and GitHub** — Version control

## How It Works

The application uses a webcam to capture video in real time.

YOLOv8 detects objects such as bottles, cups, books, and other everyday items. A basic position-based tracker follows these objects between frames to reduce repeated counting.

A vertical line divides the camera view. When a tracked object moves from left to right across the line, the camera counter increases.

The program maintains two counters:

- **Camera:** Number of objects detected crossing the line.
- **Scanned:** Number of items manually registered at checkout.

At the end of a transaction, the two counters are compared. If the camera count is greater than the scanned count, the program displays an alert about a possible discrepancy.

## Installation

The project requires Python 3.10 or newer and a working webcam.

Clone the repository:

```bash
git clone https://github.com/enio0/ScanGuard.git
cd ScanGuard
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the program:

```bash
python3 scanguard_webcam.py
```

The YOLOv8 model weights (`yolov8n.pt`) will be downloaded automatically if they are not already available.

## Controls

| Key | Action |
|---|---|
| `S` | Simulate scanning an item |
| `P` | Finish the transaction and compare counts |
| `R` | Reset the counters |
| `Q` or `Esc` | Close the application |

Make sure the webcam window is selected when using the keyboard controls.

## Example

Suppose three objects cross the detection line, but only two items are registered as scanned.

```text
CAMERA: 3
SCANNED: 2
```

When the payment check is performed, ScanGuard displays an alert indicating that one item may not have been scanned.

If both counters match, the transaction is cleared.

## Limitations

ScanGuard is a prototype, and its accuracy depends on several factors.

The pretrained YOLOv8 model only recognizes certain object categories. For example, it might mistake a drink can for a bottle or another object.

Tracking can also lose objects when they move quickly, overlap, or disappear from the webcam. This can result in missed detections or duplicate counts.

The scanned counter is simulated using the keyboard. There is no actual barcode scanner or payment integration.

A difference between the counters does not necessarily indicate theft. It only highlights a possible discrepancy.


For now, I decided to keep the project as a simple experiment in object detection, tracking, and Python programming.