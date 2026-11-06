# Face Diary

Face Diary is a small desktop face recognition app. It reads frames from your webcam, detects faces, and compares their face embeddings with people registered in a local SQLite database. Known faces are labeled by name; unmatched faces appear as **Unknown person**.

## Requirements

- Python 3.10 or newer
- A webcam
- A desktop session that can display OpenCV windows

The dependencies are listed in [`requirements.txt`](requirements.txt): OpenCV, InsightFace, NumPy, and ONNX Runtime. InsightFace may download its recognition model the first time it runs, so an internet connection may be needed on first launch.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install and start the app:

```bash
python -m pip install -r requirements.txt
python main.py
```

If you have multiple cameras and the default camera does not open, change the `cv2.VideoCapture(0)` device index in `main.py` to the appropriate camera index.

## Use

Click the **Camera Feed** window so it receives keyboard input.

- Press **R** to register a person. Keep exactly one face visible, type a name in the camera window, and press **Enter**. The app then captures 15 face samples. Press **Esc** while entering the name to cancel.
- Press **Q** to quit.

Registration is declined when the face matches someone already in the database. Names are limited to 32 characters. The app needs one visible face to begin registration.

## Data and privacy

Registered names and face embeddings are stored locally in `database/faces.db`. The database is initialized automatically when the app starts. Face embeddings are biometric data; protect the database accordingly and only register people with their knowledge and consent. No cloud storage or remote service is configured by this project.

To remove registered data, close the app and delete `database/faces.db`; it will be recreated on the next launch. This permanently removes the local face records.

## Project layout

```text
main.py                 Webcam app and interaction loop
recognition/face_engine.py  Face detection, embeddings, and matching
registration/register.py   Face sample capture for registration
database/database.py       SQLite storage helpers
database/faces.db          Local registered-face database (created/used at runtime)
testface.py                Standalone webcam face-detection check
```

To run the standalone detection check, use `python testface.py`; press **X** to close it.
