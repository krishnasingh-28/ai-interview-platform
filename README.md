# AI Interview Integrity Monitor

A local, demo-only Streamlit proof of concept for displaying observable camera signals during an interview. It detects sustained conditions such as an absent face, multiple faces, head orientation outside a configured range, approximate gaze deviation, visible cell phones, and persistent camera-quality issues.

It does **not** determine whether somebody is cheating or dishonest. Events are explainable, timestamped visual observations and must not be used as an automated hiring decision.

## Architecture

```text
Browser webcam → Streamlit/WebRTC → OpenCV frame
  → MediaPipe Face Landmarker ─┐
  → YOLO person/phone detector ├→ FrameFeatures → TemporalEngine → EventEngine → dashboard
  → quality / pose / gaze       ┘
```

All state is held in `st.session_state` for the active browser session; no frames, video, or events are written to a database or uploaded.

## Features

- Face count/presence, bounding box, and optional MediaPipe landmarks
- Approximate yaw, pitch, roll and basic pose direction using OpenCV `solvePnP`
- Conservative iris-position gaze approximation; reports `UNKNOWN` when iris landmarks are unavailable
- YOLO overlays for `person` and `cell phone`
- Frame brightness, blur, face-size, and face-quality signals
- Persistent-condition events with a single event lifecycle per uninterrupted episode
- Live status, session metrics, and a Plotly event timeline
- AI Layman Analytical Overview powered by OpenRouter (transforms vision metrics into clear, everyday insights for interviewers)

## Installation

Use Python 3.11 or 3.12 on Windows. Although the application code supports newer
Python syntax, the Windows PyTorch builds used by Ultralytics currently support
Python through 3.12. Create a virtual environment, then install dependencies:

```powershell
cd ai_interview_integrity
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local model setup

The application intentionally does not download models during a session. Obtain compatible assets yourself and place them at these exact project-relative paths:

| Component | Required path | Expected asset |
|---|---|---|
| MediaPipe Tasks Face Landmarker | `models/face_landmarker.task` | The Face Landmarker task bundle from MediaPipe's official model page |
| Ultralytics YOLO | `models/yolo11n.pt` | A locally downloaded Ultralytics YOLO nano `.pt` checkpoint trained on COCO |

The default COCO checkpoint includes class IDs `person` (0) and `cell phone` (67). Missing model assets leave the app running with an on-screen setup message; they do not crash it. If you use different local filenames, change `FACE_LANDMARKER_MODEL` or `YOLO_MODEL` in `config.py`.

## Run

```powershell
streamlit run app.py
```

Choose **Start** in the sidebar and grant the browser camera permission. The camera frames stay in the local Streamlit/WebRTC session. Stop ends processing; Reset clears the in-memory session events and metrics.

## AI Layman Analytics (OpenRouter)

The application includes an AI-powered analytical assistant that translates raw metrics (gaze deviations, head turns, secondary device detections, face absence) into an intuitive, plain-English summary for recruiters and interviewers.

1. Configure your OpenRouter key:
   - Option A: Add `OPENROUTER_API_KEY=your_key` in a local `.env` file (copied from `.env.example`).
   - Option B: Enter the API key directly in the sidebar input in the Streamlit app.
2. Select your desired AI model in the sidebar dropdown (e.g. `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-4o-mini`, `deepseek/deepseek-chat`, etc.).
3. Under the **AI Layman Analytical Summary** section on the dashboard, click **✨ Generate Layman Summary** or use **📋 Offline Rule-Based Summary**.
4. Download the generated professional PDF analytical report or review the summary directly on screen.

## Configuration and event rules

`config.py` holds defaults, while the sidebar exposes the practical demo controls. Frame-level outputs go through `TemporalEngine` before `EventEngine` creates an event.

| Event | Default persistent condition |
|---|---|
| `FACE_NOT_VISIBLE` | no face for 2.0 s |
| `MULTIPLE_FACES` | two or more faces for 1.0 s |
| `PROLONGED_HEAD_TURN` | yaw > 30° or pitch > 25° for 3.0 s |
| `SUSTAINED_GAZE_DEVIATION` | non-center, non-unknown gaze for 2.5 s |
| `POSSIBLE_SECONDARY_DEVICE` | cell phone detected for 1.0 s |
| `CAMERA_OBSTRUCTED` | persistent image/face-quality signal for 2.0 s |

One event opens when its duration first reaches the threshold, remains open while the signal remains true, then closes on recovery. This suppresses one-frame noise and duplicate alerts.

## Tests

The monitoring logic does not need a camera or model asset to test:

```powershell
pytest -q
```

## Limitations and privacy

Pose and gaze are approximate and depend on camera position, lighting, glasses, face visibility, and the chosen model. Gaze is not an indication of intent. Object detection can miss or mislabel items. Reviewers should consider these signals in context, provide notice and accessibility alternatives, limit access, and avoid recording or retention unless a separate policy and consent process permits it.

## Future architecture

For a production assessment, add explicit consent and retention controls, calibrated models evaluated across conditions, human review workflows, accessibility testing, secure audit boundaries, and legal/privacy review. Those additions are deliberately outside this local POC.
