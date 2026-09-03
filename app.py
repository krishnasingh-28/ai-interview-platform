"""Local Streamlit interview-integrity demonstration application.

The app observes video signals only.  It never labels a person as dishonest and
does not persist recorded video or session data.
"""
from __future__ import annotations
import threading
import time

import av
import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from config import (
    AVAILABLE_OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    FACE_LANDMARKER_MODEL,
    MonitoringConfig,
    YOLO_MODEL,
)
from monitoring.event_engine import EventEngine
from monitoring.feature_extractor import build_features
from monitoring.temporal_engine import TemporalEngine
from analytics.openrouter_client import OpenRouterClient, load_env_api_key
from analytics.report_generator import generate_layman_analysis, generate_offline_fallback_report
from monitoring.metrics import calculate_metrics
from utils.drawing import draw_overlays
from utils.pdf_generator import generate_pdf_report
from utils.time_utils import format_seconds
from vision.face_detector import FaceDetector
from vision.frame_quality import assess_frame_quality
from vision.gaze_estimator import estimate_gaze
from vision.head_pose import estimate_head_pose
from vision.object_detector import ObjectDetector

st.set_page_config(page_title="AI Interview Integrity Monitor", page_icon="◉", layout="wide")


@st.cache_resource(show_spinner="Loading local vision models…")
def load_models() -> tuple[FaceDetector, ObjectDetector]:
    """Load local model assets once per Streamlit server process."""
    return FaceDetector(FACE_LANDMARKER_MODEL), ObjectDetector(YOLO_MODEL)


def apply_dark_styles() -> None:
    """Keep custom dashboard elements legible with the configured dark theme."""
    st.markdown(
        """
        <style>
        .stApp { background: #0B1220; }
        [data-testid="stHeader"] { background: rgba(11, 18, 32, 0.88); }
        [data-testid="stSidebar"] { background: #101A2B; border-right: 1px solid #26354D; }
        [data-testid="stMetric"] {
            background: #151F32;
            border: 1px solid #26354D;
            border-radius: 10px;
            padding: 0.75rem;
        }
        [data-testid="stMetricLabel"] { color: #A9B9D0; }
        [data-testid="stMetricValue"] { color: #E5EDF8; }
        [data-testid="stAlert"] { border-radius: 8px; }
        [data-testid="stExpander"] { background: #151F32; border-color: #26354D; }
        .ai-summary-container {
            background: #131E31;
            border: 1px solid #2B3D5B;
            border-radius: 10px;
            padding: 1.25rem;
            margin-top: 0.75rem;
            color: #E2E8F0;
        }
        .ai-badge {
            background: #1E2D4A;
            color: #60A5FA;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.5rem;
            border: 1px solid #3B82F6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_dark_styles()


class SessionRuntime:
    """Thread-safe in-memory state accessed by WebRTC's video callback."""
    def __init__(self, config: MonitoringConfig) -> None:
        self.lock = threading.Lock()
        self.config = config
        self.temporal = TemporalEngine(config)
        self.events = EventEngine(config)
        self.start_monotonic: float | None = None
        self.frame_count = 0
        self.last_processed = 0.0
        self.last_yolo = 0.0
        self.last_detections = []
        self.current = None

    def start(self, config: MonitoringConfig) -> None:
        with self.lock:
            self.config = config
            self.temporal = TemporalEngine(config)
            self.events = EventEngine(config)
            self.start_monotonic, self.frame_count = time.monotonic(), 0
            self.last_processed = self.last_yolo = 0.0
            self.last_detections, self.current = [], None

    def reset(self) -> None:
        self.start(self.config)
        with self.lock:
            self.start_monotonic = None


class MonitoringVideoProcessor(VideoProcessorBase):
    """Processes frames at configured rates and stores results in SessionRuntime."""
    def __init__(self, runtime: SessionRuntime, face_detector: FaceDetector, object_detector: ObjectDetector,
                 show_landmarks: bool, show_yolo: bool) -> None:
        self.runtime, self.face_detector, self.object_detector = runtime, face_detector, object_detector
        self.show_landmarks, self.show_yolo = show_landmarks, show_yolo

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        with self.runtime.lock:
            if self.runtime.start_monotonic is None:
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            cfg = self.runtime.config
            if now - self.runtime.last_processed < 1.0 / max(cfg.processing_fps, 1):
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            session_time = now - self.runtime.start_monotonic
            self.runtime.last_processed = now
            if now - self.runtime.last_yolo >= 1.0 / max(cfg.yolo_fps, 1):
                self.runtime.last_detections = self.object_detector.detect(image)
                self.runtime.last_yolo = now
            detections = self.runtime.last_detections
            # MediaPipe VIDEO mode requires a timestamp that never moves backwards,
            # including after a user resets an interview session.
            face = self.face_detector.detect(image, int(now * 1000))
            pose = estimate_head_pose(face.landmarks, image.shape, cfg.yaw_threshold, cfg.pitch_threshold)
            gaze = estimate_gaze(face.landmarks)
            quality = assess_frame_quality(image, face.bbox, face.confidence)
            features = build_features(session_time, face, pose, gaze, detections, quality)
            state = self.runtime.temporal.update(features)
            self.runtime.events.update(features, state)
            self.runtime.current = features
            self.runtime.frame_count += 1
        return av.VideoFrame.from_ndarray(draw_overlays(image, features, self.show_landmarks, self.show_yolo), format="bgr24")


def initialise_state() -> None:
    if "runtime" not in st.session_state:
        st.session_state.runtime = SessionRuntime(MonitoringConfig())
    if "active" not in st.session_state:
        st.session_state.active = False
    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = None
    if "ai_model_used" not in st.session_state:
        st.session_state.ai_model_used = None
    if "ai_error" not in st.session_state:
        st.session_state.ai_error = None


def sidebar_config() -> tuple[MonitoringConfig, bool, bool, bool, str | None, str, str]:
    st.sidebar.header("Session controls")
    start_col, stop_col, reset_col = st.sidebar.columns(3)
    action = "Start Interview" if start_col.button("Start Interview") else None
    action = "Stop Interview" if stop_col.button("Stop Interview") else action
    action = "Reset Session" if reset_col.button("Reset Session") else action
    fps = st.sidebar.select_slider("Processing FPS", options=[5, 10, 15], value=10)
    st.sidebar.subheader("Detection thresholds")
    config = MonitoringConfig(
        processing_fps=fps,
        yolo_fps=max(1, min(5, fps // 2)),
        yaw_threshold=st.sidebar.slider("Head yaw threshold", 15.0, 50.0, 30.0),
        pitch_threshold=st.sidebar.slider("Head pitch threshold", 15.0, 45.0, 25.0),
        gaze_deviation_seconds=st.sidebar.slider("Gaze deviation duration", 1.0, 6.0, 2.5),
        face_absence_seconds=st.sidebar.slider("Face absence duration", 0.5, 5.0, 2.0),
        multiple_face_seconds=st.sidebar.slider("Multiple-face duration", 0.5, 4.0, 1.0),
        phone_detection_seconds=st.sidebar.slider("Phone duration", 0.5, 4.0, 1.0),
    )
    config.head_turn_seconds = st.sidebar.slider("Head-turn duration", 1.0, 6.0, 3.0)
    config.camera_obstruction_seconds = st.sidebar.slider("Camera issue duration", 0.5, 5.0, 2.0)
    st.sidebar.subheader("Overlays")
    show_landmarks = st.sidebar.toggle("Show landmarks", True)
    show_yolo = st.sidebar.toggle("Show YOLO boxes", True)
    debug = st.sidebar.toggle("Show debug metrics", False)

    st.sidebar.subheader("AI Layman Analytics (OpenRouter)")
    saved_key = load_env_api_key()
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        value=saved_key,
        type="password",
        help="OpenRouter API Key for generating plain-English analytical overviews.",
    )
    selected_model = st.sidebar.selectbox(
        "OpenRouter Model",
        options=AVAILABLE_OPENROUTER_MODELS,
        index=0,
        help="Select the AI model for summary generation.",
    )

    return config, show_landmarks, show_yolo, debug, action, api_key, selected_model


def severity(event_type: str) -> str:
    return "🔴" if event_type in {"FACE_NOT_VISIBLE", "MULTIPLE_FACES", "POSSIBLE_SECONDARY_DEVICE"} else "🟡"


def render_ai_analytics_section(metrics, events, api_key: str, selected_model: str) -> None:
    """Renders the OpenRouter Layman Analytical Summary generator and results."""
    st.subheader("🤖 AI Layman Analytical Summary")
    st.caption("Generate a plain-English overview of overall session metrics, attention patterns, and integrity indicators powered by OpenRouter.")

    col1, col2, col3 = st.columns([1.5, 1.3, 0.8])
    generate_btn = col1.button("✨ Generate Layman Summary", type="primary", use_container_width=True)
    fallback_btn = col2.button("📋 Offline Rule-Based Summary", use_container_width=True)
    clear_btn = col3.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.ai_summary = None
        st.session_state.ai_model_used = None
        st.session_state.ai_error = None
        st.rerun()

    if generate_btn:
        if not api_key:
            st.session_state.ai_error = "OpenRouter API Key is missing. Please enter your API key in the sidebar."
        else:
            with st.spinner(f"Analyzing session statistics with {selected_model} via OpenRouter…"):
                client = OpenRouterClient(api_key=api_key, model=selected_model)
                res = generate_layman_analysis(metrics, events, client)
                if res.get("success"):
                    st.session_state.ai_summary = res.get("content", "")
                    st.session_state.ai_model_used = res.get("model", selected_model)
                    st.session_state.ai_error = None
                else:
                    st.session_state.ai_error = res.get("error", "Unknown error generating summary.")

    if fallback_btn:
        st.session_state.ai_summary = generate_offline_fallback_report(metrics, events)
        st.session_state.ai_model_used = "Rule-Based Analytical Engine (Offline)"
        st.session_state.ai_error = None

    if st.session_state.ai_error:
        st.error(st.session_state.ai_error)

    if st.session_state.ai_summary:
        model_badge = f'<div class="ai-badge">Analyzed with {st.session_state.ai_model_used}</div>'
        st.markdown(model_badge, unsafe_allow_html=True)
        st.markdown(f'<div class="ai-summary-container">\n\n{st.session_state.ai_summary}\n\n</div>', unsafe_allow_html=True)
        pdf_data = generate_pdf_report(
            summary_markdown=st.session_state.ai_summary,
            metrics=metrics,
            events=events,
            model_used=st.session_state.ai_model_used or "Layman Analytical Engine",
        )
        st.download_button(
            label="📥 Download Analytical Report (.pdf)",
            data=pdf_data,
            file_name="interview_integrity_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def render_dashboard(runtime: SessionRuntime, debug: bool, api_key: str, selected_model: str) -> None:
    with runtime.lock:
        current = runtime.current
        elapsed = time.monotonic() - runtime.start_monotonic if runtime.start_monotonic else 0.0
        events = list(runtime.events.events)
        count = runtime.frame_count
        gaze_center_percentage = runtime.temporal.gaze_center_percentage()
    metrics = calculate_metrics(count, elapsed, events)
    st.subheader("Live status")
    if current is None:
        st.info("Start the session and allow browser camera access to receive local camera signals.")
    else:
        # Check for active red flags in live frame
        red_flags = []
        if not current.face_present:
            red_flags.append("🔴 Face Not Visible")
        if current.phone_detected:
            red_flags.append("🔴 Possible Secondary Device")
        if current.person_count > 1 or current.face_count > 1:
            count_label = max(current.person_count, current.face_count)
            red_flags.append(f"🔴 Multiple Faces/Persons ({count_label})")

        if red_flags:
            st.error("Active Alert: " + " | ".join(red_flags))

        face_status = "✓ Detected" if current.face_present else "🔴 Not visible"
        person_status = f"🔴 {current.person_count} Detected" if current.person_count > 1 else str(current.person_count)
        phone_status = "🔴 Possible device" if current.phone_detected else "✓ Not detected"

        a, b = st.columns(2)
        a.metric("Face", face_status)
        a.metric("Persons (YOLO)", person_status)
        a.metric("Camera", "Needs attention" if current.frame_quality and current.frame_quality.is_problematic else "✓ Good")
        a.metric("Identity", "N/A / Demo")
        b.metric("Gaze", current.gaze_direction)
        b.metric("Head pose", current.head_direction)
        b.metric("Phone", phone_status)
        b.metric("Gaze center (5 s)", "N/A" if gaze_center_percentage is None else f"{gaze_center_percentage:.0f}%")
        if debug:
            st.code(f"yaw={current.yaw:.1f}, pitch={current.pitch:.1f}, roll={current.roll:.1f} | "
                    f"gaze confidence={current.gaze_confidence:.2f} | face quality={current.face_confidence:.2f}\n"
                    f"frame quality={current.frame_quality}")
    st.subheader("Integrity events")
    if not events:
        st.caption("No sustained observable events recorded in this session.")
    for event in reversed(events):
        end = event.end_time if event.end_time is not None else elapsed
        duration = max(event.duration, end - event.start_time)
        confidence = f" · confidence {event.confidence:.2f}" if event.confidence is not None else ""
        st.write(f"{severity(event.event_type)} **{event.event_type.replace('_', ' ').title()}** — {duration:.1f}s{confidence}")
    st.subheader("Session statistics")
    cols = st.columns(4)
    cols[0].metric("Duration", format_seconds(metrics.elapsed_seconds))
    cols[1].metric("Frames / Avg FPS", f"{metrics.frame_count} / {metrics.average_fps:.1f}")
    cols[2].metric("Events", len(events))
    cols[3].metric("Face absent", f"{metrics.face_absent_seconds:.1f}s")
    detail = st.columns(4)
    detail[0].metric("Multiple-face events", metrics.multiple_face_count)
    detail[1].metric("Gaze deviation", f"{metrics.gaze_deviation_seconds:.1f}s")
    detail[2].metric("Head-turn duration", f"{metrics.head_turn_seconds:.1f}s")
    detail[3].metric("Phone episodes", metrics.phone_detection_count)
    if events:
        records = [{"event": e.event_type.replace("_", " ").title(), "start": e.start_time,
                    "end": e.end_time if e.end_time is not None else elapsed} for e in events]
        fig = px.timeline(pd.DataFrame(records), x_start="start", x_end="end", y="event", color="event", height=220)
        fig.update_layout(showlegend=False, xaxis_title="Interview time (seconds)", yaxis_title=None, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Render OpenRouter AI Layman Analytical Summary section
    render_ai_analytics_section(metrics, events, api_key, selected_model)


def main() -> None:
    initialise_state()
    config, show_landmarks, show_yolo, debug, action, api_key, selected_model = sidebar_config()
    runtime: SessionRuntime = st.session_state.runtime
    if action == "Start Interview" and not st.session_state.active:
        runtime.start(config); st.session_state.active = True
    elif action == "Stop Interview":
        st.session_state.active = False
    elif action == "Reset Session":
        runtime.reset()
        st.session_state.active = False
        st.session_state.ai_summary = None
        st.session_state.ai_model_used = None
        st.session_state.ai_error = None
    st.title("AI Interview Integrity Monitor")
    st.caption("Computer Vision Based Interview Monitoring POC")
    st.warning("Camera signals are used only to identify observable integrity events. These signals do not establish cheating or dishonesty and should not be used as an automated hiring decision.")
    face_model, object_model = load_models()
    model_warnings = [message for message in (face_model.error, object_model.error) if message]
    if model_warnings:
        st.info("Model setup: " + " | ".join(model_warnings) + ". See README for local asset setup.")
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Live camera")
        webrtc_streamer(key="integrity-camera", mode=WebRtcMode.SENDRECV, media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: MonitoringVideoProcessor(runtime, face_model, object_model, show_landmarks, show_yolo),
            async_processing=True)
    with right:
        render_dashboard(runtime, debug, api_key, selected_model)
    if st.session_state.active:
        # A periodic rerun refreshes status cards; the WebRTC callback owns frame processing.
        time.sleep(0.25)
        st.rerun()


if __name__ == "__main__":
    main()
