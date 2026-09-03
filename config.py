"""Central defaults for the local interview-integrity demonstration."""
from dataclasses import dataclass

PROCESSING_FPS = 10
YOLO_FPS = 3
HISTORY_SECONDS = 5.0
MAX_FACES = 3
FACE_MODEL_DETECTION_CONFIDENCE = 0.5
FACE_MODEL_PRESENCE_CONFIDENCE = 0.5
FACE_MODEL_TRACKING_CONFIDENCE = 0.5
FACE_CONFIDENCE_AREA_NORMALIZER = 0.18
GAZE_MIN_EYE_WIDTH = 0.01
GAZE_MIN_EYE_HEIGHT = 0.003
GAZE_HORIZONTAL_THRESHOLD = 0.045
GAZE_VERTICAL_THRESHOLD = 0.11
GAZE_CONFIDENCE_BASE = 0.3
GAZE_CONFIDENCE_SCALE = 3.0
GAZE_CONFIDENCE_CAP = 0.65
QUALITY_REASON_PENALTY = 0.25
FACE_ABSENCE_SECONDS = 2.0
MULTIPLE_FACE_SECONDS = 1.0
HEAD_TURN_SECONDS = 3.0
GAZE_DEVIATION_SECONDS = 2.5
PHONE_DETECTION_SECONDS = 1.0
CAMERA_OBSTRUCTION_SECONDS = 2.0
YAW_THRESHOLD = 30.0
PITCH_THRESHOLD = 25.0
MIN_FACE_AREA_RATIO = 0.025
MIN_FACE_CONFIDENCE = 0.45
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 220.0
MIN_BLUR_VARIANCE = 35.0
PHONE_CONFIDENCE = 0.35
YOLO_PERSON_CLASS = 0
YOLO_PHONE_CLASS = 67
YOLO_MODEL = "models/yolo11n.pt"
FACE_LANDMARKER_MODEL = "models/face_landmarker.task"

# OpenRouter Analytics Configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
AVAILABLE_OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat",
    "anthropic/claude-3.5-haiku",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
]


@dataclass
class MonitoringConfig:
    """Runtime-configurable temporal thresholds."""
    processing_fps: int = PROCESSING_FPS
    yolo_fps: int = YOLO_FPS
    face_absence_seconds: float = FACE_ABSENCE_SECONDS
    multiple_face_seconds: float = MULTIPLE_FACE_SECONDS
    head_turn_seconds: float = HEAD_TURN_SECONDS
    gaze_deviation_seconds: float = GAZE_DEVIATION_SECONDS
    phone_detection_seconds: float = PHONE_DETECTION_SECONDS
    camera_obstruction_seconds: float = CAMERA_OBSTRUCTION_SECONDS
    yaw_threshold: float = YAW_THRESHOLD
    pitch_threshold: float = PITCH_THRESHOLD
