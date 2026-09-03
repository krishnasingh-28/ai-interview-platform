"""Typed data passed between the vision and monitoring layers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeadPoseResult:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    direction: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass
class GazeResult:
    direction: str = "UNKNOWN"
    confidence: float = 0.0


@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class FrameQuality:
    score: float
    brightness: float
    blur_variance: float
    is_problematic: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class FrameFeatures:
    timestamp: float
    face_count: int = 0
    face_present: bool = False
    face_confidence: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    head_direction: str = "UNKNOWN"
    gaze_direction: str = "UNKNOWN"
    gaze_confidence: float = 0.0
    phone_detected: bool = False
    person_count: int = 0
    face_bbox: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[int, int]] = field(default_factory=list)
    object_detections: list[ObjectDetection] = field(default_factory=list)
    frame_quality: FrameQuality | None = None


@dataclass
class TemporalState:
    face_absent_duration: float = 0.0
    multiple_face_duration: float = 0.0
    head_turn_duration: float = 0.0
    gaze_deviation_duration: float = 0.0
    phone_duration: float = 0.0
    camera_problem_duration: float = 0.0
    face_absent_active: bool = False
    multiple_face_active: bool = False
    head_turn_active: bool = False
    gaze_deviation_active: bool = False
    phone_active: bool = False
    camera_problem_active: bool = False


@dataclass
class IntegrityEvent:
    event_id: str
    event_type: str
    start_time: float
    end_time: float | None = None
    duration: float = 0.0
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    frame_count: int = 0
    elapsed_seconds: float = 0.0
    average_fps: float = 0.0
    face_absent_seconds: float = 0.0
    gaze_deviation_seconds: float = 0.0
    head_turn_seconds: float = 0.0
    phone_detection_count: int = 0
    multiple_face_count: int = 0
