"""Converts sustained temporal conditions into one lifecycle event per episode."""
from __future__ import annotations
from uuid import uuid4
from config import MonitoringConfig
from monitoring.models import FrameFeatures, IntegrityEvent, TemporalState

RULES = {
    "FACE_NOT_VISIBLE": ("face_absent", "face_absent_duration", "face_absence_seconds"),
    "MULTIPLE_FACES": ("multiple_face", "multiple_face_duration", "multiple_face_seconds"),
    "PROLONGED_HEAD_TURN": ("head_turn", "head_turn_duration", "head_turn_seconds"),
    "SUSTAINED_GAZE_DEVIATION": ("gaze_deviation", "gaze_deviation_duration", "gaze_deviation_seconds"),
    "POSSIBLE_SECONDARY_DEVICE": ("phone", "phone_duration", "phone_detection_seconds"),
    "CAMERA_OBSTRUCTED": ("camera_problem", "camera_problem_duration", "camera_obstruction_seconds"),
}


class EventEngine:
    """Open events when duration crosses a threshold and close them on recovery."""
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.events: list[IntegrityEvent] = []
        self.open_events: dict[str, IntegrityEvent] = {}

    def update(self, features: FrameFeatures, state: TemporalState) -> list[IntegrityEvent]:
        for event_type, (prefix, duration_key, threshold_key) in RULES.items():
            active, duration = getattr(state, f"{prefix}_active"), getattr(state, duration_key)
            threshold = getattr(self.config, threshold_key)
            event = self.open_events.get(event_type)
            if active and duration >= threshold and event is None:
                start = features.timestamp - duration
                event = IntegrityEvent(uuid4().hex[:10], event_type, start, confidence=self._confidence(event_type, features), metadata={})
                self.events.append(event); self.open_events[event_type] = event
            if event:
                event.end_time = features.timestamp
                event.duration = max(0.0, features.timestamp - event.start_time)
                self._enrich(event, features)
                if not active: del self.open_events[event_type]
        return self.events

    @staticmethod
    def _confidence(event_type: str, f: FrameFeatures) -> float | None:
        if event_type == "FACE_NOT_VISIBLE": return None
        if event_type == "CAMERA_OBSTRUCTED": return 1.0 - f.frame_quality.score if f.frame_quality else None
        if event_type == "SUSTAINED_GAZE_DEVIATION": return f.gaze_confidence or None
        if event_type == "POSSIBLE_SECONDARY_DEVICE":
            phones = [d.confidence for d in f.object_detections if d.label == "cell phone"]
            return max(phones) if phones else None
        return f.face_confidence or None

    @staticmethod
    def _enrich(event: IntegrityEvent, f: FrameFeatures) -> None:
        if event.event_type == "MULTIPLE_FACES": event.metadata["max_face_count"] = max(event.metadata.get("max_face_count", 0), f.face_count)
        elif event.event_type == "PROLONGED_HEAD_TURN": event.metadata.update({"yaw": round(f.yaw, 1), "pitch": round(f.pitch, 1)})
        elif event.event_type == "SUSTAINED_GAZE_DEVIATION": event.metadata["latest_gaze"] = f.gaze_direction
        elif event.event_type == "CAMERA_OBSTRUCTED" and f.frame_quality: event.metadata["reasons"] = f.frame_quality.reasons

    def reset(self) -> None:
        self.events.clear(); self.open_events.clear()
