"""Rolling-window persistence calculations, independent of event decisions."""
from __future__ import annotations
from collections import deque
from config import HISTORY_SECONDS, MonitoringConfig
from monitoring.models import FrameFeatures, TemporalState


class TemporalEngine:
    """Track continuous true-condition durations from timestamped frame features."""
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.history: deque[FrameFeatures] = deque()
        self.starts: dict[str, float | None] = {key: None for key in ("face_absent", "multiple_face", "head_turn", "gaze_deviation", "phone", "camera_problem")}
        self.last_time: float | None = None

    def update(self, features: FrameFeatures) -> TemporalState:
        if self.last_time is not None and features.timestamp < self.last_time:
            self.reset()
        self.last_time = features.timestamp
        self.history.append(features)
        while self.history and features.timestamp - self.history[0].timestamp > HISTORY_SECONDS:
            self.history.popleft()
        conditions = {
            "face_absent": not features.face_present,
            "multiple_face": features.face_count >= 2,
            "head_turn": features.face_present and (abs(features.yaw) > self.config.yaw_threshold or abs(features.pitch) > self.config.pitch_threshold),
            "gaze_deviation": features.face_present and features.gaze_direction not in ("CENTER", "UNKNOWN"),
            "phone": features.phone_detected,
            "camera_problem": bool(features.frame_quality and features.frame_quality.is_problematic),
        }
        durations = {}
        for name, condition in conditions.items():
            if condition and self.starts[name] is None: self.starts[name] = features.timestamp
            if not condition: self.starts[name] = None
            durations[name] = features.timestamp - self.starts[name] if self.starts[name] is not None else 0.0
        return TemporalState(
            face_absent_duration=durations["face_absent"], multiple_face_duration=durations["multiple_face"],
            head_turn_duration=durations["head_turn"], gaze_deviation_duration=durations["gaze_deviation"],
            phone_duration=durations["phone"], camera_problem_duration=durations["camera_problem"],
            **{f"{name}_active": value for name, value in conditions.items()})

    def reset(self) -> None:
        self.history.clear(); self.last_time = None
        for key in self.starts: self.starts[key] = None

    def gaze_center_percentage(self) -> float | None:
        """Return the CENTER share of valid gaze samples in the rolling window."""
        valid = [item for item in self.history if item.gaze_direction != "UNKNOWN"]
        if not valid:
            return None
        return 100.0 * sum(item.gaze_direction == "CENTER" for item in valid) / len(valid)
