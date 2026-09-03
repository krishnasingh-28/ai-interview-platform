"""Session-level statistics derived from recorded events and frames."""
from __future__ import annotations

from monitoring.models import IntegrityEvent, SessionMetrics


def calculate_metrics(frame_count: int, elapsed: float, events: list[IntegrityEvent]) -> SessionMetrics:
    """Calculate session-wide metrics from total frame count, elapsed duration, and integrity events."""
    duration = lambda kind: sum(e.duration for e in events if e.event_type == kind)
    return SessionMetrics(
        frame_count=frame_count,
        elapsed_seconds=elapsed,
        average_fps=frame_count / elapsed if elapsed else 0.0,
        face_absent_seconds=duration("FACE_NOT_VISIBLE"),
        gaze_deviation_seconds=duration("SUSTAINED_GAZE_DEVIATION"),
        head_turn_seconds=duration("PROLONGED_HEAD_TURN"),
        phone_detection_count=sum(e.event_type == "POSSIBLE_SECONDARY_DEVICE" for e in events),
        multiple_face_count=sum(e.event_type == "MULTIPLE_FACES" for e in events),
    )
