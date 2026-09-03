"""Evidence builder module for compiling raw vision metrics and events into analytical payload structures."""
from __future__ import annotations

from typing import Any
from monitoring.models import IntegrityEvent, SessionMetrics
from utils.time_utils import format_seconds
from analytics.schemas import AnalyticsPayload, SessionSummaryMetrics


def build_session_summary_data(metrics: SessionMetrics, events: list[IntegrityEvent]) -> dict[str, Any]:
    """Structure raw metrics and events into an analytical JSON payload dictionary for the LLM."""
    event_breakdown: dict[str, dict[str, Any]] = {}
    for ev in events:
        kind = ev.event_type
        if kind not in event_breakdown:
            event_breakdown[kind] = {"count": 0, "total_duration_seconds": 0.0, "episodes": []}
        duration = max(ev.duration, (ev.end_time - ev.start_time) if ev.end_time else ev.duration)
        event_breakdown[kind]["count"] += 1
        event_breakdown[kind]["total_duration_seconds"] += duration
        event_breakdown[kind]["episodes"].append({
            "start_time_seconds": round(ev.start_time, 1),
            "duration_seconds": round(duration, 1),
            "confidence": round(ev.confidence, 2) if ev.confidence is not None else None,
        })

    # Calculate attention percentages
    elapsed = max(metrics.elapsed_seconds, 0.1)
    face_absent_pct = min(100.0, (metrics.face_absent_seconds / elapsed) * 100.0)
    gaze_away_pct = min(100.0, (metrics.gaze_deviation_seconds / elapsed) * 100.0)
    head_turn_pct = min(100.0, (metrics.head_turn_seconds / elapsed) * 100.0)

    summary_metrics = SessionSummaryMetrics(
        total_duration_formatted=format_seconds(metrics.elapsed_seconds),
        total_duration_seconds=metrics.elapsed_seconds,
        total_frames_processed=metrics.frame_count,
        average_fps=metrics.average_fps,
        total_flagged_events=len(events),
    )

    payload = AnalyticsPayload(
        session_summary=summary_metrics,
        attention_metrics={
            "face_absence_time": f"{metrics.face_absent_seconds:.1f}s ({face_absent_pct:.1f}% of interview)",
            "gaze_off_screen_time": f"{metrics.gaze_deviation_seconds:.1f}s ({gaze_away_pct:.1f}% of interview)",
            "head_turned_away_time": f"{metrics.head_turn_seconds:.1f}s ({head_turn_pct:.1f}% of interview)",
        },
        device_and_person_metrics={
            "phone_detected_episodes": metrics.phone_detection_count,
            "multiple_people_episodes": metrics.multiple_face_count,
        },
        recorded_events=event_breakdown,
    )

    return payload.to_dict()
