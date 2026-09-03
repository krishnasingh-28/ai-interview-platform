"""Data models and contract schemas for analytics input payloads and AI report outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeSummary:
    """Summary of a single flagged integrity episode."""
    start_time_seconds: float
    duration_seconds: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time_seconds": round(self.start_time_seconds, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "confidence": round(self.confidence, 2) if self.confidence is not None else None,
        }


@dataclass
class SessionSummaryMetrics:
    """Session high-level summary metrics."""
    total_duration_formatted: str
    total_duration_seconds: float
    total_frames_processed: int
    average_fps: float
    total_flagged_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration_formatted": self.total_duration_formatted,
            "total_duration_seconds": round(self.total_duration_seconds, 1),
            "total_frames_processed": self.total_frames_processed,
            "average_fps": round(self.average_fps, 1),
            "total_flagged_events": self.total_flagged_events,
        }


@dataclass
class AnalyticsPayload:
    """Structured analytics payload passed into LLM or export generators."""
    session_summary: SessionSummaryMetrics
    attention_metrics: dict[str, str]
    device_and_person_metrics: dict[str, int]
    recorded_events: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_summary": self.session_summary.to_dict(),
            "attention_metrics": self.attention_metrics,
            "device_and_person_metrics": self.device_and_person_metrics,
            "recorded_events": self.recorded_events,
        }
