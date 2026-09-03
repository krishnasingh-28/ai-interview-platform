"""Unit tests for the PDF report generator."""
import pytest
from monitoring.models import IntegrityEvent, SessionMetrics
from utils.pdf_generator import (
    _format_inline_markdown,
    _sanitize_markdown_text,
    generate_pdf_report,
    parse_markdown_to_flowables,
)
from utils.report_generator import generate_offline_fallback_report
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


@pytest.fixture
def sample_metrics() -> SessionMetrics:
    return SessionMetrics(
        frame_count=450,
        elapsed_seconds=45.0,
        average_fps=10.0,
        face_absent_seconds=2.5,
        gaze_deviation_seconds=6.0,
        head_turn_seconds=3.0,
        phone_detection_count=1,
        multiple_face_count=0,
    )


@pytest.fixture
def sample_events() -> list[IntegrityEvent]:
    return [
        IntegrityEvent(
            event_id="ev_phone_1",
            event_type="POSSIBLE_SECONDARY_DEVICE",
            start_time=12.0,
            end_time=15.0,
            duration=3.0,
            confidence=0.88,
        ),
        IntegrityEvent(
            event_id="ev_gaze_1",
            event_type="SUSTAINED_GAZE_DEVIATION",
            start_time=20.0,
            end_time=25.0,
            duration=5.0,
            confidence=0.72,
        ),
    ]


def test_sanitize_and_inline_markdown():
    text = "### 🎯 **Session Overview** with `code` & *italics* <tag>"
    sanitized = _sanitize_markdown_text(text)
    assert "[Overview]" in sanitized
    assert "🎯" not in sanitized

    formatted = _format_inline_markdown(text)
    assert "<b>Session Overview</b>" in formatted
    assert "<font face=\"Courier\"" in formatted
    assert "&lt;tag&gt;" in formatted


def test_generate_pdf_report_with_metrics_and_events(sample_metrics, sample_events):
    summary_md = generate_offline_fallback_report(sample_metrics, sample_events)
    pdf_bytes = generate_pdf_report(
        summary_markdown=summary_md,
        metrics=sample_metrics,
        events=sample_events,
        model_used="google/gemini-2.0-flash-001",
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic header
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_report_empty_events(sample_metrics):
    summary_md = "### 🎯 Clean Session\n- No anomalies detected."
    pdf_bytes = generate_pdf_report(
        summary_markdown=summary_md,
        metrics=sample_metrics,
        events=[],
        model_used="Offline Engine",
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_pdf_report_minimal():
    summary_md = "Just a plain text summary without metrics."
    pdf_bytes = generate_pdf_report(
        summary_markdown=summary_md,
        metrics=None,
        events=None,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")
