"""Tests for OpenRouter integration, analytics boundary, and layman analytical summary generator."""
from unittest.mock import MagicMock, patch
import pytest

from monitoring.models import IntegrityEvent, SessionMetrics
from analytics.openrouter_client import OpenRouterClient
from analytics.evidence_builder import build_session_summary_data
from analytics.report_generator import (
    generate_layman_analysis,
    generate_offline_fallback_report,
)
from analytics.schemas import AnalyticsPayload, SessionSummaryMetrics


@pytest.fixture
def sample_metrics() -> SessionMetrics:
    return SessionMetrics(
        frame_count=300,
        elapsed_seconds=30.0,
        average_fps=10.0,
        face_absent_seconds=3.0,
        gaze_deviation_seconds=5.0,
        head_turn_seconds=4.0,
        phone_detection_count=1,
        multiple_face_count=0,
    )


@pytest.fixture
def sample_events() -> list[IntegrityEvent]:
    return [
        IntegrityEvent(
            event_id="ev1",
            event_type="POSSIBLE_SECONDARY_DEVICE",
            start_time=10.0,
            end_time=12.0,
            duration=2.0,
            confidence=0.85,
        ),
        IntegrityEvent(
            event_id="ev2",
            event_type="SUSTAINED_GAZE_DEVIATION",
            start_time=15.0,
            end_time=20.0,
            duration=5.0,
            confidence=0.60,
        ),
    ]


def test_build_session_summary_data(sample_metrics, sample_events):
    data = build_session_summary_data(sample_metrics, sample_events)
    assert data["session_summary"]["total_duration_seconds"] == 30.0
    assert data["session_summary"]["total_frames_processed"] == 300
    assert data["session_summary"]["total_flagged_events"] == 2
    assert "POSSIBLE_SECONDARY_DEVICE" in data["recorded_events"]
    assert "SUSTAINED_GAZE_DEVIATION" in data["recorded_events"]
    assert data["device_and_person_metrics"]["phone_detected_episodes"] == 1


def test_generate_offline_fallback_report(sample_metrics, sample_events):
    report = generate_offline_fallback_report(sample_metrics, sample_events)
    assert "Session Overview & Quick Verdict" in report
    assert "Focus & Visual Attention" in report
    assert "Surroundings & Device Activity" in report
    assert "Fairness & Ethics Note" in report
    assert "**Secondary Devices (Phone)**: 1 episode(s)" in report


def test_openrouter_client_unconfigured():
    client = OpenRouterClient(api_key="")
    assert not client.is_configured
    result = client.generate_completion("system", "user")
    assert not result["success"]
    assert "API key is missing" in result["error"]


@patch("analytics.openrouter_client.requests.post")
def test_openrouter_client_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "google/gemini-2.0-flash-001",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "### 🎯 Session Overview\nOverall the candidate performed well with a minor phone alert.",
                }
            }
        ],
        "usage": {"total_tokens": 150},
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="test-key-123", model="google/gemini-2.0-flash-001")
    result = client.generate_completion("system prompt", "user prompt")

    assert result["success"] is True
    assert "Session Overview" in result["content"]
    assert result["model"] == "google/gemini-2.0-flash-001"
    assert result["usage"]["total_tokens"] == 150


@patch("analytics.openrouter_client.requests.post")
def test_openrouter_client_api_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {
        "error": {"message": "Invalid API Key provided"}
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="bad-key", model="google/gemini-2.0-flash-001")
    result = client.generate_completion("system", "user")

    assert result["success"] is False
    assert "Invalid API Key" in result["error"]
    assert result["status_code"] == 401


@patch("analytics.openrouter_client.requests.post")
def test_generate_layman_analysis_end_to_end(mock_post, sample_metrics, sample_events):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "google/gemini-2.0-flash-001",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "### 🎯 Executive Summary\nThe candidate was attentive throughout the session.",
                }
            }
        ],
    }
    mock_post.return_value = mock_response

    client = OpenRouterClient(api_key="valid-key", model="google/gemini-2.0-flash-001")
    res = generate_layman_analysis(sample_metrics, sample_events, client)

    assert res["success"] is True
    assert "Executive Summary" in res["content"]


def test_legacy_utils_reexport_compatibility(sample_metrics, sample_events):
    """Ensure legacy imports from utils.* function seamlessly via re-export shims."""
    from utils.metrics import calculate_metrics as legacy_calculate_metrics
    from utils.openrouter_client import OpenRouterClient as LegacyOpenRouterClient
    from utils.report_generator import build_session_summary_data as legacy_build_summary

    m = legacy_calculate_metrics(100, 10.0, sample_events)
    assert m.frame_count == 100

    c = LegacyOpenRouterClient(api_key="legacy-test")
    assert c.api_key == "legacy-test"

    s = legacy_build_summary(sample_metrics, sample_events)
    assert "session_summary" in s
