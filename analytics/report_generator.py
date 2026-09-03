"""Generates plain-English (layman's terms) analytical session reports using OpenRouter."""
from __future__ import annotations

import json
from typing import Any
from monitoring.models import IntegrityEvent, SessionMetrics
from analytics.evidence_builder import build_session_summary_data
from analytics.openrouter_client import OpenRouterClient
from utils.time_utils import format_seconds

SYSTEM_ANALYST_PROMPT = """You are an expert AI Interview Integrity Analyst. Your role is to translate computer vision metrics, head pose angles, gaze tracking data, and object detection flags into clear, intuitive, and balanced plain-English (layman's terms) for HR interviewers and hiring managers.

Guidelines:
1. Speak in accessible, non-technical everyday language. Avoid jargon like 'solvePnP', 'bounding box', 'yaw/pitch degrees', or 'confidence threshold' unless explained simply.
2. Maintain an ethical, fair, and objective tone. Clearly distinguish between normal human behaviors (e.g., looking up to think, adjusting posture, natural blinking) and sustained anomalies.
3. Explicitly reinforce that visual camera signals are purely observational cues and do NOT determine cheating or dishonesty.
4. Format your analysis using clean Markdown with distinct sections:
   - 🎯 **Session Overview & Quick Verdict**: A concise 2-3 sentence summary of how the session went in plain terms.
   - 👁️ **Focus & Visual Attention**: Clear breakdown of where the candidate's attention was directed (eye contact, looking away, thinking patterns).
   - 📱 **Surroundings & Device Activity**: Whether any phones, extra people, or background interruptions were spotted.
   - 💡 **Interviewer Guidance & Notes**: Practical advice for the human reviewer (e.g., moments to review or questions to clarify).
   - 🛡️ **Important Context & Fairness Note**: Brief reminder that technical limitations and natural body language must be considered.
"""


def generate_layman_analysis(
    metrics: SessionMetrics,
    events: list[IntegrityEvent],
    client: OpenRouterClient,
) -> dict[str, Any]:
    """Send formatted session statistics to OpenRouter and return the layman analytical summary."""
    summary_data = build_session_summary_data(metrics, events)

    user_prompt = (
        "Here are the recorded session statistics and vision metrics from the completed interview session:\n\n"
        f"```json\n{json.dumps(summary_data, indent=2)}\n```\n\n"
        "Please analyze these statistics and generate a comprehensive, easy-to-understand analytical overview in layman's terms."
    )

    result = client.generate_completion(
        system_prompt=SYSTEM_ANALYST_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=1500,
    )

    return result


def generate_offline_fallback_report(metrics: SessionMetrics, events: list[IntegrityEvent]) -> str:
    """Provide an immediate, rule-based plain-English summary if OpenRouter API is not configured."""
    elapsed = max(metrics.elapsed_seconds, 0.1)
    duration_str = format_seconds(metrics.elapsed_seconds)
    total_events = len(events)

    has_phone = metrics.phone_detection_count > 0
    has_multiple_faces = metrics.multiple_face_count > 0
    face_absent_pct = (metrics.face_absent_seconds / elapsed) * 100.0
    gaze_away_pct = (metrics.gaze_deviation_seconds / elapsed) * 100.0

    # Determine overall status
    if has_phone or has_multiple_faces:
        status_verdict = "⚠️ **Notable events detected** — The system observed potential secondary devices or additional persons that warrant human review."
    elif total_events == 0:
        status_verdict = "✅ **Clean & consistent session** — The candidate maintained steady screen presence throughout the interview with no sustained anomalies."
    else:
        status_verdict = "ℹ️ **Standard session with minor observations** — A few brief visual deviations were noted, which are often typical of natural thinking or shifting posture."

    lines = [
        f"### 🎯 Session Overview & Quick Verdict",
        f"{status_verdict}",
        f"",
        f"- **Interview Duration**: {duration_str} ({metrics.frame_count} frames analyzed)",
        f"- **Total Flagged Episodes**: {total_events}",
        f"",
        f"### 👁️ Focus & Visual Attention",
        f"- **Face Presence**: Visible for {100.0 - face_absent_pct:.1f}% of the time ({metrics.face_absent_seconds:.1f}s absent).",
        f"- **Eye Contact & Gaze**: Looked away from center for approx. {metrics.gaze_deviation_seconds:.1f}s ({gaze_away_pct:.1f}% of session).",
        f"- **Head Posture**: Prolonged head-turns totaled {metrics.head_turn_seconds:.1f}s.",
        f"",
        f"### 📱 Surroundings & Device Activity",
        f"- **Secondary Devices (Phone)**: {metrics.phone_detection_count} episode(s) detected.",
        f"- **Additional People**: {metrics.multiple_face_count} episode(s) with multiple faces visible.",
        f"",
        f"### 💡 Interviewer Guidance",
        f"- Natural thinking often involves glancing upward or sideways while formulating answers.",
        f"- If phone or multiple-face episodes were noted, cross-reference with the question timestamps during manual review.",
        f"",
        f"### 🛡️ Fairness & Ethics Note",
        f"*These automated signals reflect observable camera patterns only and do not establish intent or dishonesty. Always apply human context and judgment.*",
    ]

    return "\n".join(lines)
