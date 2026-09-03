"""Analytics boundary for evidence building, schemas, OpenRouter AI client, and report generation."""
from __future__ import annotations

from analytics.evidence_builder import build_session_summary_data
from analytics.openrouter_client import OpenRouterClient, load_env_api_key
from analytics.report_generator import generate_layman_analysis, generate_offline_fallback_report
from analytics.schemas import AnalyticsPayload, EpisodeSummary, SessionSummaryMetrics

__all__ = [
    "AnalyticsPayload",
    "EpisodeSummary",
    "SessionSummaryMetrics",
    "build_session_summary_data",
    "OpenRouterClient",
    "load_env_api_key",
    "generate_layman_analysis",
    "generate_offline_fallback_report",
]
