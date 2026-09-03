"""Backward-compatibility re-export shim for report_generator module.

Deprecated: Import directly from analytics.report_generator or analytics.evidence_builder instead.
"""
from __future__ import annotations

from analytics.evidence_builder import build_session_summary_data
from analytics.report_generator import (
    SYSTEM_ANALYST_PROMPT,
    generate_layman_analysis,
    generate_offline_fallback_report,
)

__all__ = [
    "SYSTEM_ANALYST_PROMPT",
    "build_session_summary_data",
    "generate_layman_analysis",
    "generate_offline_fallback_report",
]
