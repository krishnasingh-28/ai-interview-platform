"""Backward-compatibility re-export shim for calculate_metrics.

Deprecated: Import calculate_metrics directly from monitoring.metrics instead.
"""
from __future__ import annotations

from monitoring.metrics import calculate_metrics

__all__ = ["calculate_metrics"]
