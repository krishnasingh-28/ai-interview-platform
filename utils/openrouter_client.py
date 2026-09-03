"""Backward-compatibility re-export shim for OpenRouterClient.

Deprecated: Import directly from analytics.openrouter_client instead.
"""
from __future__ import annotations

from analytics.openrouter_client import OpenRouterClient, load_env_api_key

__all__ = ["OpenRouterClient", "load_env_api_key"]
