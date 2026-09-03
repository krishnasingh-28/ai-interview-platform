"""OpenRouter API client for generating AI analytical summaries."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import requests

from config import DEFAULT_OPENROUTER_MODEL, OPENROUTER_API_URL


def load_env_api_key() -> str:
    """Read OPENROUTER_API_KEY from environment or .env file."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key

    # Check local .env file
    env_paths = [
        Path(".env"),
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        if key:
                            return key
            except Exception:
                pass
    return ""


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENROUTER_MODEL,
        api_url: str = OPENROUTER_API_URL,
        timeout: float = 35.0,
    ) -> None:
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = load_env_api_key().strip()
        self.model = model or DEFAULT_OPENROUTER_MODEL
        self.api_url = api_url
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is available."""
        return bool(self.api_key)

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        """Send a chat completion request to OpenRouter.

        Returns a dictionary with 'success', 'content' (if successful), or 'error'.
        """
        if not self.is_configured:
            return {
                "success": False,
                "error": "OpenRouter API key is missing. Please provide a key in the sidebar or .env file.",
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-interview-integrity-monitor",
            "X-Title": "AI Interview Integrity Monitor",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    usage = data.get("usage", {})
                    return {
                        "success": True,
                        "content": content,
                        "model": data.get("model", self.model),
                        "usage": usage,
                    }
                return {
                    "success": False,
                    "error": "Received empty or unexpected response format from OpenRouter.",
                }

            # Handle error status codes
            try:
                err_data = response.json()
                err_msg = err_data.get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text or f"HTTP {response.status_code}"

            return {
                "success": False,
                "error": f"OpenRouter API error (HTTP {response.status_code}): {err_msg}",
                "status_code": response.status_code,
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Request to OpenRouter timed out after {self.timeout} seconds.",
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error connecting to OpenRouter: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during OpenRouter call: {e}",
            }
