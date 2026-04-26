"""LLM API configuration with validation and environment variable support.

Provides Pydantic-based configuration for LLM timeouts, retries,
and circuit breaker settings to ensure production-ready LLM integration.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM API configuration with production-ready defaults.

    Supports environment variable overrides via LLM_* prefix.

    Environment Variables:
        LLM_TIMEOUT: Request timeout in seconds (default: 30)
        LLM_MAX_RETRIES: Maximum retry attempts (default: 5)
        LLM_CIRCUIT_FAIL_MAX: Consecutive failures before opening circuit (default: 3)
        LLM_CIRCUIT_RESET_SECONDS: Circuit open duration (default: 60)
    """

    timeout: int = Field(
        default=30,
        ge=1,
        le=600,
        alias="LLM_TIMEOUT",
        description="Request timeout in seconds for LLM API calls",
    )

    max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        alias="LLM_MAX_RETRIES",
        description="Maximum number of retry attempts for transient errors",
    )

    circuit_fail_max: int = Field(
        default=3,
        ge=1,
        le=10,
        alias="LLM_CIRCUIT_FAIL_MAX",
        description="Consecutive failures before circuit breaker opens",
    )

    circuit_reset_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        alias="LLM_CIRCUIT_RESET_SECONDS",
        description="Circuit breaker open duration in seconds",
    )

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Ensure timeout is reasonable (1-600 seconds)."""
        if v < 1 or v > 600:
            raise ValueError("timeout must be between 1 and 600 seconds")
        return v

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        case_sensitive=False,
        extra="allow",
    )


# Default singleton for backward compatibility
_default_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get or create LLMConfig singleton.

    Returns:
        LLMConfig instance with defaults or environment variable overrides.
    """
    global _default_config
    if _default_config is None:
        _default_config = LLMConfig()
    return _default_config
