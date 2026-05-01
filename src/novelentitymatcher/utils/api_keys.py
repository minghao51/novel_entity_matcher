import os

PROVIDER_ENV_MAP: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_api_key(provider: str) -> str | None:
    env_var = PROVIDER_ENV_MAP.get(provider)
    if env_var is None:
        return None
    return os.getenv(env_var)


def get_all_api_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    for provider, env_var in PROVIDER_ENV_MAP.items():
        key = os.getenv(env_var)
        if key:
            keys[provider] = key
    return keys


def provider_to_env_var(provider: str) -> str | None:
    return PROVIDER_ENV_MAP.get(provider)
