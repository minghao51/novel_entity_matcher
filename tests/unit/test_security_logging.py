from novelentitymatcher.exceptions import _redact_api_keys


class TestRedactApiKeys:
    def test_redacts_openai_key(self):
        result = _redact_api_keys("api key is sk-abc123def456ghi789jkl012")
        assert "sk-abc123def456ghi789jkl012" not in result
        assert "...REDACTED..." in result

    def test_redacts_anthropic_key(self):
        result = _redact_api_keys("key=sk-ant-api03-abcdefghijklmnop")
        assert "sk-ant" not in result

    def test_redacts_openrouter_key(self):
        result = _redact_api_keys("Bearer sk-or-v1-abcdefghijklmnopqrstuv")
        assert "sk-or-v1" not in result

    def test_redacts_huggingface_key(self):
        result = _redact_api_keys("token hf_abcdefghijklmnop")
        assert "hf_" not in result

    def test_keeps_regular_text_unchanged(self):
        text = "This is a normal error message with no API keys"
        result = _redact_api_keys(text)
        assert result == text

    def test_handles_multiple_keys_in_single_string(self):
        text = "First key: sk-abc123def456ghi789jkl012 and second key: sk-ant-def456"
        result = _redact_api_keys(text)
        assert result.count("...REDACTED...") == 2

    def test_handles_empty_string(self):
        assert _redact_api_keys("") == ""

    def test_handles_nearby_text_not_mistaken_for_keys(self):
        text = "The word 'skill' and 'sky' should remain"
        result = _redact_api_keys(text)
        assert result == text

    def test_redacts_google_ai_key(self):
        result = _redact_api_keys("key=AIzaSyA1234567890abcdefghijklmnopqrstuvwx")
        assert "AIza" not in result
        assert "...REDACTED..." in result

    def test_redacts_google_oauth_token(self):
        result = _redact_api_keys("Bearer ya29.a0AfH6SMBx1234567890abcdef")
        assert "ya29" not in result
        assert "...REDACTED..." in result
