from novelentitymatcher.core.normalizer import TextNormalizer


class TestTextNormalizer:
    """Tests for TextNormalizer - text normalization utilities."""

    def test_normalizer_init_defaults(self):
        normalizer = TextNormalizer()
        assert normalizer.lowercase is True
        assert normalizer.remove_accents is False
        assert normalizer.remove_punctuation is False

    def test_normalizer_init_custom(self):
        normalizer = TextNormalizer(
            lowercase=False, remove_accents=True, remove_punctuation=True
        )
        assert normalizer.lowercase is False
        assert normalizer.remove_accents is True
        assert normalizer.remove_punctuation is True

    def test_normalizer_lowercase(self):
        normalizer = TextNormalizer(lowercase=True)
        assert normalizer.normalize("HELLO WORLD") == "hello world"

    def test_normalizer_preserve_case(self):
        normalizer = TextNormalizer(lowercase=False)
        assert normalizer.normalize("Hello World") == "Hello World"

    def test_normalizer_remove_accents(self):
        normalizer = TextNormalizer(remove_accents=True)
        assert normalizer.normalize("café") == "cafe"

    def test_normalizer_remove_punctuation(self):
        normalizer = TextNormalizer(lowercase=False, remove_punctuation=True)
        assert normalizer.normalize("Hello, World!") == "Hello World"

    def test_normalizer_combined(self):
        normalizer = TextNormalizer(
            lowercase=True, remove_accents=True, remove_punctuation=True
        )
        assert normalizer.normalize("HELLO, World!") == "hello world"

    def test_normalizer_strip_whitespace(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("  hello   world  ") == "hello world"

    def test_normalizer_empty_string(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""

    def test_normalizer_unicode_normalization(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize("Ångström")
        assert "a" in result
        assert "ngstr" in result

    def test_normalize_entity_name(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize_entity_name("  Deutschland  ") == "deutschland"

    def test_normalize_entity_name_full(self):
        normalizer = TextNormalizer(
            lowercase=True, remove_accents=True, remove_punctuation=True
        )
        assert (
            normalizer.normalize_entity_name("United States (USA)")
            == "united states usa"
        )
