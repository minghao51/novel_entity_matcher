from novelentitymatcher.utils.preprocessing import (
    tokenize,
    remove_stopwords,
    lemmatize,
    clean_text,
    extract_aliases,
)


class TestPreprocessing:
    """Tests for text preprocessing utilities."""

    def test_tokenize(self):
        tokens = tokenize("Hello world!")
        assert len(tokens) == 2
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_preserve_case(self):
        tokens = tokenize("Hello World", lowercase=False)
        assert "Hello" in tokens
        assert "World" in tokens

    def test_remove_stopwords(self):
        tokens = ["the", "quick", "brown", "fox"]
        filtered = remove_stopwords(tokens)
        assert "the" not in filtered
        assert "quick" in filtered

    def test_lemmatize(self):
        lemma = lemmatize("running")
        assert "run" in lemma.lower()

    def test_clean_text(self):
        cleaned = clean_text("Hello,   World!  ")
        assert cleaned == "hello world"

    def test_clean_text_preserve_punctuation(self):
        cleaned = clean_text("Hello, World!", remove_punct=False)
        assert "," in cleaned

    def test_extract_aliases(self):
        text = "USA (United States), UK (United Kingdom)"
        aliases = extract_aliases(text)
        assert "USA" in aliases
        assert "UK" in aliases
