import re
import unicodedata


class TextNormalizer:
    """Text normalization utilities for entity matching."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_accents: bool = False,
        remove_punctuation: bool = False,
    ):
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()
        text = unicodedata.normalize("NFKD", text)

        if self.remove_accents:
            text = "".join(c for c in text if not unicodedata.combining(c))

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # Backwards-compatible alias
    normalize_entity_name = normalize

    # Backwards-compatible alias
