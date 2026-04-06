import re
from typing import List

__all__ = ["tokenize", "remove_stopwords", "lemmatize", "clean_text", "extract_aliases"]

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True
    _lemmatizer = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    _lemmatizer = None


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []

    if lowercase:
        text = text.lower()

    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def remove_stopwords(tokens: List[str], lang: str = "english") -> List[str]:
    """Remove stopwords from token list."""
    if not NLTK_AVAILABLE:
        return tokens

    try:
        stop_words = set(stopwords.words(lang))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words(lang))

    return [t for t in tokens if t.lower() not in stop_words]


def lemmatize(word: str) -> str:
    """Lemmatize a word to its base form."""
    if _lemmatizer is None:
        return word

    try:
        return _lemmatizer.lemmatize(word)
    except LookupError:
        return word


def clean_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Clean text by removing extra whitespace and optionally punctuation."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    if remove_punct:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_aliases(text: str) -> List[str]:
    """Extract potential aliases from text (abbreviations in parentheses)."""
    pattern = r"\b([A-Z]{2,})\s*\([^)]+\)|([A-Z]{2,})"
    matches = re.findall(pattern, text)
    aliases = []
    for match in matches:
        for group in match:
            if group:
                aliases.append(group)
    return list(set(aliases))
