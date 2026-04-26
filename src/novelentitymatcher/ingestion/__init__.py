"""Data ingestion package for semantic matcher datasets."""

from .currencies import run as run_currencies
from .industries import run as run_industries
from .languages import run as run_languages
from .occupations import run as run_occupations
from .products import run as run_products
from .timezones import run as run_timezones
from .universities import run as run_universities

__all__ = [
    "run_currencies",
    "run_industries",
    "run_languages",
    "run_occupations",
    "run_products",
    "run_timezones",
    "run_universities",
]
