"""Text normalization for factorized tokenization.

Provides functions to normalize text values and create text tokens.
Normalization ensures consistent tokenization across different text formats.
"""

import re

from .tokens import text_token


def normalize_text(text: str) -> str | None:
    """Normalize text value for tokenization.

    Normalization steps:
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Replace internal whitespace with underscores
    4. Remove non-alphanumeric characters (except underscore)

    Args:
        text: Raw text value

    Returns:
        Normalized string, or None if result is empty
    """
    if not text:
        return None

    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w_]", "", normalized)

    return normalized if normalized else None


def make_text_token(text: str) -> str | None:
    """Create text token from raw text value.

    Args:
        text: Raw text value

    Returns:
        Token string like "TXT:oral", or None if text normalizes to empty
    """
    normalized = normalize_text(text)
    if normalized is None:
        return None
    return text_token(normalized)
