# stt_benchmark/utils/text_normalize.py

"""
Text normalization utilities for consistent metric computation.
"""

import re
import unicodedata
from typing import Optional


class TextNormalizer:
    """Configurable text normalizer for ASR/AST evaluation."""

    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 unicode_form: Optional[str] = None,
                 collapse_whitespace: bool = False):
        """
        Initialize normalizer.

        Args:
            lowercase: Convert text to lowercase.
            remove_punctuation: Remove punctuation marks (keeps in-word
                apostrophes and hyphens).
            unicode_form: Unicode normalization form to apply first, e.g.
                "NFC" or "NFKC". None disables it. NFC reconciles composed vs
                decomposed diacritics WITHOUT removing them, so tone marks are
                preserved.
            collapse_whitespace: Collapse runs of whitespace to a single space
                and strip ends.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.unicode_form = unicode_form
        self.collapse_whitespace = collapse_whitespace

    def normalize(self, text: str) -> str:
        """Apply all configured normalizations, in a fixed order."""
        if not text:
            return ""

        # Unicode normalization first, so downstream regex sees composed chars.
        if self.unicode_form:
            text = unicodedata.normalize(self.unicode_form, text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            # Remove punctuation but keep apostrophes within words and hyphens
            text = re.sub(r"[^\w\s\-']", "", text)
            # Clean up orphan apostrophes/hyphens
            text = re.sub(r"\s['\-]\s", " ", text)

        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def get_config(self) -> dict:
        """Return normalizer configuration."""
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "unicode_form": self.unicode_form,
            "collapse_whitespace": self.collapse_whitespace,
        }

# Default normalizer instance
DEFAULT_NORMALIZER = TextNormalizer(lowercase=True, remove_punctuation=True)
