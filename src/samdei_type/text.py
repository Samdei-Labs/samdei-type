"""Khmer text normalization."""
import re
import unicodedata

from .khnormal import khnormal

_ASCII_DIGITS = {0x17E0 + i: str(i) for i in range(10)}  # Khmer digits ០-៩


def normalize_khmer(text: str) -> str:
    """The canonical form of Khmer text: NFC, deprecated and invisible characters replaced or removed,
    SIL khnormal character order, ASCII digits, single spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u17a3", "\u17a2").replace("\u17a4", "\u17a2\u17b6")  # deprecated independent vowels
    for ch in "\u17b4\u17b5\u200b\u2060\ufeff":  # inherent-vowel signs, zero-width space, word joiner, BOM
        text = text.replace(ch, "")
    text = khnormal(text).translate(_ASCII_DIGITS)
    return re.sub(r"[ \t\u00a0\u3000]+", " ", text).strip()
