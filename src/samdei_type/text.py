"""Khmer text utilities: normalization, character classes, cleaning."""
from __future__ import annotations

import re
import unicodedata

from .khnormal import khnormal

KH_DIGITS = "០១២៣៤៥៦៧៨៩"
_DIGIT_MAP = {ord(k): str(i) for i, k in enumerate(KH_DIGITS)}
ZW = "​‌‍⁠﻿"

CONSONANTS = "".join(chr(c) for c in range(0x1780, 0x17A3))
INDEP_VOWELS = "".join(chr(c) for c in range(0x17A3, 0x17B4))
DEP_VOWELS = "".join(chr(c) for c in range(0x17B6, 0x17C6))
COENG = "្"
SHIFTERS = "៉៊"
# modifiers that can follow a cluster (nikahit, reahmuk, yuukaleapintu, bantoc, robat, toandakhiat, ...)
MODIFIERS = "ំះៈ់៌៍៎៏័៑៓៝"

_kh_re = re.compile(r"[ក-៿᧠-᧿]")
_latin_re = re.compile(r"[A-Za-z]")


def is_khmer(ch: str) -> bool:
    return "ក" <= ch <= "៿" or "᧠" <= ch <= "᧿"


def normalize_khmer(text: str, ascii_digits: bool = True, keep_zwsp: bool = False) -> str:
    """Canonical form for Khmer targets (and corpus text)."""
    text = unicodedata.normalize("NFC", text)
    # deprecated / discouraged characters
    text = text.replace("ឣ", "អ").replace("ឤ", "អា")
    text = text.replace("឴", "").replace("឵", "")
    if not keep_zwsp:
        text = text.replace("​", "")
    text = text.replace("⁠", "").replace("﻿", "")
    text = khnormal(text)
    if ascii_digits:
        text = text.translate(_DIGIT_MAP)
    text = re.sub(r"[ \t 　]+", " ", text).strip()
    return text


def strip_spaces(text: str) -> str:
    return re.sub(r"[\s​]+", "", text)


def khmer_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace() and not c.isdigit() and c not in ZW]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_khmer(c)) / len(letters)


SENT_SPLIT = re.compile(r"(?<=[។៕!?])\s*|\n+")


def split_sentences(doc: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(doc) if s and s.strip()]
