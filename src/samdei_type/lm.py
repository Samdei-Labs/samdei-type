"""The language models of the reranker, read from a released model's lm/ folder.

CharLM: hashed character n-gram counts on text without spaces, with stupid backoff (.npz).
WordLM: unigram and bigram counts of khmercut words, with stupid backoff (.json.gz).
"""
import gzip
import json
import math

import numpy as np
from khmercut import tokenize

BOS_CH, EOS_CH = "\x02", "\x03"
BACKOFF = 0.4
_HASH_MUL = 1099511628211
_U64 = (1 << 64) - 1


class CharLM:
    def __init__(self, counts: list[np.ndarray], total: int):
        self.counts = counts  # counts[n - 1]: hashed counts of the n-grams, n = 1..order; size a power of two
        self.order = len(counts)
        self.mask = len(counts[0]) - 1
        self.total = total

    @classmethod
    def load(cls, path: str) -> "CharLM":
        """An .npz file with the arrays arr_0 .. arr_{order-1} and meta = [order, hash bits, total]."""
        with np.load(path) as z:
            order, _, total = (int(x) for x in z["meta"])
            return cls([z[f"arr_{i}"] for i in range(order)], total)

    def _index(self, gram: str) -> int:
        hv, n = 0, len(gram)
        for ch in gram:
            hv = (hv * _HASH_MUL + ord(ch) + n) & _U64
        return (hv ^ (hv >> 29)) & self.mask

    def _count(self, gram: str) -> int:
        return int(self.counts[len(gram) - 1][self._index(gram)])

    def logprob(self, text: str) -> float:
        s = BOS_CH + "".join(text.split()) + EOS_CH
        lp = 0.0
        for i in range(1, len(s)):
            lp += self._cond(s[max(0, i - self.order + 1): i], s[i])
        return lp

    def _cond(self, hist: str, ch: str) -> float:
        pen = 0.0
        while hist:
            num = self._count(hist + ch)
            if num > 0:
                den = self._count(hist)
                if den > 0:
                    return pen + math.log(num / den)
            pen += math.log(BACKOFF)
            hist = hist[1:]
        return pen + math.log((self._count(ch) + 1) / (self.total + 200))


class WordLM:
    FORMAT = "samdei-type-wordlm"

    def __init__(self, unigrams: dict[str, int], bigrams: dict[tuple[str, str], int], total: int):
        self.uni, self.bi, self.total = unigrams, bigrams, total

    @classmethod
    def load(cls, path: str) -> "WordLM":
        """Gzipped JSON: {"format": "samdei-type-wordlm", "version": 1, "total": N, "unigrams": {word: count},
        "bigrams": [[word1, word2, count], ...]}."""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("format") != cls.FORMAT:
            raise ValueError(f"{path}: not a Samdei Type word LM file")
        return cls(obj["unigrams"], {(a, b): c for a, b, c in obj["bigrams"]}, obj["total"])

    def logprob(self, text: str) -> tuple[float, int]:
        """(log probability, number of out-of-vocabulary words)."""
        toks = ["<s>"] + [t for t in tokenize(text) if not t.isspace()] + ["</s>"]
        lp, oov = 0.0, 0
        V = len(self.uni) + 1
        for a, b in zip(toks, toks[1:]):
            nb = self.bi.get((a, b), 0)
            if nb and self.uni.get(a):
                lp += math.log(nb / self.uni[a])
            else:
                nu = self.uni.get(b, 0)
                oov += nu == 0
                lp += math.log(BACKOFF) + math.log((nu + 0.1) / (self.total + 0.1 * V))
        return lp, oov
