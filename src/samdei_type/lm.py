"""Compact Khmer language models for n-best reranking.

CharLM: hashed character n-gram counts (numpy), stupid backoff, on space-stripped text.
WordLM: unigram/bigram word counts from khmercut segmentation, stupid backoff.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

BOS_CH = "\x02"
EOS_CH = "\x03"


class CharLM:
    def __init__(self, order: int = 7, bits: int = 23):
        self.order, self.bits = order, bits
        self.size = 1 << bits
        self.mask = np.uint64(self.size - 1)
        self.counts = [np.zeros(self.size, dtype=np.int32) for _ in range(order + 1)]  # index = n-gram length
        self.total = 0
        self.alpha = 0.4

    @staticmethod
    def _codes(text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)

    def fit(self, lines, chunk_chars: int = 20_000_000) -> "CharLM":
        buf, n = [], 0
        for line in lines:
            s = BOS_CH + "".join(line.split()) + EOS_CH
            buf.append(s); n += len(s)
            if n >= chunk_chars:
                self._add("".join(buf)); buf, n = [], 0
        if buf:
            self._add("".join(buf))
        return self

    def _add(self, text: str) -> None:
        c = self._codes(text)
        self.total += len(c)
        P = np.uint64(1099511628211)
        # per-order polynomial hash of each n-gram (must match _count)
        for n in range(1, self.order + 1):
            if len(c) < n:
                break
            hv = np.zeros(len(c) - n + 1, dtype=np.uint64)
            for k in range(n):
                hv = hv * P + c[k: len(c) - n + 1 + k] + np.uint64(n)
            idx = (hv ^ (hv >> np.uint64(29))) & self.mask
            self.counts[n] += np.bincount(idx.astype(np.int64), minlength=self.size).astype(np.int32)

    def _count(self, gram: str) -> int:
        P = 1099511628211
        hv = 0
        n = len(gram)
        for ch in gram:
            hv = (hv * P + ord(ch) + n) & 0xFFFFFFFFFFFFFFFF
        idx = (hv ^ (hv >> 29)) & (self.size - 1)
        return int(self.counts[n][idx])

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
            pen += math.log(self.alpha)
            hist = hist[1:]
        num = self._count(ch)
        return pen + math.log((num + 1) / (self.total + 200))

    def save(self, path: str) -> None:
        np.savez_compressed(path, *self.counts[1:], meta=np.array([self.order, self.bits, self.total]))

    @classmethod
    def load(cls, path: str) -> "CharLM":
        z = np.load(path)
        order, bits, total = [int(x) for x in z["meta"]]
        lm = cls(order, bits)
        lm.counts = [np.zeros(1, dtype=np.int32)] + [z[f"arr_{i}"] for i in range(order)]
        lm.total = total
        return lm


class WordLM:
    def __init__(self):
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.total = 0

    def fit(self, lines) -> "WordLM":
        from khmercut import tokenize
        for line in lines:
            toks = ["<s>"] + [t for t in tokenize(line) if not t.isspace()] + ["</s>"]
            self.uni.update(toks); self.bi.update(zip(toks, toks[1:]))
        self.total = sum(self.uni.values())
        return self

    def prune(self, min_uni: int = 2, min_bi: int = 2) -> "WordLM":
        self.uni = Counter({k: v for k, v in self.uni.items() if v >= min_uni})
        self.bi = Counter({k: v for k, v in self.bi.items() if v >= min_bi})
        return self

    def logprob(self, text: str) -> tuple[float, int]:
        """Returns (logprob, n_oov)."""
        from khmercut import tokenize
        toks = ["<s>"] + [t for t in tokenize(text) if not t.isspace()] + ["</s>"]
        lp, oov = 0.0, 0
        V = len(self.uni) + 1
        for a, b in zip(toks, toks[1:]):
            nb = self.bi.get((a, b), 0)
            if nb and self.uni.get(a):
                lp += math.log(nb / self.uni[a])
            else:
                nu = self.uni.get(b, 0)
                if nu == 0:
                    oov += 1
                lp += math.log(0.4) + math.log((nu + 0.1) / (self.total + 0.1 * V))
        return lp, oov

    def save(self, path: str) -> None:
        """A path ending in .json or .json.gz writes the plain JSON format (see save_json); anything else a pickle."""
        if str(path).endswith((".json", ".json.gz")):
            return self.save_json(path)
        import pickle
        with open(path, "wb") as f:
            pickle.dump((dict(self.uni), dict(self.bi), self.total), f)

    FORMAT = "samdei-type-wordlm"

    def save_json(self, path: str) -> None:
        """Plain JSON, no pickle (the released format):
        {"format": "samdei-type-wordlm", "version": 1, "total": N, "unigrams": {word: count},
         "bigrams": [[word1, word2, count], ...]}. Gzipped, with a fixed gzip header so the bytes are
        reproducible, when the path ends in .gz. Words are khmercut tokens and may contain any character."""
        import gzip
        import json
        obj = {"format": self.FORMAT, "version": 1, "total": self.total, "unigrams": dict(self.uni),
               "bigrams": [[a, b, c] for (a, b), c in self.bi.items()]}
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as raw:
            if str(path).endswith(".gz"):
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                    f.write(data)
            else:
                raw.write(data)

    @classmethod
    def load(cls, path: str) -> "WordLM":
        """Reads the plain JSON format (.json / .json.gz, as released) or a pickle (.pkl: load only files you made)."""
        lm = cls()
        if str(path).endswith((".json", ".json.gz")):
            import gzip
            import json
            with (gzip.open if str(path).endswith(".gz") else open)(path, "rb") as f:
                obj = json.loads(f.read().decode("utf-8"))
            if obj.get("format") != cls.FORMAT:
                raise ValueError(f"{path}: not a Samdei Type word LM file")
            lm.total = obj["total"]
            lm.uni = Counter(obj["unigrams"])
            lm.bi = Counter({(a, b): c for a, b, c in obj["bigrams"]})
            return lm
        import pickle
        with open(path, "rb") as f:
            uni, bi, total = pickle.load(f)
        lm.uni, lm.bi, lm.total = Counter(uni), Counter(bi), total
        return lm
