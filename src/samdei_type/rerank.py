"""n-best reranking: model logprob + a*charLM + b*wordLM - c*OOV + d*channel log P(src | hyp).

Weights come from the rerank.json of a released model (key "rerank_weights"; see load_weights).
"""
from __future__ import annotations

import json
from pathlib import Path

WEIGHT_KEYS = ("char", "word", "oov", "channel")


def load_weights(eval_json: str | Path, key: str = "rerank_weights") -> dict[str, float]:
    w = dict.fromkeys(WEIGHT_KEYS, 0.0)
    p = Path(eval_json)
    if p.exists():
        w.update(json.loads(p.read_text()).get(key, {}))
    return w


class Reranker:
    def __init__(self, weights: dict[str, float], charlm: str | None = None, wordlm: str | None = None,
                 channel: str | None = None, device: str = "cpu"):
        self.w = dict.fromkeys(WEIGHT_KEYS, 0.0) | weights
        self.device = device
        self.clm = self.wlm = self.ch = None
        if self.w["char"] and charlm and Path(charlm).exists():
            from .lm import CharLM
            self.clm = CharLM.load(charlm)
        if (self.w["word"] or self.w["oov"]) and wordlm and Path(wordlm).exists():
            from .lm import WordLM
            self.wlm = WordLM.load(wordlm)
        if self.w["channel"] and channel and Path(channel).exists():
            from .model import load
            self.ch = load(channel, device)[:3]

    @property
    def active(self) -> bool:
        return bool(self.clm or self.wlm or self.ch)

    def rerank(self, src: str, hyps: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """hyps: [(text, model logprob)]. Returns [(text, total score)] best first."""
        if not hyps or not self.active:
            return list(hyps)
        if self.ch:
            from .evaluation import seq_logprob
            chl = seq_logprob(*self.ch, [h for h, _ in hyps], [src] * len(hyps), self.device)
        else:
            chl = [0.0] * len(hyps)
        w, out = self.w, []
        for (h, lp), c in zip(hyps, chl):
            cl = self.clm.logprob(h) if self.clm else 0.0
            wl, oov = self.wlm.logprob(h) if self.wlm else (0.0, 0)
            out.append((h, lp + w["char"] * cl + w["word"] * wl - w["oov"] * oov + w["channel"] * c))
        return sorted(out, key=lambda x: -x[1])
