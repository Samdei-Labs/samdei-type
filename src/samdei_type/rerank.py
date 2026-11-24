"""n-best reranking. Each hypothesis y of an input x is scored as

    log P(y|x) + char * log P_char(y) + word * log P_word(y) - oov * #OOV(y) + channel * log P_ch(x|y)

with the weights of a released model's rerank.json ("rerank_weights"). P_ch is the channel model, which reads
Khmer and predicts the romanized input: a separate model, or a two-way model's own Khmer -> Latin direction.
"""
from .evaluation import seq_logprob
from .lm import CharLM, WordLM
from .model import channel_view, load


class Reranker:
    def __init__(self, weights: dict[str, float], char_lm: str | None = None, word_lm: str | None = None,
                 channel=None, device: str = "cpu"):
        """char_lm, word_lm: LM files; channel: the channel model's folder, or a loaded (model, source vocab,
        target vocab). Each is loaded only if its weight is not zero."""
        self.w = {"char": 0.0, "word": 0.0, "oov": 0.0, "channel": 0.0} | weights
        self.clm = CharLM.load(char_lm) if char_lm and self.w["char"] else None
        self.wlm = WordLM.load(word_lm) if word_lm and (self.w["word"] or self.w["oov"]) else None
        self.ch = None
        if channel is not None and self.w["channel"]:
            self.ch = channel_view(*(channel if isinstance(channel, tuple) else load(channel, device)))

    @property
    def active(self) -> bool:
        return bool(self.clm or self.wlm or self.ch)

    def rerank(self, src: str, hyps: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """hyps: [(text, model log probability)] for input src. Returns [(text, total score)], best first."""
        if not hyps or not self.active:
            return list(hyps)
        texts = [h for h, _ in hyps]
        ch = seq_logprob(*self.ch, texts, [src] * len(hyps)) if self.ch else [0.0] * len(hyps)
        w, out = self.w, []
        for (h, lp), c in zip(hyps, ch):
            cl = self.clm.logprob(h) if self.clm else 0.0
            wl, oov = self.wlm.logprob(h) if self.wlm else (0.0, 0)
            out.append((h, lp + w["char"] * cl + w["word"] * wl - w["oov"] * oov + w["channel"] * c))
        return sorted(out, key=lambda x: -x[1])
