"""Metrics (CER, exact match), input chunking and batched decoding."""
from __future__ import annotations

import torch

from .model import EOS, PAD, Seq2Seq, Vocab, beam_cached, greedy_cached
from .text import normalize_khmer, strip_spaces


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def canon(s: str) -> str:
    """Comparison form: normalized, no spaces, lowercase; coeng-DA and coeng-TA are merged
    (they render identically and both spellings are in common use, e.g. ក្ដៅ / ក្តៅ), and so are the two
    independent vowels QOO ឲ and ឱ (both standard in ឲ្យ / ឱ្យ; merged 2026-11-18)."""
    return (strip_spaces(normalize_khmer(s)).lower()
            .replace("\u17d2\u178a", "\u17d2\u178f").replace("\u17b2", "\u17b1"))


def score(hyps: list[str], refs: list[str], nbest: list[list[str]] | None = None) -> dict:
    errs = chars = exact = topk = 0
    for i, (h, r) in enumerate(zip(hyps, refs)):
        hc, rc = canon(h), canon(r)
        errs += edit_distance(hc, rc); chars += max(1, len(rc)); exact += hc == rc
        if nbest is not None:
            topk += any(canon(x) == rc for x in nbest[i])
    n = max(1, len(refs))
    out = {"cer": errs / chars, "exact": exact / n, "n": len(refs)}
    if nbest is not None:
        out["topk"] = topk / n
    return out


def chunk_source(s: str, max_words: int = 8) -> list[str]:
    """Split a romanized message into pieces of at most `max_words` words, also breaking after punctuation.

    The models were trained on short pieces (median 26 characters); long messages are converted piece by piece
    and the outputs are joined (samdei_type.pipeline.Converter and the samdei-type command do this)."""
    out, cur = [], []
    for tok in s.split():
        cur.append(tok)
        if len(cur) >= max_words or tok[-1] in ",.;:?!\u17d4":
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out or [s]


def batch_encode(strs: list[str], vocab: Vocab, device, max_len: int = 250) -> torch.Tensor:
    ids = [vocab.encode(s[:max_len - 1]) for s in strs]
    L = max(len(x) for x in ids)
    t = torch.full((len(ids), L), PAD, dtype=torch.long)
    for i, x in enumerate(ids):
        t[i, :len(x)] = torch.tensor(x)
    return t.to(device)


@torch.no_grad()
def translate(model: Seq2Seq, sv: Vocab, tv: Vocab, srcs: list[str], device, beam: int = 1,
              bs: int = 64, max_len: int = 200) -> tuple[list[str], list[list[tuple[str, float]]]]:
    model.eval()
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    best: list[str] = [""] * len(srcs)
    nbest: list[list[tuple[str, float]]] = [[] for _ in srcs]
    step = bs if beam == 1 else max(1, bs // beam)
    for i in range(0, len(order), step):
        idx = order[i:i + step]
        x = batch_encode([srcs[j] for j in idx], sv, device)
        ml = min(max_len, 2 * x.size(1) + 10)  # target is rarely >2x the romanized length
        if beam == 1:
            outs = greedy_cached(model, x, max_len=ml)  # KV cache: same output as model.greedy, ~3x faster
            for j, o in zip(idx, outs):
                best[j] = tv.decode(o); nbest[j] = [(best[j], 0.0)]
        else:
            outs = beam_cached(model, x, k=beam, max_len=ml)  # same n-best as model.beam, ~5x faster
            for j, hyps in zip(idx, outs):
                nbest[j] = [(tv.decode(h), s) for h, s in hyps]
                best[j] = nbest[j][0][0]
    return best, nbest


@torch.no_grad()
def seq_logprob(model: Seq2Seq, sv: Vocab, tv: Vocab, srcs: list[str], tgts: list[str], device,
                bs: int = 256) -> list[float]:
    """Teacher-forced log P(tgt | src) per pair (sum over target chars incl. EOS)."""
    import torch.nn.functional as F
    from .model import BOS
    model.eval()
    out = [0.0] * len(srcs)
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]) + len(tgts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        x = batch_encode([srcs[j] for j in idx], sv, device)
        ys = [[BOS] + tv.encode(tgts[j][:248]) for j in idx]
        L = max(len(y) for y in ys)
        y = torch.full((len(ys), L), PAD, dtype=torch.long)
        for k, row in enumerate(ys):
            y[k, :len(row)] = torch.tensor(row)
        y = y.to(device)
        logp = F.log_softmax(model(x, y[:, :-1]).float(), -1)
        tok = logp.gather(-1, y[:, 1:, None]).squeeze(-1)
        tok = tok.masked_fill(y[:, 1:] == PAD, 0.0)
        for j, v in zip(idx, tok.sum(-1).tolist()):
            out[j] = v
    return out
