"""Scoring (CER and exact match, as for Samdei Type Bench), input chunking and batched decoding."""
import re

import torch
import torch.nn.functional as F

from .model import Seq2Seq, beam_cached, greedy_cached
from .text import normalize_khmer
from .vocab import BOS, MAX_INPUT_CHARS, PAD, Vocab


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
    """The form outputs and references are compared in: normalized, without spaces, lowercase, with coeng-DA
    merged into coeng-TA (they render the same; ក្ដៅ / ក្តៅ) and QOO ឲ merged into ឱ (ឲ្យ / ឱ្យ are both standard)."""
    s = re.sub(r"[\s\u200b]+", "", normalize_khmer(s)).lower()
    return s.replace("\u17d2\u178a", "\u17d2\u178f").replace("\u17b2", "\u17b1")


def score(hyps: list[str], refs: list[str]) -> dict:
    """CER (total edit distance / total reference length) and exact match, on canon() forms."""
    errs = chars = exact = 0
    for h, r in zip(hyps, refs, strict=True):
        h, r = canon(h), canon(r)
        errs += edit_distance(h, r)
        chars += max(1, len(r))
        exact += h == r
    return {"cer": errs / max(1, chars), "exact": exact / max(1, len(refs)), "n": len(refs)}


def chunk_source(s: str, max_words: int = 8) -> list[str]:
    """Split a romanized message into pieces of at most max_words words, also breaking after punctuation.

    The models were trained on short pieces (median 26 characters), so long messages are decoded piece by piece
    and the outputs joined with a space (as pipeline.Converter does)."""
    out, cur = [], []
    for tok in s.split():
        cur.append(tok)
        if len(cur) >= max_words or tok[-1] in ",.;:?!\u17d4":
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out or [s]


def _pad(rows: list[list[int]], device) -> torch.Tensor:
    t = torch.full((len(rows), max(len(r) for r in rows)), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        t[i, :len(r)] = torch.tensor(r)
    return t.to(device)


@torch.no_grad()
def translate(model: Seq2Seq, sv: Vocab, tv: Vocab, srcs: list[str], beam: int = 5, bs: int = 64,
              max_len: int = 200) -> list[list[tuple[str, float]]]:
    """Decode srcs in batches of similar length. Returns per input the n-best [(text, log probability)], best
    first; beam=1 is greedy decoding (one hypothesis, score 0)."""
    model.eval()
    device = next(model.parameters()).device
    nbest: list[list[tuple[str, float]]] = [[] for _ in srcs]
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    step = bs if beam == 1 else max(1, bs // beam)
    for i in range(0, len(order), step):
        idx = order[i:i + step]
        x = _pad([sv.encode(srcs[j][:MAX_INPUT_CHARS]) for j in idx], device)
        ml = min(max_len, 2 * x.size(1) + 10)  # outputs are rarely more than twice as long as the input
        if beam == 1:
            for j, ids in zip(idx, greedy_cached(model, x, max_len=ml)):
                nbest[j] = [(tv.decode(ids), 0.0)]
        else:
            for j, hyps in zip(idx, beam_cached(model, x, k=beam, max_len=ml)):
                nbest[j] = [(tv.decode(ids), s) for ids, s in hyps]
    return nbest


@torch.no_grad()
def seq_logprob(model: Seq2Seq, sv: Vocab, tv: Vocab, srcs: list[str], tgts: list[str],
                bs: int = 256) -> list[float]:
    """Teacher-forced log P(tgt | src) per pair: the sum over the target characters and </s>."""
    model.eval()
    device = next(model.parameters()).device
    out = [0.0] * len(srcs)
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]) + len(tgts[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        x = _pad([sv.encode(srcs[j][:MAX_INPUT_CHARS]) for j in idx], device)
        y = _pad([[BOS] + tv.encode(tgts[j][:248]) for j in idx], device)
        logp = F.log_softmax(model(x, y[:, :-1]).float(), -1)
        tok = logp.gather(-1, y[:, 1:, None]).squeeze(-1).masked_fill(y[:, 1:] == PAD, 0.0)
        for j, v in zip(idx, tok.sum(-1).tolist()):
            out[j] = v
    return out
