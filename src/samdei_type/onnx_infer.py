"""Beam search over the int8 ONNX graphs in a released model's onnx/ folder, with numpy and onnxruntime only.

This is the reference for phone runtimes. encoder_kv runs once per input and returns each decoder layer's
cross-attention keys and values; decoder_step decodes one token for all beams, with a self-attention cache.

    conv = OnnxConverter("samdei-type-mini/onnx")
    conv.beam("knhom jong tov phsar", k=5)     # [(khmer, log probability), ...], best first
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .vocab import BOS, EOS, MAX_INPUT_CHARS, Vocab


class OnnxConverter:
    def __init__(self, onnx_dir: str, threads: int = 1):
        d = Path(onnx_dir)
        meta = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        self.sv, self.tv = Vocab(meta["src_vocab"]), Vocab(meta["tgt_vocab"])
        self.L, self.H, self.dh = meta["layers"], meta["heads"], meta["head_dim"]
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "encoder_kv.int8.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(d / "decoder_step.int8.onnx"), so, providers=prov)

    def beam(self, s: str, k: int = 5, max_len: int = 200) -> list[tuple[str, float]]:
        """Beam search with the same search and stopping rule as model.beam_cached. Returns the n-best
        [(text, log probability)], ranked by log probability / length."""
        L = self.L
        src = np.array([self.sv.encode(s[:MAX_INPUT_CHARS])], dtype=np.int64)
        xkv = self.enc.run(None, {"src": src})  # xk0, xv0, xk1, xv1, ...
        xk = [np.repeat(xkv[2 * i], k, 0) for i in range(L)]
        xv = [np.repeat(xkv[2 * i + 1], k, 0) for i in range(L)]
        pk = pv = [np.zeros((1, self.H, 0, self.dh), np.float32)] * L
        seqs, scores, finished = [[]], np.zeros(1), []
        for pos in range(min(max_len, 2 * src.shape[1] + 10)):
            n = len(seqs)
            feeds = {"tok": np.array([q[-1] if q else BOS for q in seqs], np.int64), "pos": np.array([pos], np.int64)}
            feeds |= {f"pk{i}": pk[i] for i in range(L)} | {f"pv{i}": pv[i] for i in range(L)}
            feeds |= {f"xk{i}": xk[i][:n] for i in range(L)} | {f"xv{i}": xv[i][:n] for i in range(L)}
            res = self.dec.run(None, feeds)  # logits, then the new pk0.., pv0..
            lg = res[0].astype(np.float64)
            lp = lg - lg.max(-1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
            cand = (scores[:, None] + lp).ravel()
            top = np.argsort(-cand, kind="stable")[: 2 * k]
            V = lp.shape[1]
            keep = []
            for c in top:
                b, t = divmod(int(c), V)
                if t == EOS:
                    finished.append((seqs[b], cand[c], cand[c] / (len(seqs[b]) + 1)))
                elif len(keep) < k:
                    keep.append((b, t, cand[c]))
            rows = np.array([b for b, _, _ in keep])
            seqs = [seqs[b] + [t] for b, t, _ in keep]
            scores = np.array([sc for _, _, sc in keep])
            pk = [res[1 + i][rows] for i in range(L)]
            pv = [res[1 + L + i][rows] for i in range(L)]
            if len(finished) >= k and sorted((f[1] for f in finished), reverse=True)[k - 1] >= scores.max():
                break
        if not finished:
            finished = [(seqs[0], scores[0], scores[0])]
        finished.sort(key=lambda f: -f[2])
        return [(self.tv.decode(q), float(sc)) for q, sc, _ in finished[:k]]
