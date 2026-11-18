"""Inference over the exported KV-cached ONNX graphs (the onnx/ folder of a released model).

This is the reference for the phone runtime: plain numpy around two ONNX Runtime sessions, no torch.

    conv = OnnxConverter("samdei-type-mini/onnx")           # int8 graphs + vocab.json
    conv.beam("knhom jong tov phsar", k=5)                   # [(khmer, logprob), ...] best first
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import BOS, EOS, Vocab


class OnnxConverter:
    def __init__(self, onnx_dir: str, int8: bool = True, threads: int = 1, ckpt: str | None = None):
        import onnxruntime as ort
        d = Path(onnx_dir)
        meta_path = d / "vocab.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:  # exports made before vocab.json existed: read the vocab from the checkpoint
            from .model import load
            m, sv, tv, _ = load(ckpt or str(d.parent / "best.pt"), "cpu")
            meta = {"src_vocab": sv.to_json(), "tgt_vocab": tv.to_json(), "layers": len(m.decoder.layers),
                    "heads": m.cfg.nhead, "head_dim": m.cfg.d_model // m.cfg.nhead}
        self.sv, self.tv = Vocab(meta["src_vocab"]), Vocab(meta["tgt_vocab"])
        self.L, self.H, self.dh = meta["layers"], meta["heads"], meta["head_dim"]
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        sfx = ".int8" if int8 else ""
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / f"encoder_kv{sfx}.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(d / f"decoder_step{sfx}.onnx"), so, providers=prov)

    def _empty(self, n: int = 1):
        return [np.zeros((n, self.H, 0, self.dh), np.float32) for _ in range(self.L)]

    def greedy(self, s: str, max_len: int = 200) -> str:
        L = self.L
        src = np.array([self.sv.encode(s[:249])], dtype=np.int64)
        xkv = self.enc.run(None, {"src": src})  # outputs ordered xk0, xv0, xk1, xv1, ...
        feeds = {f"xk{i}": xkv[2 * i] for i in range(L)} | {f"xv{i}": xkv[2 * i + 1] for i in range(L)}
        pk, pv = self._empty(), self._empty()
        tok, out = BOS, []
        for pos in range(min(max_len, 2 * src.shape[1] + 10)):
            res = self.dec.run(None, {"tok": np.array([tok], np.int64), "pos": np.array([pos], np.int64)} | feeds |
                               {f"pk{i}": pk[i] for i in range(L)} | {f"pv{i}": pv[i] for i in range(L)})
            tok = int(res[0][0].argmax())
            if tok == EOS:
                break
            out.append(tok)
            pk, pv = res[1:1 + L], res[1 + L:1 + 2 * L]
        return self.tv.decode(out)

    def beam(self, s: str, k: int = 5, max_len: int = 200) -> list[tuple[str, float]]:
        """Beam search, all alive beams in one decoder call per step; same search and stopping rule as
        samdei_type.model.beam_cached. Returns the n-best [(text, logprob)], best first."""
        L = self.L
        src = np.array([self.sv.encode(s[:249])], dtype=np.int64)
        xkv = self.enc.run(None, {"src": src})
        xk = [np.repeat(xkv[2 * i], k, 0) for i in range(L)]; xv = [np.repeat(xkv[2 * i + 1], k, 0) for i in range(L)]
        pk, pv = self._empty(), self._empty()
        seqs, scores, finished = [[]], np.zeros(1), []
        for pos in range(min(max_len, 2 * src.shape[1] + 10)):
            n = len(seqs)
            feeds = {"tok": np.array([q[-1] if q else BOS for q in seqs], np.int64), "pos": np.array([pos], np.int64)}
            feeds |= {f"pk{i}": pk[i] for i in range(L)} | {f"pv{i}": pv[i] for i in range(L)}
            feeds |= {f"xk{i}": xk[i][:n] for i in range(L)} | {f"xv{i}": xv[i][:n] for i in range(L)}
            res = self.dec.run(None, feeds)
            lg = res[0].astype(np.float64)
            lp = lg - lg.max(-1, keepdims=True); lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
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
            pk = [res[1 + i][rows] for i in range(L)]; pv = [res[1 + L + i][rows] for i in range(L)]
            if len(finished) >= k and sorted((f[1] for f in finished), reverse=True)[k - 1] >= scores.max():
                break
        if not finished:
            finished = [(seqs[0], scores[0], scores[0])]
        finished.sort(key=lambda f: -f[2])
        return [(self.tv.decode(q), float(sc)) for q, sc, _ in finished[:k]]
