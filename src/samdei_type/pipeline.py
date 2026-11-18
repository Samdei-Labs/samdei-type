"""Convert romanized Khmer with a released model folder, decoded as in the paper.

    from samdei_type.pipeline import Converter
    conv = Converter("samdei/samdei-type-mini")      # a Hugging Face repo id or a local folder
    conv.convert("knhom jong tov phsar")             # 'ខ្ញុំចង់ទៅផ្សារ'

A released model folder holds `rerank.json` (beam size, piece length, rerank weights and the paths of the
language models and the channel model), the PyTorch weights (`config.json` + `model.safetensors`) and/or the
int8 ONNX graphs (`onnx/`), and the files that rerank.json names (`lm/`, `channel/`).

Decoding: the input is cut into pieces of at most 8 words, breaking after punctuation (evaluation.chunk_source);
each piece gets a beam search (k = 5) and its n-best list is reranked (rerank.Reranker); the best outputs are
joined with a space.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from .evaluation import chunk_source

RERANK_CONFIG = "rerank.json"
_BACKENDS = ("auto", "torch", "onnx")


def _pick(d: Path, backend: str) -> str:
    if backend == "auto":
        return "torch" if (d / "config.json").is_file() else "onnx"
    return backend


def _resolve(model: str | Path, backend: str, revision: str | None = None) -> tuple[Path, str]:
    """(folder, backend) for a local folder, or for a Hugging Face repo id after downloading what the backend
    needs: the ONNX backend skips the root PyTorch weights, the torch backend skips onnx/."""
    d = Path(model).expanduser()
    if d.is_dir():
        return d, _pick(d, backend)
    if d.exists() or str(model).startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model}: not a model folder")
    from huggingface_hub import snapshot_download
    repo = str(model)
    if backend == "auto":  # small files first, to see which weights the repo has
        backend = _pick(Path(snapshot_download(repo, revision=revision, allow_patterns=[RERANK_CONFIG, "config.json"])),
                        backend)
    ignore = ["onnx/*"] if backend == "torch" else ["model.safetensors"]
    return Path(snapshot_download(repo, revision=revision, ignore_patterns=ignore)), backend


class Converter:
    """Romanized Khmer -> Khmer script with a released model (local folder or Hugging Face repo id).

    backend: "torch" (config.json + model.safetensors), "onnx" (onnx/ int8 graphs, onnxruntime) or "auto"
    (torch when the folder has PyTorch weights, else onnx). rerank: use the language models and channel model
    named in rerank.json (the setting of the published scores). beam / max_words: override rerank.json.
    """

    def __init__(self, model: str | Path, backend: str = "auto", device: str = "cpu", rerank: bool = True,
                 beam: int | None = None, max_words: int | None = None, threads: int = 1,
                 revision: str | None = None):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, not {backend!r}")
        d, backend = _resolve(model, backend, revision)
        self.dir, self.device, self.backend = d, device, backend
        cfg_path = d / RERANK_CONFIG
        self.config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self.beam = beam or self.config.get("beam", 5)
        self.max_words = max_words or self.config.get("chunk_max_words", 8)
        if backend == "torch":
            from .model import load
            self.model, self.sv, self.tv, _ = load(str(d), device)
            self.onnx = None
        else:
            if not (d / "onnx").is_dir():
                raise FileNotFoundError(f"{d}: no onnx/ folder (and no config.json for the torch backend)")
            try:
                import onnxruntime  # noqa: F401
            except ImportError as e:
                raise ImportError("the onnx backend needs onnxruntime: pip install 'samdei-type[deploy]'") from e
            from .onnx_infer import OnnxConverter
            self.onnx = OnnxConverter(str(d / "onnx"), threads=threads)
        self.reranker = self._reranker() if rerank else None

    def _reranker(self):
        from .rerank import Reranker, load_weights
        if not self.config:
            warnings.warn(f"{self.dir}: no {RERANK_CONFIG}; decoding without reranking")
            return None
        weights = load_weights(self.dir / RERANK_CONFIG)
        paths = {}
        for key, weight_keys in (("char_lm", ("char",)), ("word_lm", ("word", "oov")), ("channel_model", ("channel",))):
            rel = self.config.get(key)
            if not rel or not any(weights.get(k) for k in weight_keys):
                continue
            p = self.dir / rel
            if not p.exists():
                warnings.warn(f"{p} (named in {RERANK_CONFIG}) is missing; its rerank term is skipped")
                continue
            paths[key] = str(p)
        rr = Reranker(weights, paths.get("char_lm"), paths.get("word_lm"), paths.get("channel_model"), self.device)
        return rr if rr.active else None

    def nbest(self, pieces: list[str]) -> list[list[tuple[str, float]]]:
        """Per piece: [(khmer, score)] best first; scores are reranked totals when reranking is on."""
        if self.onnx is not None:
            lists = [self.onnx.beam(p, k=self.beam) for p in pieces]
        else:
            from .evaluation import translate
            _, lists = translate(self.model, self.sv, self.tv, pieces, self.device, beam=self.beam)
        if self.reranker is not None:
            lists = [self.reranker.rerank(p, hyps) for p, hyps in zip(pieces, lists)]
        return lists

    def convert_batch(self, texts: list[str], chunk: bool = True) -> list[str]:
        """Convert several messages; all their pieces are decoded together (batched on the torch backend)."""
        pieces, owner = [], []
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            for p in (chunk_source(t, self.max_words) if chunk else [t.strip()]):
                pieces.append(p); owner.append(i)
        outs: list[list[str]] = [[] for _ in texts]
        for i, hyps in zip(owner, self.nbest(pieces) if pieces else []):
            outs[i].append(hyps[0][0] if hyps else "")
        return [" ".join(o) for o in outs]

    def convert(self, text: str, chunk: bool = True) -> str:
        return self.convert_batch([text], chunk)[0]
