"""Convert romanized Khmer with a released model, decoded as in the paper.

    from samdei_type.pipeline import Converter
    conv = Converter("samdei/samdei-type-mini")      # a Hugging Face repo id or a local folder
    conv.convert("knhom jong tov phsar")             # 'ខ្ញុំចង់ទៅផ្សារ'

A released model folder holds the PyTorch weights (config.json, model.safetensors), the int8 ONNX graphs
(onnx/), and rerank.json: the backend, beam size, piece length and rerank weights of the published scores, and
the paths of the language models (lm/) and the channel model (channel/).

The input is cut into pieces of at most 8 words, breaking after punctuation (evaluation.chunk_source). Each
piece gets a beam search with k = 5, its n-best list is reranked (rerank.Reranker), and the best outputs of
the pieces are joined with a space.
"""
import importlib.util
import json
import warnings
from pathlib import Path

from .evaluation import chunk_source

RERANK_CONFIG = "rerank.json"
_BACKENDS = ("auto", "torch", "onnx")


def _pick(d: Path, backend: str) -> str:
    """auto: the backend rerank.json names (the one the published scores used), else torch if the folder has
    PyTorch weights, else onnx. ONNX falls back to torch when onnxruntime is not installed."""
    if backend != "auto":
        return backend
    cfg = d / RERANK_CONFIG
    named = json.loads(cfg.read_text(encoding="utf-8")).get("backend") if cfg.is_file() else None
    has_torch = (d / "config.json").is_file()
    if named == "onnx" and has_torch and importlib.util.find_spec("onnxruntime") is None:
        warnings.warn("onnxruntime is not installed, so the PyTorch weights are used; for the int8 graphs "
                      "the published scores used: pip install 'samdei-type[deploy]'")
        return "torch"
    if named in ("torch", "onnx"):
        return named
    return "torch" if has_torch else "onnx"


def _resolve(model: str | Path, backend: str, revision: str | None) -> tuple[Path, str]:
    """(folder, backend) for a local folder, or for a Hugging Face repo id after downloading the files the
    backend needs (the ONNX backend skips model.safetensors, the torch backend skips onnx/)."""
    d = Path(model).expanduser()
    if d.is_dir():
        return d, _pick(d, backend)
    if d.exists() or str(model).startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model}: not a model folder")
    from huggingface_hub import snapshot_download
    repo = str(model)
    try:
        if backend == "auto":  # the small files first, to see which weights the repo has
            small = snapshot_download(repo, revision=revision, allow_patterns=[RERANK_CONFIG, "config.json"])
            backend = _pick(Path(small), backend)
        ignore = ["onnx/*"] if backend == "torch" else ["model.safetensors"]
        return Path(snapshot_download(repo, revision=revision, ignore_patterns=ignore)), backend
    except Exception as e:  # not found, private, no connection, ...
        raise OSError(f"could not download {repo} from the Hugging Face Hub ({type(e).__name__}). Check the "
                      "repo id and the connection, or pass a local model folder.") from e


class Converter:
    """Romanized Khmer -> Khmer script with a released model (a local folder or a Hugging Face repo id).

    backend: "torch" (config.json + model.safetensors), "onnx" (the int8 graphs in onnx/, needs onnxruntime)
    or "auto" (the backend named in rerank.json: onnx for Mini, torch for Base). rerank: rerank with the language models
    and the channel model named in rerank.json, as for the published scores. beam, max_words: override
    rerank.json.
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
        self.onnx = None
        if backend == "torch":
            from .model import load
            self.model, self.sv, self.tv = load(d, device)
        else:
            if not (d / "onnx").is_dir():
                raise FileNotFoundError(f"{d}: no onnx/ folder (and no config.json for the torch backend)")
            try:
                from .onnx_infer import OnnxConverter
            except ImportError as e:
                raise ImportError("the onnx backend needs onnxruntime: pip install 'samdei-type[deploy]'") from e
            self.onnx = OnnxConverter(str(d / "onnx"), threads=threads)
        self.reranker = self._reranker() if rerank else None

    def _reranker(self):
        from .rerank import Reranker
        if not self.config:
            warnings.warn(f"{self.dir}: no {RERANK_CONFIG}; decoding without reranking")
            return None
        paths = {}
        for key in ("char_lm", "word_lm", "channel_model"):
            rel = self.config.get(key)
            if not rel:
                continue
            p = self.dir / rel
            if p.exists():
                paths[key] = str(p)
            else:
                warnings.warn(f"{p} (named in {RERANK_CONFIG}) is missing; its rerank term is skipped")
        rr = Reranker(self.config.get("rerank_weights", {}), paths.get("char_lm"), paths.get("word_lm"),
                      paths.get("channel_model"), self.device)
        return rr if rr.active else None

    def pieces(self, text: str, chunk: bool = True) -> list[str]:
        """The pieces text is decoded in: chunk_source pieces, or the whole text if chunk is False."""
        if not text.strip():
            return []
        return chunk_source(text, self.max_words) if chunk else [text.strip()]

    def nbest(self, pieces: list[str]) -> list[list[tuple[str, float]]]:
        """Per piece: [(khmer, score)], best first; with reranking, the scores are the reranked totals."""
        if self.onnx is not None:
            lists = [self.onnx.beam(p, k=self.beam) for p in pieces]
        else:
            from .evaluation import translate
            lists = translate(self.model, self.sv, self.tv, pieces, beam=self.beam)
        if self.reranker is not None:
            lists = [self.reranker.rerank(p, hyps) for p, hyps in zip(pieces, lists)]
        return lists

    def convert_batch(self, texts: list[str], chunk: bool = True) -> list[str]:
        """Convert several messages. Their pieces are decoded together (batched on the torch backend)."""
        per_text = [self.pieces(t, chunk) for t in texts]
        best = [hyps[0][0] if hyps else "" for hyps in self.nbest([p for ps in per_text for p in ps])]
        out, i = [], 0
        for ps in per_text:
            out.append(" ".join(best[i:i + len(ps)]))
            i += len(ps)
        return out

    def convert(self, text: str, chunk: bool = True) -> str:
        return self.convert_batch([text], chunk)[0]
