"""Convert romanized Khmer with a released model, decoded as in the paper.

    from samdei_type.pipeline import Converter
    conv = Converter("samdei/samdei-type-mini")      # a Hugging Face repo id or a local folder
    conv.convert("knhom jong tov phsar")             # 'ខ្ញុំចង់ទៅផ្សារ'

A released model folder holds the PyTorch weights (config.json, model.safetensors), the int8 ONNX graphs
(onnx/), and rerank.json: the backend, beam size, piece length and rerank weights of the published scores, the
paths of the language models (lm/) and the channel model ("channel_model"; "channel_self": the model itself).

The input is cut into pieces of at most 8 words, breaking after punctuation (evaluation.chunk_source). Each
piece gets a beam search with k = 5, its n-best list is reranked (rerank.Reranker), and the best outputs of
the pieces are joined with a space. A comma not between digits becomes a space (text.drop_commas).
"""
import importlib.util
import json
import warnings
from pathlib import Path

from .evaluation import chunk_khmer, chunk_source
from .text import drop_commas, normalize_khmer
from .vocab import REV_TAG

RERANK_CONFIG = "rerank.json"
_BACKENDS = ("auto", "torch", "onnx")
_TORCH_FILES = ("config.json", "model.safetensors")
_ONNX_FILES = ("onnx/encoder_kv.int8.onnx", "onnx/decoder_step.int8.onnx", "onnx/vocab.json")
# rerank.json key -> the rerank weights that use it (rerank.Reranker loads a model only if one of them is nonzero)
_RERANK_FILES = {"char_lm": ("char",), "word_lm": ("word", "oov"), "channel_model": ("channel",)}


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


def _needed(cfg: dict, backend: str, rerank: bool) -> list[str]:
    """The files a Converter reads from a model folder besides rerank.json: the backend's weights and, with
    rerank, the language models and the channel model that rerank.json names and gives a nonzero weight."""
    files = list(_TORCH_FILES if backend == "torch" else _ONNX_FILES)
    if rerank:
        weights = cfg.get("rerank_weights", {})
        for key, uses in _RERANK_FILES.items():
            rel = cfg.get(key)
            if rel and any(weights.get(u) for u in uses):
                rel = rel.rstrip("/")
                files += [f"{rel}/{f}" for f in _TORCH_FILES] if key == "channel_model" else [rel]
        if cfg.get("channel_self") and weights.get("channel"):  # the channel is the model's own torch weights
            files += [f for f in _TORCH_FILES if f not in files]
    return files


def _download(repo: str, revision: str | None, backend: str, rerank: bool) -> tuple[Path, str]:
    """Fetch only the files a Converter needs, one at a time. A file that is cached and up to date is not
    fetched again and prints nothing (snapshot_download draws its progress bars on every call)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    def get(name: str, rev: str | None) -> Path | None:
        try:
            return Path(hf_hub_download(repo, name, revision=rev))
        except LocalEntryNotFoundError:  # offline and not cached
            raise
        except EntryNotFoundError:  # not in the repo
            return None

    found = get(RERANK_CONFIG, revision)
    rev = found.parent.name if found else revision  # the commit, so that all files come from the same one
    torch_cfg = get("config.json", rev)  # the torch backend reads it; _pick checks for it
    found = found or torch_cfg
    if found is None:
        raise ValueError(f"{repo}: no {RERANK_CONFIG} or config.json, so not a Samdei Type model")
    d = found.parent
    backend = _pick(d, backend)
    cfg = json.loads((d / RERANK_CONFIG).read_text(encoding="utf-8")) if (d / RERANK_CONFIG).is_file() else {}
    for name in _needed(cfg, backend, rerank):
        get(name, rev)
    return d, backend


def _resolve(model: str | Path, backend: str, revision: str | None, rerank: bool) -> tuple[Path, str]:
    """(folder, backend) for a local folder, or for a Hugging Face repo id after downloading the files that
    the backend and the reranker need."""
    d = Path(model).expanduser()
    if d.is_dir():
        return d, _pick(d, backend)
    if d.exists() or str(model).startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model}: not a model folder")
    try:
        return _download(str(model), revision, backend, rerank)
    except ValueError:
        raise
    except Exception as e:  # not found, private, no connection, ...
        raise OSError(f"could not download {model} from the Hugging Face Hub ({type(e).__name__}). Check the "
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
        d, backend = _resolve(model, backend, revision, rerank)
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
        channel = paths.get("channel_model")
        if self.config.get("channel_self"):
            channel = (self.model, self.sv, self.tv) if self.onnx is None else str(self.dir)
        rr = Reranker(self.config.get("rerank_weights", {}), paths.get("char_lm"), paths.get("word_lm"),
                      channel, self.device)
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

    def convert_batch(self, texts: list[str], chunk: bool = True, reverse: bool = False) -> list[str]:
        """Convert several messages. Their pieces are decoded together (batched on the torch backend)."""
        if reverse:
            per_text = [self._khmer_pieces(t, chunk) for t in texts]
            best = self._greedy([REV_TAG + p for ps in per_text for p in ps])
        else:
            per_text = [self.pieces(t, chunk) for t in texts]
            best = [hyps[0][0] if hyps else "" for hyps in self.nbest([p for ps in per_text for p in ps])]
        out, i = [], 0
        for ps in per_text:
            joined = " ".join(best[i:i + len(ps)])
            out.append(joined if reverse else drop_commas(joined))
            i += len(ps)
        return out

    def convert(self, text: str, chunk: bool = True, reverse: bool = False) -> str:
        return self.convert_batch([text], chunk, reverse)[0]

    def _khmer_pieces(self, text: str, chunk: bool) -> list[str]:
        sv = self.sv if self.onnx is None else self.onnx.sv
        if REV_TAG not in sv.stoi:
            raise ValueError(f"{self.dir}: this model converts romanized Khmer to Khmer script only")
        text = normalize_khmer(text)
        if not text:
            return []
        return chunk_khmer(text, self.max_words) if chunk else [text]

    def _greedy(self, pieces: list[str]) -> list[str]:
        if self.onnx is not None:
            return [self.onnx.beam(p, k=1)[0][0] for p in pieces]
        from .evaluation import translate
        return [hyps[0][0] for hyps in translate(self.model, self.sv, self.tv, pieces, beam=1)]
