"""Samdei Type: romanized Khmer (Latin chat spelling) to Khmer script.

Inference code for the released models samdei/samdei-type-mini and samdei/samdei-type-base:

- samdei_type.pipeline    Converter: a released model folder or Hub repo id -> convert(text), decoded as in the paper
- samdei_type.model       the character-level Transformer encoder-decoder, loading, KV-cached greedy and beam search
- samdei_type.onnx_infer  int8 ONNX decoding with onnxruntime (the reference for phone runtimes)
- samdei_type.rerank      n-best reranking with n-gram LMs (samdei_type.lm) and a reverse channel model
- samdei_type.evaluation  input chunking, batched decoding, CER / exact-match scoring
- samdei_type.text        Khmer normalization (with samdei_type.khnormal, vendored from SIL)
- samdei_type.cli         the samdei-type command
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("samdei-type")
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "0.0.0"
