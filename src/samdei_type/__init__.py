"""Samdei Type: romanized Khmer (Latin chat spelling) to Khmer script.

Inference code for the released models samdei/samdei-type-mini and samdei/samdei-type-base:

- samdei_type.pipeline    Converter: a model folder or Hub repo id -> convert(text), decoded as in the paper
- samdei_type.model       the encoder-decoder, load(), KV-cached greedy and beam search
- samdei_type.onnx_infer  beam search over the int8 ONNX graphs with onnxruntime (the reference for phones)
- samdei_type.rerank      n-best reranking with the language models (samdei_type.lm) and a channel model
- samdei_type.evaluation  input chunking, batched decoding, CER and exact-match scoring
- samdei_type.text        Khmer normalization (with samdei_type.khnormal, vendored from SIL)
- samdei_type.vocab       character vocabularies
- samdei_type.cli         the samdei-type command
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("samdei-type")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0.0.0"
