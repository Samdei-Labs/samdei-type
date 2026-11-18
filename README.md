# Samdei Type

Samdei Type converts romanized Khmer, the Latin spelling people use in chat, to Khmer script:

```
knhom jong tov phsar      →  ខ្ញុំចង់ទៅផ្សារ
```

This repository holds the inference code for the released models: model loading, beam search, int8 ONNX decoding, reranking with n-gram language models and a channel model, and a command-line converter. The weights are on the Hugging Face Hub:

| Model | Parameters | Weights | Reranker (as in the paper) |
|---|---|---|---|
| [Samdei Type Mini](https://huggingface.co/samdei/samdei-type-mini) | 7.1M | int8 ONNX | word LM |
| [Samdei Type Base](https://huggingface.co/samdei/samdei-type-base) | 18.2M | PyTorch (safetensors) and int8 ONNX | character LM, word LM, channel model |

Both are character-level Transformer encoder-decoders. The benchmark is [Samdei Type Bench](https://huggingface.co/datasets/samdei/samdei-type-bench). The model cards give the scores, the training data and the limitations. Paper: `<paper-url>`.

Samdei (សំដី, "what you say") is our team, Samdei Labs.

## Install

Python 3.12 or later.

```bash
pip install "samdei-type[deploy] @ git+https://github.com/Samdei-Labs/samdei-type.git"
```

The `deploy` extra adds `onnxruntime`, which Samdei Type Mini needs. Without it, only the PyTorch weights of Samdei Type Base can be used. From a clone: `pip install -e ".[deploy]"`.

## Command line

```bash
samdei-type "knhom jong tov phsar"                                   # Mini, downloaded from the Hub
samdei-type -m samdei/samdei-type-base "bong nham bay nov?"          # Base, full reranker
samdei-type -m ./samdei-type-mini "ot mean wifi te"                  # a local model folder
samdei-type -m ./samdei-type-mini --no-rerank "ot mean wifi te"      # beam search only
echo "saek nh tv rean" | samdei-type -m ./samdei-type-mini -         # one message per line
samdei-type --nbest "jam knhom bontech"                              # also the n-best list with scores
```

`python -m samdei_type` runs the same command. To get a model folder: `hf download samdei/samdei-type-mini --local-dir samdei-type-mini`.

## Python

```python
from samdei_type.pipeline import Converter

conv = Converter("samdei/samdei-type-base")       # or a local folder; backend="onnx" for the int8 graphs
print(conv.convert("knhom jong tov phsar"))       # ខ្ញុំចង់ទៅផ្សារ
print(conv.convert_batch(["bong nham bay nov?", "ot mean wifi te"]))
```

`Converter` reads the model folder's `rerank.json`, so it decodes as the published scores were computed. The same steps with the lower-level pieces:

```python
import json
from huggingface_hub import snapshot_download
from samdei_type.evaluation import chunk_source, translate
from samdei_type.model import load
from samdei_type.rerank import Reranker

d = snapshot_download("samdei/samdei-type-base")
model, sv, tv = load(d)                                          # config.json + model.safetensors
weights = json.load(open(f"{d}/rerank.json"))["rerank_weights"]
rr = Reranker(weights, char_lm=f"{d}/lm/char5_nowiki.npz", word_lm=f"{d}/lm/wordfull_nowiki.json.gz",
              channel=f"{d}/channel")
pieces = chunk_source("knhom jong tov phsar", 8)
nbest = translate(model, sv, tv, pieces, beam=5)
print(" ".join(rr.rerank(p, hyps)[0][0] for p, hyps in zip(pieces, nbest)))
```

For the int8 graphs, `OnnxConverter(f"{d}/onnx").beam(piece, k=5)` from `samdei_type.onnx_infer` gives a piece's n-best list. That module uses only numpy and onnxruntime.

## How decoding works

1. The input is cut into pieces of at most eight words, with a break after any punctuation mark (`samdei_type.evaluation.chunk_source`). The models were trained on short pieces.
2. Each piece gets a beam search with k = 5 (`samdei_type.model.beam_cached`, or `OnnxConverter.beam`). Inputs are truncated to 249 characters.
3. Each hypothesis y of input x is scored as `log P(y|x) + a·log P_char(y) + b·log P_word(y) − c·#OOV(y) + d·log P_ch(x|y)` (`samdei_type.rerank`). The channel model P_ch reads Khmer and predicts the romanized input, so it checks that a hypothesis explains all of the input. The weights a–d come from the model's `rerank.json`.
4. The best hypotheses of the pieces are joined with a space.

## Package layout

| Module | What it does |
|---|---|
| `samdei_type.pipeline` | `Converter`: a model folder or Hub repo id → `convert(text)` |
| `samdei_type.model` | the encoder-decoder, `load`, KV-cached greedy and beam search |
| `samdei_type.onnx_infer` | int8 ONNX decoding with numpy and onnxruntime: the reference to port to a phone runtime |
| `samdei_type.rerank`, `samdei_type.lm` | n-best reranking; hashed character n-gram LM and word bigram LM |
| `samdei_type.evaluation` | input chunking, batched decoding, CER and exact match (`score`, as used for Samdei Type Bench) |
| `samdei_type.text`, `samdei_type.khnormal` | Khmer normalization |
| `samdei_type.vocab` | character vocabularies |
| `samdei_type.cli` | the `samdei-type` command |

Training code and the data generator are not part of this repository. The training data are on the Hub: [samdei/samdei-type-pretraining-data](https://huggingface.co/datasets/samdei/samdei-type-pretraining-data) and [samdei/samdei-type-finetuning-data](https://huggingface.co/datasets/samdei/samdei-type-finetuning-data).

## Tests

```bash
python -m unittest discover -s tests
SAMDEI_TYPE_MODEL=./samdei-type-mini python -m unittest discover -s tests   # also an end-to-end conversion
```

## License

The code is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). `src/samdei_type/khnormal.py` is vendored from SIL Global's [khmer-character-specification](https://github.com/sillsdev/khmer-character-specification) under the MIT License. The models and datasets on the Hub have their own licenses, given in their cards.

## Citation

```bibtex
@misc{samdei-type-2026,
  title  = {Samdei Type: Separating What Is Written from How It Is Spelled for Romanized Khmer Transliteration},
  author = {<authors>},
  year   = {2026},
  url    = {<paper-url>}
}
```
