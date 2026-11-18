# Samdei Type

Samdei Type converts romanized Khmer, the Latin spelling many people use in chat, to Khmer script:

```console
$ samdei-type "knhom jong tov phsar"
ខ្ញុំចង់ទៅផ្សារ
```

This package runs the released Samdei Type models: a command-line converter and a Python API, with beam search,
int8 ONNX decoding and the reranking used in the paper. The models are on the Hugging Face Hub and are downloaded
on first use:

| Model | Parameters | Decoded as in the paper with | First download |
|---|---|---|---|
| [Samdei Type Mini](https://huggingface.co/samdei/samdei-type-mini) | 7.1M | int8 ONNX graphs + word LM | about 9 MB |
| [Samdei Type Base](https://huggingface.co/samdei/samdei-type-base) | 18.2M | PyTorch weights + character LM, word LM and channel model | about 125 MB |

Both are character-level Transformer encoder-decoders. Mini is small enough for a phone keyboard; Base is more
accurate. Their scores on [Samdei Type Bench](https://huggingface.co/datasets/samdei/samdei-type-bench), training
data and limitations are in the model cards. All releases: the
[Samdei Type collection](https://huggingface.co/collections/samdei/samdei-type-romanized-khmer-khmer-script).
Paper: [technical report (PDF)](https://github.com/Samdei-Labs/samdei-type/blob/main/paper/samdei-type.pdf).

Samdei (សំដី, "what you say") is our team, Samdei Labs.

## Install

```bash
pip install "samdei-type[deploy]"
```

Python 3.10 or later. The `deploy` extra adds `onnxruntime` for the int8 ONNX graphs, which Mini is decoded with.
Without it, Mini falls back to its PyTorch weights. PyTorch is needed either way: on macOS the install
takes about 1 GB. On Linux, pip installs PyTorch with its CUDA libraries, several GB; for a CPU-only install, add
PyTorch's CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The latest code from GitHub: `pip install "samdei-type[deploy] @ git+https://github.com/Samdei-Labs/samdei-type.git"`.

## Command line

```console
$ samdei-type "knhom jong tov phsar"                      # Samdei Type Mini
ខ្ញុំចង់ទៅផ្សារ
$ samdei-type -m samdei/samdei-type-base "bong nham bay nov?"
បងញ៉ាំបាយនៅ?
$ echo "saek nh tv rean" | samdei-type -                  # one message per line from stdin
ស្អែកខ្ញុំទៅរៀន
$ samdei-type --nbest "jam knhom bontech"                 # also each piece's candidates and scores
ចាំខ្ញុំបន្តិច
       -8.16  ចាំខ្ញុំបន្តិច
      -11.31  ចាំ ខ្ញុំបន្តិច
      ...
```

`samdei-type --help` lists the options: `--backend torch|onnx`, `--beam`, `--no-rerank` (beam search only),
`--no-chunk`, `--tsv`, `--device`, `--threads`. `python -m samdei_type` runs the same command.

The first run downloads the files the model uses into the Hugging Face cache (`~/.cache/huggingface`). After
that it only checks the Hub for updates and works offline; set `HF_HUB_OFFLINE=1` to skip the check.
`-m` also takes a local folder:

```bash
hf download samdei/samdei-type-mini --local-dir samdei-type-mini
samdei-type -m ./samdei-type-mini "ot mean wifi te"
```

## Python

```python
from samdei_type.pipeline import Converter

conv = Converter("samdei/samdei-type-base")              # a Hub repo id or a local model folder
conv.convert("knhom jong tov phsar")                     # 'ខ្ញុំចង់ទៅផ្សារ'
conv.convert_batch(["bong nham bay nov?", "ot mean wifi te"])
                                                         # ['បងញ៉ាំបាយនៅ?', 'អត់មាន wifi ទេ']
mini = Converter("samdei/samdei-type-mini")              # int8 ONNX graphs, as rerank.json says
mini.nbest(mini.pieces("jam knhom bontech"))             # [[('ចាំខ្ញុំបន្តិច', -8.16), ...]]
```

`Converter` reads the model's `rerank.json`, so it decodes as the published scores were computed: the backend,
the beam size, the pieces and the rerank weights. `backend="torch"` or `"onnx"` overrides the backend. The same steps
with the lower-level pieces:

```python
import json
from huggingface_hub import snapshot_download
from samdei_type.evaluation import chunk_source, translate
from samdei_type.model import load
from samdei_type.rerank import Reranker

d = snapshot_download("samdei/samdei-type-base")
model, sv, tv = load(d)                                  # config.json + model.safetensors
weights = json.load(open(f"{d}/rerank.json"))["rerank_weights"]
rr = Reranker(weights, char_lm=f"{d}/lm/char5_nowiki.npz", word_lm=f"{d}/lm/wordfull_nowiki.json.gz",
              channel=f"{d}/channel")
pieces = chunk_source("knhom jong tov phsar", 8)
nbest = translate(model, sv, tv, pieces, beam=5)
print(" ".join(rr.rerank(p, hyps)[0][0] for p, hyps in zip(pieces, nbest)))
```

For the int8 graphs, `OnnxConverter(f"{d}/onnx").beam(piece, k=5)` from `samdei_type.onnx_infer` gives a piece's
n-best list. That module uses only numpy and onnxruntime, and is the reference to port to a phone runtime.

## How decoding works

1. The input is cut into pieces of at most eight words, with a break after any punctuation mark
   (`samdei_type.evaluation.chunk_source`). The models were trained on short pieces.
2. Each piece gets a beam search with k = 5 (`samdei_type.model.beam_cached`, or `OnnxConverter.beam`). A piece
   longer than 249 characters is cut.
3. Each hypothesis y of input x is scored as `log P(y|x) + a·log P_char(y) + b·log P_word(y) − c·#OOV(y) + d·log P_ch(x|y)`
   (`samdei_type.rerank`). The channel model P_ch reads Khmer and predicts the romanized input, so it checks that a
   hypothesis explains all of the input. The weights a–d come from the model's `rerank.json`.
4. The best hypotheses of the pieces are joined with a space.

## Limitations

- The models convert what people type in chat. They are not a dictionary: an unusual spelling, a name or a rare
  word can come out as a common word that sounds alike. The model cards give the error rates and examples.
- English words are often spelled out in Khmer script (`see you` → ស៊ីយូ); some, like `wifi`, are kept.
- Emoji, URLs and @mentions are not passed through: emoji are dropped and other symbols can change.
  Split them off before converting if you need to keep them.

## Package layout

| Module | What it does |
|---|---|
| `samdei_type.pipeline` | `Converter`: a model folder or Hub repo id → `convert(text)` |
| `samdei_type.model` | the encoder-decoder, `load`, KV-cached greedy and beam search |
| `samdei_type.onnx_infer` | int8 ONNX decoding with numpy and onnxruntime |
| `samdei_type.rerank`, `samdei_type.lm` | n-best reranking; hashed character n-gram LM and word bigram LM |
| `samdei_type.evaluation` | input chunking, batched decoding, CER and exact match (`score`, as for Samdei Type Bench) |
| `samdei_type.text`, `samdei_type.khnormal` | Khmer normalization |
| `samdei_type.vocab` | character vocabularies |
| `samdei_type.cli` | the `samdei-type` command |

Training code and the data generator are not part of this package. The training data are on the Hub:
[samdei/samdei-type-pretraining-data](https://huggingface.co/datasets/samdei/samdei-type-pretraining-data) and
[samdei/samdei-type-finetuning-data](https://huggingface.co/datasets/samdei/samdei-type-finetuning-data).

## Development

```bash
git clone https://github.com/Samdei-Labs/samdei-type && cd samdei-type
pip install -e ".[deploy]"
python -m unittest discover -s tests
SAMDEI_TYPE_MODEL=samdei/samdei-type-mini python -m unittest discover -s tests   # also an end-to-end conversion
```

## License

The code is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`). `src/samdei_type/khnormal.py` is
vendored from SIL Global's [khmer-character-specification](https://github.com/sillsdev/khmer-character-specification)
under the MIT License. The models and datasets on the Hub have their own licenses, given in their cards.

## Citation

The technical report is [paper/samdei-type.pdf](https://github.com/Samdei-Labs/samdei-type/blob/main/paper/samdei-type.pdf).

```bibtex
@misc{samdei-type-2026,
  title  = {Samdei Type: Separating What Is Written from How It Is Spelled for Romanized Khmer Transliteration},
  author = {Mengsay Loem},
  year   = {2026},
  note   = {Technical report},
  url    = {https://github.com/Samdei-Labs/samdei-type/blob/main/paper/samdei-type.pdf}
}
```
