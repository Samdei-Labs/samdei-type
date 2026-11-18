"""Character-level Transformer encoder-decoder (deep encoder, shallow decoder)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


class Vocab:
    def __init__(self, chars: list[str]):
        self.itos = SPECIALS + [c for c in chars if c not in SPECIALS]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s: str, bos: bool = False, eos: bool = True) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in s]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)

    def __len__(self):
        return len(self.itos)

    def to_json(self):
        return self.itos[len(SPECIALS):]


@dataclass
class ModelConfig:
    src_vocab: int
    tgt_vocab: int
    d_model: int = 384
    nhead: int = 6
    enc_layers: int = 6
    dec_layers: int = 3
    ff: int = 1536
    dropout: float = 0.1
    max_len: int = 256


class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.src_emb = nn.Embedding(cfg.src_vocab, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab, d, padding_idx=PAD)
        self.src_pos = nn.Embedding(cfg.max_len, d)
        self.tgt_pos = nn.Embedding(cfg.max_len, d)
        enc_layer = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ff, cfg.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.nhead, cfg.ff, cfg.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, cfg.tgt_vocab)
        self.scale = math.sqrt(d)

    def encode(self, src: torch.Tensor):
        pos = torch.arange(src.size(1), device=src.device)
        x = self.src_emb(src) * self.scale + self.src_pos(pos)[None]
        mask = src == PAD
        return self.encoder(x, src_key_padding_mask=mask), mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, mem_mask: torch.Tensor):
        T = tgt_in.size(1)
        pos = torch.arange(T, device=tgt_in.device)
        y = self.tgt_emb(tgt_in) * self.scale + self.tgt_pos(pos)[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.decoder(y, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_in == PAD,
                         memory_key_padding_mask=mem_mask)
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, max_len: int = 200) -> list[list[int]]:
        memory, mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, memory, mask)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if bool(done.all()):
                break
        return [row[1:].tolist() for row in ys]

    @torch.no_grad()
    def beam(self, src: torch.Tensor, k: int = 5, max_len: int = 200, len_pen: float = 1.0):
        """Batched beam search. Returns per input a list of (ids, logprob_sum) sorted best-first."""
        B = src.size(0)
        NEG = float("-inf")
        memory, mask = self.encode(src)
        memory = memory.repeat_interleave(k, 0)
        mask = mask.repeat_interleave(k, 0)
        dev = src.device
        ys = torch.full((B * k, 1), BOS, dtype=torch.long, device=dev)
        scores = torch.full((B, k), NEG, device=dev)
        scores[:, 0] = 0.0
        base = (torch.arange(B, device=dev) * k)[:, None]
        finished: list[list[tuple[list[int], float, float]]] = [[] for _ in range(B)]
        done = [False] * B
        for step in range(max_len):
            logp = F.log_softmax(self.decode(ys, memory, mask)[:, -1].float(), -1)  # (B*k, V)
            V = logp.size(-1)
            cand = (scores.view(-1, 1) + logp).view(B, k * V)
            top, idx = cand.topk(2 * k, dim=-1)  # (B, 2k)
            beam_idx, tok = idx // V, idx % V
            is_eos = tok == EOS
            # record finished hypotheses (one host sync per step)
            fin = (is_eos & (top > NEG)).nonzero().tolist()
            if fin:
                rows = torch.tensor([b * k + int(beam_idx[b, j]) for b, j in fin], device=dev)
                prev = ys[rows, 1:].tolist()
                sc = top[[b for b, _ in fin], [j for _, j in fin]].tolist()
                L = ys.size(1)  # tokens incl. BOS
                for (b, _), p, s in zip(fin, prev, sc):
                    if not done[b]:
                        finished[b].append((p, s, s / (L ** len_pen)))
            sel_sc, sel_j = top.masked_fill(is_eos, NEG).topk(k, dim=-1)  # best k non-EOS continuations
            rows = (base + beam_idx.gather(1, sel_j)).view(-1)
            ys = torch.cat([ys[rows], tok.gather(1, sel_j).view(-1, 1)], 1)
            scores = sel_sc
            # an input is done once it has k finished hyps no worse than its best alive one
            best_alive = scores.max(-1).values.tolist()
            for b in range(B):
                if not done[b] and len(finished[b]) >= k and \
                        sorted((x[1] for x in finished[b]), reverse=True)[k - 1] >= best_alive[b]:
                    done[b] = True
            if any(done):
                scores = scores.masked_fill(torch.tensor(done, device=dev)[:, None], NEG)
            if all(done) or max(best_alive) == NEG:
                break
        out = []
        for b in range(B):
            hyps = finished[b]
            if not hyps:
                hyps = [(ys[b * k][1:].tolist(), scores[b, 0].item(), scores[b, 0].item())]
            hyps.sort(key=lambda x: -x[2])
            out.append([(h[0], h[1]) for h in hyps[:k]])
        return out


def save(path: str, model: Seq2Seq, src_vocab: Vocab, tgt_vocab: Vocab, extra: dict | None = None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "src_vocab": src_vocab.to_json(),
                "tgt_vocab": tgt_vocab.to_json(), "extra": extra or {}}, path)


def load(path: str, device: str = "cpu"):
    """A model from a training checkpoint (`.pt`, a pickle: load only files you made) or from a released
    model folder (`config.json` + `model.safetensors`, local or a Hugging Face repo id; see load_pretrained)."""
    if Path(path).suffix not in (".pt", ".pth"):
        return load_pretrained(path, device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Seq2Seq(ModelConfig(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    return model, Vocab(ck["src_vocab"]), Vocab(ck["tgt_vocab"]), ck.get("extra", {})


# ---------------------------------------------------------------------------------------------
# Released model folders, in the Hugging Face transformers format (format_version 2):
#   config.json, generation_config.json, model.safetensors, tokenizer.json, tokenizer_config.json.
# Seq2Seq is exactly transformers' BlenderbotForConditionalGeneration (pre-LN encoder-decoder, learned
# positions from 0, embeddings scaled by sqrt(d), final LayerNorms, no embedding LayerNorm), so the folder
# loads with AutoModelForSeq2SeqLM / AutoTokenizer and no custom code. Blenderbot has one vocabulary for
# both sides, so the two are stacked: ids 0..T-1 are the target (Khmer side) vocabulary, ids T+j the source
# vocabulary (id T+0 is never used; padding is id 0 on both sides, whose embedding rows are zero in every
# checkpoint). The output layer covers only target ids; source ids get a fixed logit bias of -1e4.
# samdei_type reads the same files without transformers (load_pretrained maps the weights back).
# format_version 1 folders (model_type "samdei-type", samdei_type's own weight names) still load.
# ---------------------------------------------------------------------------------------------
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
GENERATION_CONFIG_NAME = "generation_config.json"
TOKENIZER_NAME = "tokenizer.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
MODEL_TYPE = "samdei-type"            # key of samdei_type's metadata block in config.json (format 1: model_type)
HF_MODEL_TYPE = "blenderbot"
HF_ARCHITECTURE = "BlenderbotForConditionalGeneration"
FORMAT_VERSION = 2
TGT_MARK = ""                   # prefix of target token names in tokenizer.json, removed when decoding
SRC_SPECIALS = ["<src_pad>", "<src_s>", "</src>", "<src_unk>"]
MASKED_LOGIT = -1e4
MAX_INPUT_CHARS = 249                 # inputs are cut to this length; with </src>, 250 encoder positions


def _layer_pairs(cfg: ModelConfig) -> list[tuple[str, str, str]]:
    """(samdei_type name, transformers name, kind): kind 'copy', or 'qkv' (packed in_proj -> q/k/v_proj)."""
    out = []
    for i in range(cfg.enc_layers):
        a, b = f"encoder.layers.{i}.", f"model.encoder.layers.{i}."
        out += [(a + "self_attn.in_proj", b + "self_attn.", "qkv"),
                (a + "self_attn.out_proj", b + "self_attn.out_proj", "copy"),
                (a + "norm1", b + "self_attn_layer_norm", "copy"), (a + "norm2", b + "final_layer_norm", "copy"),
                (a + "linear1", b + "fc1", "copy"), (a + "linear2", b + "fc2", "copy")]
    for i in range(cfg.dec_layers):
        a, b = f"decoder.layers.{i}.", f"model.decoder.layers.{i}."
        out += [(a + "self_attn.in_proj", b + "self_attn.", "qkv"),
                (a + "self_attn.out_proj", b + "self_attn.out_proj", "copy"),
                (a + "multihead_attn.in_proj", b + "encoder_attn.", "qkv"),
                (a + "multihead_attn.out_proj", b + "encoder_attn.out_proj", "copy"),
                (a + "norm1", b + "self_attn_layer_norm", "copy"), (a + "norm2", b + "encoder_attn_layer_norm", "copy"),
                (a + "norm3", b + "final_layer_norm", "copy"),
                (a + "linear1", b + "fc1", "copy"), (a + "linear2", b + "fc2", "copy")]
    out += [("encoder.norm", "model.encoder.layer_norm", "copy"), ("decoder.norm", "model.decoder.layer_norm", "copy")]
    return out


def to_hf_state(model: Seq2Seq) -> dict[str, torch.Tensor]:
    """samdei_type weights -> BlenderbotForConditionalGeneration weights (fp32)."""
    cfg = model.cfg
    sd = {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()}
    T, S, d = cfg.tgt_vocab, cfg.src_vocab, cfg.d_model
    shared = torch.cat([sd["tgt_emb.weight"], sd["src_emb.weight"]], 0)
    head = torch.cat([sd["out.weight"], torch.zeros(S, d)], 0)
    bias = torch.cat([sd["out.bias"], torch.full((S,), MASKED_LOGIT)], 0)[None]
    out = {"model.shared.weight": shared, "model.encoder.embed_tokens.weight": shared.clone(),
           "model.decoder.embed_tokens.weight": shared.clone(),
           "model.encoder.embed_positions.weight": sd["src_pos.weight"],
           "model.decoder.embed_positions.weight": sd["tgt_pos.weight"],
           "lm_head.weight": head, "final_logits_bias": bias}
    for a, b, kind in _layer_pairs(cfg):
        if kind == "qkv":
            for part, w, bb in zip("qkv", sd[a + "_weight"].chunk(3, 0), sd[a + "_bias"].chunk(3, 0)):
                out[f"{b}{part}_proj.weight"], out[f"{b}{part}_proj.bias"] = w, bb
        else:
            out[b + ".weight"], out[b + ".bias"] = sd[a + ".weight"], sd[a + ".bias"]
    return {k: v.contiguous() for k, v in out.items()}


def from_hf_state(cfg: ModelConfig, hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """BlenderbotForConditionalGeneration weights (as written by to_hf_state) -> samdei_type weights."""
    T, S = cfg.tgt_vocab, cfg.src_vocab
    shared = hf["model.shared.weight"]
    out = {"tgt_emb.weight": shared[:T], "src_emb.weight": shared[T:T + S],
           "src_pos.weight": hf["model.encoder.embed_positions.weight"],
           "tgt_pos.weight": hf["model.decoder.embed_positions.weight"],
           "out.weight": hf["lm_head.weight"][:T], "out.bias": hf["final_logits_bias"][0, :T]}
    for a, b, kind in _layer_pairs(cfg):
        if kind == "qkv":
            out[a + "_weight"] = torch.cat([hf[f"{b}{p}_proj.weight"] for p in "qkv"], 0)
            out[a + "_bias"] = torch.cat([hf[f"{b}{p}_proj.bias"] for p in "qkv"], 0)
        else:
            out[a + ".weight"], out[a + ".bias"] = hf[b + ".weight"], hf[b + ".bias"]
    return {k: v.clone().contiguous() for k, v in out.items()}


def hf_config(model: Seq2Seq, src_vocab: Vocab, tgt_vocab: Vocab, training: dict | None = None) -> dict:
    c, T = model.cfg, len(tgt_vocab)
    return {
        "architectures": [HF_ARCHITECTURE], "model_type": HF_MODEL_TYPE,
        "vocab_size": T + len(src_vocab), "d_model": c.d_model,
        "encoder_layers": c.enc_layers, "decoder_layers": c.dec_layers,
        "encoder_attention_heads": c.nhead, "decoder_attention_heads": c.nhead,
        "encoder_ffn_dim": c.ff, "decoder_ffn_dim": c.ff, "activation_function": "gelu",
        "max_position_embeddings": c.max_len, "scale_embedding": True,
        "dropout": c.dropout, "attention_dropout": c.dropout, "activation_dropout": c.dropout,
        "encoder_layerdrop": 0.0, "decoder_layerdrop": 0.0, "init_std": 0.02,
        "pad_token_id": PAD, "bos_token_id": BOS, "eos_token_id": EOS, "decoder_start_token_id": BOS,
        "forced_eos_token_id": None, "encoder_no_repeat_ngram_size": 0,
        "is_encoder_decoder": True, "tie_word_embeddings": False, "use_cache": True,
        "torch_dtype": "float32",
        MODEL_TYPE.replace("-", "_"): {
            "format_version": FORMAT_VERSION, "special_tokens": SPECIALS, "source_special_tokens": SRC_SPECIALS,
            "source_offset": T, "source": src_vocab.to_json(), "target": tgt_vocab.to_json(),
            "max_input_chars": MAX_INPUT_CHARS, "training": training or {}},
    }


def hf_generation_config() -> dict:
    """Beam 5, as the released models are evaluated (without the LM rerank, which is not part of transformers)."""
    return {"decoder_start_token_id": BOS, "bos_token_id": BOS, "eos_token_id": EOS, "pad_token_id": PAD,
            "forced_eos_token_id": None, "num_beams": 5, "max_length": 201, "length_penalty": 1.0}


def hf_tokenizer(src_vocab: Vocab, tgt_vocab: Vocab) -> tuple[dict, dict]:
    """(tokenizer.json, tokenizer_config.json) for a character-level PreTrainedTokenizerFast: encoding maps each
    character of the input to its source id and appends </src>; decoding maps target ids back to characters."""
    T = len(tgt_vocab)
    names = list(SPECIALS) + [TGT_MARK + ch for ch in tgt_vocab.itos[len(SPECIALS):]]
    names += SRC_SPECIALS + src_vocab.itos[len(SPECIALS):]
    assert len(set(names)) == len(names), "token names collide"
    vocab = {n: i for i, n in enumerate(names)}
    special_ids = list(range(len(SPECIALS))) + [T + j for j in range(len(SRC_SPECIALS))]
    added = [{"id": i, "content": names[i], "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i in special_ids]
    eos_src = SRC_SPECIALS[EOS]
    tok = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "[\\s\\S]"}, "behavior": "Isolated", "invert": False},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": eos_src, "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": eos_src, "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": eos_src, "type_id": 1}}],
            "special_tokens": {eos_src: {"id": eos_src, "ids": [T + EOS], "tokens": [eos_src]}}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": TGT_MARK}, "content": ""}, {"type": "Fuse"}]},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": SRC_SPECIALS[UNK]},
    }
    tok_cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": MAX_INPUT_CHARS + 1,
               "padding_side": "right", "truncation_side": "right", "clean_up_tokenization_spaces": False,
               "bos_token": SPECIALS[BOS], "eos_token": SPECIALS[EOS], "pad_token": SPECIALS[PAD],
               "unk_token": SRC_SPECIALS[UNK], "model_input_names": ["input_ids", "attention_mask"]}
    return tok, tok_cfg


def save_pretrained(out_dir: str | Path, model: Seq2Seq, src_vocab: Vocab, tgt_vocab: Vocab,
                    training: dict | None = None) -> None:
    """Write a transformers-format model folder (fp32). `training` is stored as is in config.json under
    "samdei_type" (keep it to public facts, e.g. {"step": 3000})."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, obj):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    dump(CONFIG_NAME, hf_config(model, src_vocab, tgt_vocab, training))
    dump(GENERATION_CONFIG_NAME, hf_generation_config())
    tok, tok_cfg = hf_tokenizer(src_vocab, tgt_vocab)
    dump(TOKENIZER_NAME, tok)
    dump(TOKENIZER_CONFIG_NAME, tok_cfg)
    save_file(to_hf_state(model), str(out / WEIGHTS_NAME), metadata={"format": "pt"})


def load_pretrained(path_or_repo: str | Path, device: str = "cpu", subfolder: str | None = None,
                    revision: str | None = None):
    """A released model: a local folder (or its config.json / model.safetensors) or a Hugging Face repo id,
    e.g. load_pretrained("samdei/samdei-type-base") or load_pretrained("samdei/samdei-type-base", subfolder="channel").
    Only that folder's config.json and model.safetensors are downloaded. Returns
    (model, src_vocab, tgt_vocab, training_info)."""
    from safetensors.torch import load_file
    d = Path(path_or_repo)
    if d.suffix in (".json", ".safetensors"):
        d = d.parent
    if subfolder:
        d = d / subfolder
    if not (d / CONFIG_NAME).is_file():
        from huggingface_hub import snapshot_download
        pre = f"{subfolder.strip('/')}/" if subfolder else ""
        root = snapshot_download(str(path_or_repo), revision=revision,
                                 allow_patterns=[pre + CONFIG_NAME, pre + WEIGHTS_NAME])
        d = Path(root) / pre
    cfg = json.loads((d / CONFIG_NAME).read_text(encoding="utf-8"))
    meta = cfg.get(MODEL_TYPE.replace("-", "_"))
    if cfg.get("model_type") == HF_MODEL_TYPE and meta and meta.get("format_version") == FORMAT_VERSION:
        if meta["special_tokens"] != SPECIALS or meta["source_offset"] != len(meta["target"]) + len(SPECIALS):
            raise ValueError(f"{d / CONFIG_NAME}: unexpected vocabulary layout")
        sv, tv = Vocab(meta["source"]), Vocab(meta["target"])
        mc = ModelConfig(src_vocab=len(sv), tgt_vocab=len(tv), d_model=cfg["d_model"],
                         nhead=cfg["encoder_attention_heads"], enc_layers=cfg["encoder_layers"],
                         dec_layers=cfg["decoder_layers"], ff=cfg["encoder_ffn_dim"], dropout=cfg["dropout"],
                         max_len=cfg["max_position_embeddings"])
        model = Seq2Seq(mc)
        model.load_state_dict(from_hf_state(mc, load_file(str(d / WEIGHTS_NAME), device="cpu")))
        training = meta.get("training", {})
    elif cfg.get("model_type") == MODEL_TYPE:  # format_version 1
        v = cfg["vocab"]
        if v["special_tokens"] != SPECIALS:
            raise ValueError(f"{d / CONFIG_NAME}: unexpected special tokens {v['special_tokens']}")
        model = Seq2Seq(ModelConfig(**{f.name: cfg[f.name] for f in fields(ModelConfig) if f.name in cfg}))
        model.load_state_dict(load_file(str(d / WEIGHTS_NAME), device="cpu"))
        sv, tv, training = Vocab(v["source"]), Vocab(v["target"]), cfg.get("training", {})
    else:
        raise ValueError(f"{d / CONFIG_NAME}: not a Samdei Type model (model_type={cfg.get('model_type')!r})")
    model.to(device).eval()
    return model, sv, tv, training


# ---------------------------------------------------------------------------------------------
# Incremental (KV-cached) decoding. Re-implements the pre-LN nn.TransformerDecoderLayer forward with
# the same weights so each step costs O(t) instead of re-running the whole prefix (phone latency).
# ---------------------------------------------------------------------------------------------

def _heads(x: torch.Tensor, h: int) -> torch.Tensor:  # (B, T, D) -> (B, h, T, D/h)
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)


def cross_proj(layer: nn.Module, d: int):
    """Cross-attention (q, k, v) projections as ((W, b), ...). Uses the pre-split copies made by
    split_cross_proj when present: in ONNX, slicing the packed in_proj_weight at run time leaves it fp32
    (dynamic int8 quantization only touches initializers that feed MatMul directly)."""
    if hasattr(layer, "xq_w"):
        return (layer.xq_w, layer.xq_b), (layer.xk_w, layer.xk_b), (layer.xv_w, layer.xv_b)
    W, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
    return (W[:d], b[:d]), (W[d:2 * d], b[d:2 * d]), (W[2 * d:], b[2 * d:])


def split_cross_proj(model: Seq2Seq) -> Seq2Seq:
    """Store each decoder layer's cross-attention q/k/v weights as separate buffers (for export)."""
    d = model.cfg.d_model
    for layer in model.decoder.layers:
        W, b = layer.multihead_attn.in_proj_weight.detach(), layer.multihead_attn.in_proj_bias.detach()
        for i, n in enumerate("qkv"):
            layer.register_buffer(f"x{n}_w", W[i * d:(i + 1) * d].clone(), persistent=False)
            layer.register_buffer(f"x{n}_b", b[i * d:(i + 1) * d].clone(), persistent=False)
    return model


def cross_kv(model: Seq2Seq, memory: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per decoder layer: cross-attention K, V from the encoder memory (computed once)."""
    d, h = model.cfg.d_model, model.cfg.nhead
    out = []
    for layer in model.decoder.layers:
        _, (wk, bk), (wv, bv) = cross_proj(layer, d)
        out.append((_heads(F.linear(memory, wk, bk), h), _heads(F.linear(memory, wv, bv), h)))
    return out


def decode_step(model: Seq2Seq, tok: torch.Tensor, pos: int, cache: list, xkv: list,
                mem_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, list]:
    """One decoder step. tok: (B,) ids at position `pos`; cache: per layer (K, V) of previous steps or None.
    mem_bias: (B, 1, 1, S) additive mask for padded memory (or None). Returns (logits (B, V), new cache)."""
    d, h = model.cfg.d_model, model.cfg.nhead
    x = model.tgt_emb(tok)[:, None] * model.scale + model.tgt_pos.weight[pos][None, None]
    new_cache = []
    for li, layer in enumerate(model.decoder.layers):
        # self-attention over cached keys/values
        sa = layer.self_attn
        q, k, v = F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
        q, k, v = _heads(q, h), _heads(k, h), _heads(v, h)
        if cache[li] is not None:
            k = torch.cat([cache[li][0], k], 2); v = torch.cat([cache[li][1], v], 2)
        new_cache.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + sa.out_proj(a.transpose(1, 2).reshape(x.size(0), 1, d))
        # cross-attention
        ca = layer.multihead_attn
        q = _heads(F.linear(layer.norm2(x), *cross_proj(layer, d)[0]), h)
        a = F.scaled_dot_product_attention(q, xkv[li][0], xkv[li][1], attn_mask=mem_bias)
        x = x + ca.out_proj(a.transpose(1, 2).reshape(x.size(0), 1, d))
        # feed-forward
        x = x + layer.linear2(F.gelu(layer.linear1(layer.norm3(x))))
    x = model.decoder.norm(x)
    return model.out(x[:, 0]), new_cache


@torch.no_grad()
def greedy_cached(model: Seq2Seq, src: torch.Tensor, max_len: int = 200) -> list[list[int]]:
    memory, mask = model.encode(src)
    xkv = cross_kv(model, memory)
    mem_bias = torch.zeros(mask.shape, dtype=memory.dtype, device=src.device).masked_fill(mask, float("-inf"))
    mem_bias = mem_bias[:, None, None, :]
    B = src.size(0)
    tok = torch.full((B,), BOS, dtype=torch.long, device=src.device)
    cache = [None] * len(model.decoder.layers)
    out = torch.zeros(B, 0, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for pos in range(max_len):
        logits, cache = decode_step(model, tok, pos, cache, xkv, mem_bias)
        tok = logits.argmax(-1)
        tok = torch.where(done, torch.full_like(tok, PAD), tok)
        out = torch.cat([out, tok[:, None]], 1)
        done |= tok == EOS
        if bool(done.all()):
            break
    return out.tolist()


@torch.no_grad()
def beam_cached(model: Seq2Seq, src: torch.Tensor, k: int = 5, max_len: int = 200, len_pen: float = 1.0):
    """Same search as Seq2Seq.beam (same outputs), but with KV-cached decoder steps: each step feeds only
    the newest token of each of the B*k rows and reorders the self-attention cache by the chosen parents."""
    B = src.size(0)
    NEG = float("-inf")
    dev = src.device
    memory, mask = model.encode(src)
    xkv = [(xk.repeat_interleave(k, 0), xv.repeat_interleave(k, 0)) for xk, xv in cross_kv(model, memory)]
    mem_bias = torch.zeros(mask.shape, dtype=memory.dtype, device=dev).masked_fill(mask, float("-inf"))
    mem_bias = mem_bias.repeat_interleave(k, 0)[:, None, None, :]
    ys = torch.full((B * k, 1), BOS, dtype=torch.long, device=dev)
    cache = [None] * len(model.decoder.layers)
    scores = torch.full((B, k), NEG, device=dev)
    scores[:, 0] = 0.0
    base = (torch.arange(B, device=dev) * k)[:, None]
    finished: list[list[tuple[list[int], float, float]]] = [[] for _ in range(B)]
    done = [False] * B
    for step in range(max_len):
        logits, cache = decode_step(model, ys[:, -1], step, cache, xkv, mem_bias)
        logp = F.log_softmax(logits.float(), -1)  # (B*k, V)
        V = logp.size(-1)
        cand = (scores.view(-1, 1) + logp).view(B, k * V)
        top, idx = cand.topk(2 * k, dim=-1)
        beam_idx, tok = idx // V, idx % V
        is_eos = tok == EOS
        fin = (is_eos & (top > NEG)).nonzero().tolist()
        if fin:
            rows = torch.tensor([b * k + int(beam_idx[b, j]) for b, j in fin], device=dev)
            prev = ys[rows, 1:].tolist()
            sc = top[[b for b, _ in fin], [j for _, j in fin]].tolist()
            L = ys.size(1)
            for (b, _), p, s in zip(fin, prev, sc):
                if not done[b]:
                    finished[b].append((p, s, s / (L ** len_pen)))
        sel_sc, sel_j = top.masked_fill(is_eos, NEG).topk(k, dim=-1)
        rows = (base + beam_idx.gather(1, sel_j)).view(-1)
        ys = torch.cat([ys[rows], tok.gather(1, sel_j).view(-1, 1)], 1)
        cache = [(kc[rows], vc[rows]) for kc, vc in cache]
        scores = sel_sc
        best_alive = scores.max(-1).values.tolist()
        for b in range(B):
            if not done[b] and len(finished[b]) >= k and \
                    sorted((x[1] for x in finished[b]), reverse=True)[k - 1] >= best_alive[b]:
                done[b] = True
        if any(done):
            scores = scores.masked_fill(torch.tensor(done, device=dev)[:, None], NEG)
        if all(done) or max(best_alive) == NEG:
            break
    out = []
    for b in range(B):
        hyps = finished[b]
        if not hyps:
            hyps = [(ys[b * k][1:].tolist(), scores[b, 0].item(), scores[b, 0].item())]
        hyps.sort(key=lambda x: -x[2])
        out.append([(h[0], h[1]) for h in hyps[:k]])
    return out
