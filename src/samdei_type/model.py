"""The model: a character-level Transformer encoder-decoder, loading a released model folder, and
KV-cached greedy and beam search."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .vocab import BOS, EOS, PAD, REV_TAG, SPECIALS, TaggedVocab, Vocab


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
    """Pre-LN encoder-decoder with GELU, learned positions and embeddings scaled by sqrt(d_model)."""

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

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor):
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask)


# A released model folder is in the transformers format of BlenderbotForConditionalGeneration, which has the
# same architecture: config.json, model.safetensors and tokenizer files. Blenderbot has one vocabulary, so the
# two are stacked: ids 0..T-1 are the target (Khmer) vocabulary and ids T+j the source vocabulary. config.json
# keeps both character lists under "samdei_type". load() maps the weights back without needing transformers.

def _layer_pairs(cfg: ModelConfig) -> list[tuple[str, str, str]]:
    """(Seq2Seq name, Blenderbot name, kind): kind is 'copy', or 'qkv' for a packed in_proj split into q/k/v_proj."""
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


def _from_hf_state(cfg: ModelConfig, hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    T, S = cfg.tgt_vocab, cfg.src_vocab
    # transformers 5 trains the encoder's and the decoder's embedding tables, not model.shared, so a model
    # fine-tuned there is read from those two tables
    enc = hf.get("model.encoder.embed_tokens.weight", hf["model.shared.weight"])
    dec = hf.get("model.decoder.embed_tokens.weight", hf["model.shared.weight"])
    out = {"tgt_emb.weight": dec[:T], "src_emb.weight": enc[T:T + S],
           "src_pos.weight": hf["model.encoder.embed_positions.weight"],
           "tgt_pos.weight": hf["model.decoder.embed_positions.weight"],
           "out.weight": hf["lm_head.weight"][:T], "out.bias": hf["final_logits_bias"][0, :T]}
    for a, b, kind in _layer_pairs(cfg):
        if kind == "qkv":
            out[a + "_weight"] = torch.cat([hf[f"{b}{p}_proj.weight"] for p in "qkv"], 0)
            out[a + "_bias"] = torch.cat([hf[f"{b}{p}_proj.bias"] for p in "qkv"], 0)
        else:
            out[a + ".weight"], out[a + ".bias"] = hf[b + ".weight"], hf[b + ".bias"]
    return out


def load(folder: str | Path, device: str = "cpu") -> tuple[Seq2Seq, Vocab, Vocab]:
    """A released model folder (config.json and model.safetensors). Returns (model, source vocab, target vocab)."""
    d = Path(folder)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    meta = cfg.get("samdei_type") or {}
    if cfg.get("model_type") != "blenderbot" or meta.get("format_version") != 2:
        raise ValueError(f"{d}: not a Samdei Type model folder")
    if meta["special_tokens"] != SPECIALS or meta["source_offset"] != len(SPECIALS) + len(meta["target"]):
        raise ValueError(f"{d}: unexpected vocabulary layout")
    sv, tv = Vocab(meta["source"]), Vocab(meta["target"])
    mc = ModelConfig(src_vocab=len(sv), tgt_vocab=len(tv), d_model=cfg["d_model"],
                     nhead=cfg["encoder_attention_heads"], enc_layers=cfg["encoder_layers"],
                     dec_layers=cfg["decoder_layers"], ff=cfg["encoder_ffn_dim"], dropout=cfg["dropout"],
                     max_len=cfg["max_position_embeddings"])
    model = Seq2Seq(mc)
    model.load_state_dict(_from_hf_state(mc, load_file(str(d / "model.safetensors"))))
    return model.to(device).eval(), sv, tv


def two_way(sv: Vocab) -> bool:
    """Whether a model with source vocabulary sv also converts Khmer -> Latin (inputs that start with REV_TAG)."""
    return REV_TAG in sv.stoi


def channel_view(model: Seq2Seq, sv: Vocab, tv: Vocab) -> tuple[Seq2Seq, Vocab, Vocab]:
    """(model, sv, tv) to score log P(romanized | Khmer) with: a two-way model reads its Khmer input after
    REV_TAG; a channel model (Khmer -> Latin only) is used as it is."""
    return model, (TaggedVocab(sv) if two_way(sv) else sv), tv


# KV-cached decoding: the forward pass of the pre-LN nn.TransformerDecoderLayer, written out with the same
# weights, so that each step only computes the newest position.

def _heads(x: torch.Tensor, h: int) -> torch.Tensor:  # (B, T, D) -> (B, h, T, D/h)
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)


def _memory_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, S) padding mask -> additive attention bias (B, 1, 1, S)."""
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(mask, float("-inf"))[:, None, None]


def cross_kv(model: Seq2Seq, memory: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per decoder layer, the cross-attention keys and values of the encoder output (computed once per input)."""
    d, h = model.cfg.d_model, model.cfg.nhead
    out = []
    for layer in model.decoder.layers:
        W, b = layer.multihead_attn.in_proj_weight, layer.multihead_attn.in_proj_bias
        k = F.linear(memory, W[d:2 * d], b[d:2 * d])
        v = F.linear(memory, W[2 * d:], b[2 * d:])
        out.append((_heads(k, h), _heads(v, h)))
    return out


def decode_step(model: Seq2Seq, tok: torch.Tensor, pos: int, cache: list, xkv: list,
                mem_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, list]:
    """One decoder step for the tokens tok (B,) at position pos. cache: per layer the self-attention (K, V) of
    the earlier steps, or None at the first step. Returns (logits (B, V), new cache)."""
    d, h = model.cfg.d_model, model.cfg.nhead
    x = model.tgt_emb(tok)[:, None] * model.scale + model.tgt_pos.weight[pos][None, None]
    new_cache = []
    for li, layer in enumerate(model.decoder.layers):
        sa = layer.self_attn
        q, k, v = F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
        q, k, v = _heads(q, h), _heads(k, h), _heads(v, h)
        if cache[li] is not None:
            k = torch.cat([cache[li][0], k], 2)
            v = torch.cat([cache[li][1], v], 2)
        new_cache.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + sa.out_proj(a.transpose(1, 2).reshape(x.size(0), 1, d))

        ca = layer.multihead_attn
        q = _heads(F.linear(layer.norm2(x), ca.in_proj_weight[:d], ca.in_proj_bias[:d]), h)
        a = F.scaled_dot_product_attention(q, xkv[li][0], xkv[li][1], attn_mask=mem_bias)
        x = x + ca.out_proj(a.transpose(1, 2).reshape(x.size(0), 1, d))

        x = x + layer.linear2(F.gelu(layer.linear1(layer.norm3(x))))
    x = model.decoder.norm(x)
    return model.out(x[:, 0]), new_cache


@torch.no_grad()
def greedy_cached(model: Seq2Seq, src: torch.Tensor, max_len: int = 200) -> list[list[int]]:
    """Greedy decoding of a batch of source ids (B, S). Returns the output ids per input (padded after </s>)."""
    memory, mask = model.encode(src)
    xkv = cross_kv(model, memory)
    mem_bias = _memory_bias(mask, memory.dtype)
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
def beam_cached(model: Seq2Seq, src: torch.Tensor, k: int = 5, max_len: int = 200,
                len_pen: float = 1.0) -> list[list[tuple[list[int], float]]]:
    """Batched beam search over source ids (B, S). Each step feeds the newest token of the B*k beams and
    reorders the self-attention cache by their parents. Returns per input the n-best [(ids, log probability)],
    ranked by log probability / length ** len_pen."""
    B = src.size(0)
    NEG = float("-inf")
    dev = src.device
    memory, mask = model.encode(src)
    xkv = [(xk.repeat_interleave(k, 0), xv.repeat_interleave(k, 0)) for xk, xv in cross_kv(model, memory)]
    mem_bias = _memory_bias(mask, memory.dtype).repeat_interleave(k, 0)
    ys = torch.full((B * k, 1), BOS, dtype=torch.long, device=dev)
    cache = [None] * len(model.decoder.layers)
    scores = torch.full((B, k), NEG, device=dev)
    scores[:, 0] = 0.0
    base = (torch.arange(B, device=dev) * k)[:, None]
    finished: list[list[tuple[list[int], float, float]]] = [[] for _ in range(B)]  # (ids, logprob, ranking score)
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
        # keep the k best continuations that do not end the sequence
        sel_sc, sel_j = top.masked_fill(is_eos, NEG).topk(k, dim=-1)
        rows = (base + beam_idx.gather(1, sel_j)).view(-1)
        ys = torch.cat([ys[rows], tok.gather(1, sel_j).view(-1, 1)], 1)
        cache = [(kc[rows], vc[rows]) for kc, vc in cache]
        scores = sel_sc
        # an input is done once it has k finished hypotheses no worse than its best live beam
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
        hyps = finished[b] or [(ys[b * k][1:].tolist(), scores[b, 0].item(), scores[b, 0].item())]
        hyps.sort(key=lambda x: -x[2])
        out.append([(h[0], h[1]) for h in hyps[:k]])
    return out
