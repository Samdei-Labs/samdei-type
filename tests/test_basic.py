"""Offline tests: no model download. Run with `python -m unittest discover -s tests` (or pytest).

Set SAMDEI_TYPE_MODEL to a released model folder (or Hub repo id) to also run the end-to-end test.
"""
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from samdei_type.evaluation import canon, chunk_khmer, chunk_source, score
from samdei_type.lm import BOS_CH, EOS_CH, CharLM, WordLM
from samdei_type.model import ModelConfig, Seq2Seq, beam_cached, channel_view, greedy_cached, two_way
from samdei_type.text import drop_commas, normalize_khmer
from samdei_type.vocab import BOS, EOS, REV_TAG, TaggedVocab, Vocab


class TestText(unittest.TestCase):
    def test_chunk_source(self):
        self.assertEqual(chunk_source("a b c d e f g h i j", 8), ["a b c d e f g h", "i j"])
        self.assertEqual(chunk_source("bong sok sabay te? knhom sok sabay", 8),
                         ["bong sok sabay te?", "knhom sok sabay"])
        self.assertEqual(chunk_source("", 8), [""])

    def test_canon_and_score(self):
        self.assertEqual(canon("ក្ដៅ"), canon("ក្តៅ"))   # coeng DA = coeng TA
        self.assertEqual(canon("ឲ្យ"), canon("ឱ្យ"))     # the two QOO spellings
        s = score(["ខ្ញុំ ចង់", "ទៅ"], ["ខ្ញុំចង់", "ផ្សារ"])
        self.assertEqual((s["n"], s["exact"]), (2, 0.5))

    def test_normalize(self):
        for t in ["ខ្ញុំចង់ទៅផ្សារ", "សួស្ដី\u200bបង", "១២៣"]:
            n = normalize_khmer(t)
            self.assertEqual(normalize_khmer(n), n)
        self.assertEqual(normalize_khmer("១២៣"), "123")
        self.assertEqual(normalize_khmer("សួស្ដី\u200bបង"), normalize_khmer("សួស្ដីបង"))  # zero-width space removed

    def test_drop_commas(self):
        self.assertEqual(drop_commas("បើបងយល់ព្រម, ខ្ញុំទៅ"), "បើបងយល់ព្រម ខ្ញុំទៅ")
        self.assertEqual(drop_commas("ថ្លៃ 5,000 រៀល, ណា"), "ថ្លៃ 5,000 រៀល ណា")   # a comma between digits stays
        self.assertEqual(drop_commas("a ,b,,c,"), "a b c")

    def test_chunk_khmer(self):
        text = "ខ្ញុំចង់ទៅផ្សារ ហើយបងទៅណា"
        pieces = chunk_khmer(text, 2)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("".join(pieces).replace(" ", ""), text.replace(" ", ""))
        self.assertEqual(chunk_khmer(text, 100), [text])


class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sv = Vocab(list("abcdefghijklmnopqrstuvwxyz "))
        self.tv = Vocab([chr(c) for c in range(0x1780, 0x17A3)] + [" "])
        cfg = ModelConfig(len(self.sv), len(self.tv), d_model=32, nhead=4, enc_layers=2, dec_layers=2, ff=64,
                          dropout=0.0, max_len=64)
        self.model = Seq2Seq(cfg).eval()
        self.src = torch.tensor([self.sv.encode("knhom jong tov"), self.sv.encode("bay") + [0] * 11])

    def test_vocab_roundtrip(self):
        self.assertEqual(self.sv.decode(self.sv.encode("knhom")), "knhom")

    def test_two_way_vocab(self):
        self.assertFalse(two_way(self.sv))
        self.assertIs(channel_view(self.model, self.sv, self.tv)[1], self.sv)   # a channel model is used as it is
        sv2 = Vocab(self.sv.itos[4:] + [REV_TAG])
        self.assertTrue(two_way(sv2))
        tagged = channel_view(self.model, sv2, self.tv)[1]
        self.assertIsInstance(tagged, TaggedVocab)
        self.assertEqual(tagged.encode("ab"), sv2.encode(REV_TAG + "ab"))

    @torch.no_grad()
    def test_greedy_matches_full_decoder(self):
        """The KV-cached steps give the same tokens as re-running the full decoder on the prefix."""
        ys = torch.full((2, 1), BOS)
        for _ in range(10):
            ys = torch.cat([ys, self.model(self.src, ys)[:, -1].argmax(-1, keepdim=True)], 1)
        cached = greedy_cached(self.model, self.src, max_len=10)
        for row, full in zip(cached, ys[:, 1:].tolist()):
            n = full.index(EOS) + 1 if EOS in full else len(full)
            self.assertEqual(row[:n], full[:n])

    @torch.no_grad()
    def test_beam_scores_are_logprobs(self):
        """Each beam hypothesis's score is its log probability under the full decoder."""
        for b, hyps in enumerate(beam_cached(self.model, self.src, k=3, max_len=12)):
            self.assertTrue(hyps)
            for ids, s in hyps:
                y = torch.tensor([[BOS] + ids + ([EOS] if len(ids) < 12 else [])])
                logp = F.log_softmax(self.model(self.src[b:b + 1], y[:, :-1]), -1)
                self.assertAlmostEqual(logp.gather(-1, y[:, 1:, None]).sum().item(), s, places=3)


class TestLM(unittest.TestCase):
    lines = ["ខ្ញុំចង់ទៅផ្សារ", "ខ្ញុំទៅសាលារៀន", "បងសុខសប្បាយទេ"]

    def char_lm(self, order=4, size=1 << 12):
        lm = CharLM([np.zeros(size, np.int32) for _ in range(order)], 0)
        for line in self.lines:
            s = BOS_CH + line + EOS_CH
            lm.total += len(s)
            for n in range(1, order + 1):
                for i in range(len(s) - n + 1):
                    lm.counts[n - 1][lm._index(s[i:i + n])] += 1
        return lm

    def test_charlm(self):
        lm = self.char_lm()
        self.assertGreater(lm.logprob("ខ្ញុំចង់ទៅផ្សារ"), lm.logprob("រាស្ផទៅងច់ញុំខ្"))
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "c.npz")
            np.savez_compressed(p, *lm.counts, meta=np.array([lm.order, 12, lm.total]))
            self.assertEqual(CharLM.load(p).logprob("ខ្ញុំទៅផ្សារ"), lm.logprob("ខ្ញុំទៅផ្សារ"))

    def test_wordlm(self):
        obj = {"format": "samdei-type-wordlm", "version": 1, "total": 8,
               "unigrams": {"<s>": 2, "</s>": 2, "ខ្ញុំ": 2, "ទៅ": 1, "ផ្សារ": 1},
               "bigrams": [["<s>", "ខ្ញុំ", 2], ["ខ្ញុំ", "ទៅ", 1], ["ទៅ", "ផ្សារ", 1], ["ផ្សារ", "</s>", 1]]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.json.gz"
            p.write_bytes(gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8")))
            lm = WordLM.load(str(p))
        seen, oov = lm.logprob("ខ្ញុំទៅផ្សារ")
        self.assertEqual(oov, 0)
        self.assertGreater(seen, lm.logprob("ផ្សារទៅខ្ញុំ")[0])
        self.assertGreaterEqual(lm.logprob("សាលារៀន")[1], 1)


class TestBackend(unittest.TestCase):
    def test_auto_follows_rerank_json(self):
        import importlib.util
        import warnings
        from samdei_type.pipeline import _pick
        has_ort = importlib.util.find_spec("onnxruntime") is not None
        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = Path(d)
            self.assertEqual(_pick(d, "auto"), "onnx")                     # no PyTorch weights
            (d / "config.json").write_text("{}")
            self.assertEqual(_pick(d, "auto"), "torch")                    # no rerank.json: PyTorch weights
            (d / "rerank.json").write_text('{"backend": "onnx"}')
            self.assertEqual(_pick(d, "auto"), "onnx" if has_ort else "torch")
            self.assertEqual(_pick(d, "torch"), "torch")

    def test_needed_files(self):
        from samdei_type.pipeline import _needed
        mini = {"rerank_weights": {"char": 0.0, "word": 0.3, "oov": 1.0, "channel": 0.0},
                "char_lm": None, "word_lm": "lm/w.json.gz", "channel_model": None}
        self.assertEqual(_needed(mini, "onnx", True), ["onnx/encoder_kv.int8.onnx", "onnx/decoder_step.int8.onnx",
                                                       "onnx/vocab.json", "lm/w.json.gz"])
        base = {"rerank_weights": {"char": 0.3, "word": 0.3, "oov": 1.0, "channel": 0.3},
                "char_lm": "lm/c.npz", "word_lm": "lm/w.json.gz", "channel_model": "channel/"}
        self.assertEqual(_needed(base, "torch", True), ["config.json", "model.safetensors", "lm/c.npz",
                                                        "lm/w.json.gz", "channel/config.json",
                                                        "channel/model.safetensors"])
        self.assertEqual(_needed(base, "torch", False), ["config.json", "model.safetensors"])
        base["rerank_weights"]["char"] = 0.0  # a model with weight 0 is not loaded, so not downloaded
        self.assertNotIn("lm/c.npz", _needed(base, "torch", True))
        v1 = {"rerank_weights": {"char": 0.2, "word": 0.3, "oov": 0.0, "channel": 0.5},
              "char_lm": "lm/c.npz", "word_lm": "lm/w.json.gz", "channel_model": None, "channel_self": True}
        self.assertEqual(_needed(v1, "torch", True), ["config.json", "model.safetensors", "lm/c.npz", "lm/w.json.gz"])
        self.assertEqual(_needed(v1, "onnx", True)[-2:], ["config.json", "model.safetensors"])  # the channel's weights


@unittest.skipUnless(os.environ.get("SAMDEI_TYPE_MODEL"), "set SAMDEI_TYPE_MODEL to a released model")
class TestReleasedModel(unittest.TestCase):
    def test_convert(self):
        from samdei_type.pipeline import Converter
        conv = Converter(os.environ["SAMDEI_TYPE_MODEL"])
        self.assertEqual(canon(conv.convert("knhom jong tov phsar")), canon("ខ្ញុំចង់ទៅផ្សារ"))
        self.assertNotIn(",", conv.convert("ot mean wifi te, bong"))


if __name__ == "__main__":
    unittest.main()
