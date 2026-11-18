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

from samdei_type.evaluation import canon, chunk_source, score
from samdei_type.lm import BOS_CH, EOS_CH, CharLM, WordLM
from samdei_type.model import ModelConfig, Seq2Seq, beam_cached, greedy_cached
from samdei_type.text import normalize_khmer
from samdei_type.vocab import BOS, EOS, Vocab


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


@unittest.skipUnless(os.environ.get("SAMDEI_TYPE_MODEL"), "set SAMDEI_TYPE_MODEL to a released model")
class TestReleasedModel(unittest.TestCase):
    def test_convert(self):
        from samdei_type.pipeline import Converter
        conv = Converter(os.environ["SAMDEI_TYPE_MODEL"])
        self.assertEqual(canon(conv.convert("knhom jong tov phsar")), canon("ខ្ញុំចង់ទៅផ្សារ"))


if __name__ == "__main__":
    unittest.main()
