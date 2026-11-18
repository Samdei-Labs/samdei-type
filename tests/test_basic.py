"""Offline tests: no model download. Run with `python -m unittest discover -s tests` (or pytest).

Set SAMDEI_TYPE_MODEL to a released model folder (or Hub repo id) to also run the end-to-end test.
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch

from samdei_type.evaluation import canon, chunk_source, score
from samdei_type.lm import CharLM, WordLM
from samdei_type.model import ModelConfig, Seq2Seq, Vocab, beam_cached, greedy_cached, load, save_pretrained
from samdei_type.text import normalize_khmer


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
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["exact"], 0.5)

    def test_normalize_idempotent(self):
        for t in ["ខ្ញុំចង់ទៅផ្សារ", "សួស្ដី​បង", "១២៣"]:
            n = normalize_khmer(t)
            self.assertEqual(normalize_khmer(n), n)
        self.assertEqual(normalize_khmer("១២៣"), "123")


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

    def test_cached_search_matches_full(self):
        self.assertEqual(greedy_cached(self.model, self.src, max_len=20)[0][:5],
                         self.model.greedy(self.src, max_len=20)[0][:5])
        a = beam_cached(self.model, self.src, k=3, max_len=12)
        b = self.model.beam(self.src, k=3, max_len=12)
        self.assertEqual([[h for h, _ in x] for x in a], [[h for h, _ in x] for x in b])

    def test_save_and_load_pretrained(self):
        with tempfile.TemporaryDirectory() as d:
            save_pretrained(d, self.model, self.sv, self.tv)
            m2, sv2, tv2, _ = load(d)
            self.assertEqual(sv2.itos, self.sv.itos)
            self.assertEqual(tv2.itos, self.tv.itos)
            for k, v in self.model.state_dict().items():
                self.assertTrue(torch.equal(v, m2.state_dict()[k]), k)


class TestLM(unittest.TestCase):
    lines = ["ខ្ញុំចង់ទៅផ្សារ", "ខ្ញុំទៅសាលារៀន", "បងសុខសប្បាយទេ"]

    def test_charlm_prefers_seen_text(self):
        lm = CharLM(order=4, bits=12).fit(self.lines)
        self.assertGreater(lm.logprob("ខ្ញុំចង់ទៅផ្សារ"), lm.logprob("រាស្ផទៅងច់ញុំខ្"))

    def test_wordlm_json_roundtrip(self):
        lm = WordLM().fit(self.lines)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "w.json.gz")
            lm.save(p)
            lm2 = WordLM.load(p)
        self.assertEqual((lm.uni, lm.bi, lm.total), (lm2.uni, lm2.bi, lm2.total))
        self.assertEqual(lm.logprob("ខ្ញុំចង់ទៅផ្សារ"), lm2.logprob("ខ្ញុំចង់ទៅផ្សារ"))


@unittest.skipUnless(os.environ.get("SAMDEI_TYPE_MODEL"), "set SAMDEI_TYPE_MODEL to a released model")
class TestReleasedModel(unittest.TestCase):
    def test_convert(self):
        from samdei_type.pipeline import Converter
        conv = Converter(os.environ["SAMDEI_TYPE_MODEL"])
        self.assertEqual(canon(conv.convert("knhom jong tov phsar")), canon("ខ្ញុំចង់ទៅផ្សារ"))


if __name__ == "__main__":
    unittest.main()
