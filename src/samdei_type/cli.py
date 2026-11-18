"""Command line: romanized Khmer -> Khmer script.

  samdei-type "knhom jong tov phsar"                                 # Samdei Type Mini from the Hub
  samdei-type -m samdei/samdei-type-base "bong nham bay nov?"        # Base, full reranker
  samdei-type -m ./samdei-type-mini --no-rerank "ot mean wifi te"   # a local model folder, beam search only
  echo "saek nh tv rean" | samdei-type -m ./samdei-type-mini -      # one message per input line
  samdei-type -m ./samdei-type-mini --nbest "jam knhom bontech"      # also the n-best list of each piece

Also runs as `python -m samdei_type`.
"""
import argparse
import sys

DEFAULT_MODEL = "samdei/samdei-type-mini"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="samdei-type", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="+", help="romanized message(s), or - to read one message per line from stdin")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help=f"local model folder or Hugging Face repo id (default: {DEFAULT_MODEL})")
    ap.add_argument("--backend", choices=["auto", "torch", "onnx"], default="auto",
                    help="torch: config.json + model.safetensors; onnx: onnx/ int8 graphs (needs onnxruntime); "
                         "auto: as in rerank.json (Mini: onnx, Base: torch)")
    ap.add_argument("--beam", type=int, default=None, help="beam size (default: rerank.json, else 5)")
    ap.add_argument("--no-rerank", action="store_true", help="skip the LM / channel reranking")
    ap.add_argument("--no-chunk", action="store_true", help="decode each message whole, not in 8-word pieces")
    ap.add_argument("--nbest", action="store_true", help="also print each piece's n-best list with scores")
    ap.add_argument("--tsv", action="store_true", help="print 'input<TAB>output' lines")
    ap.add_argument("--device", default="cpu", help="torch device for the torch backend and the channel model")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads (onnx backend)")
    a = ap.parse_args(argv)

    from .pipeline import Converter

    texts = [line.rstrip("\n") for line in sys.stdin] if a.text == ["-"] else a.text
    try:
        conv = Converter(a.model, backend=a.backend, device=a.device, rerank=not a.no_rerank, beam=a.beam,
                         threads=a.threads)
    except (OSError, ImportError, ValueError) as e:
        sys.exit(f"samdei-type: {e}")
    chunk = not a.no_chunk
    if not a.nbest:
        for src, out in zip(texts, conv.convert_batch(texts, chunk)):
            print(f"{src}\t{out}" if a.tsv else out)
        return
    for src in texts:
        pieces = conv.pieces(src, chunk)
        lists = conv.nbest(pieces)
        out = " ".join(hyps[0][0] if hyps else "" for hyps in lists)
        print(f"{src}\t{out}" if a.tsv else out)
        for piece, hyps in zip(pieces, lists):
            if len(pieces) > 1:
                print(f"  [{piece}]")
            for h, sc in hyps:
                print(f"    {sc:8.2f}  {h}")
