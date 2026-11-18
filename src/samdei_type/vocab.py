"""Character vocabularies. Both sides share the special tokens, ids 0-3."""

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
MAX_INPUT_CHARS = 249  # longer inputs are cut; with </s>, 250 encoder positions


class Vocab:
    def __init__(self, chars: list[str]):
        self.itos = SPECIALS + [c for c in chars if c not in SPECIALS]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """The ids of the characters of s, then </s>. Unknown characters become <unk>."""
        return [self.stoi.get(c, UNK) for c in s] + [EOS]

    def decode(self, ids) -> str:
        """The characters of ids up to the first </s>, without special tokens."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.itos[i])
        return "".join(out)
