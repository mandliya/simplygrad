"""
Character-level tokenizer.

Maps each unique character in a training corpus to an integer ID.
Simple, dependency-free, and well-suited for small-scale experiments
like training a transformer on TinyShakespeare.
"""

import json
from typing import List


class CharTokenizer:
    """
    Character-level tokenizer: every unique character is one token.

    Usage:
        tok = CharTokenizer()
        tok.train("hello world")
        ids = tok.encode("hello")   # [3, 1, 4, 4, 5]
        text = tok.decode(ids)      # "hello"
    """

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0

    def train(self, text: str) -> None:
        """Build vocabulary from all unique characters in text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        """Convert a string to a list of token IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return "".join(self.id_to_char[i] for i in ids)

    def save(self, path: str) -> None:
        """Save the vocabulary mapping to a JSON file."""
        with open(path, "w") as f:
            json.dump({"char_to_id": self.char_to_id}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a tokenizer from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(i): ch for ch, i in tok.char_to_id.items()}
        tok.vocab_size = len(tok.char_to_id)
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
