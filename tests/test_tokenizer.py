"""
tests/test_tokenizer.py — Tests for CharTokenizer.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transformers"))

import pytest
from tokenizer import CharTokenizer


class TestCharTokenizer:
    def test_vocab_size(self):
        tok = CharTokenizer()
        tok.train("hello")
        assert tok.vocab_size == 4  # h, e, l, o

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.train("hello world")
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_sorted_deterministic_vocab(self):
        """Vocab should be sorted, so same input always gives same IDs."""
        tok = CharTokenizer()
        tok.train("bac")
        assert tok.char_to_id["a"] < tok.char_to_id["b"] < tok.char_to_id["c"]

    def test_unknown_char_raises(self):
        tok = CharTokenizer()
        tok.train("abc")
        with pytest.raises(KeyError):
            tok.encode("z")

    def test_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer()
        tok.train("the quick brown fox")
        path = str(tmp_path / "tok.json")
        tok.save(path)

        loaded = CharTokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size
        text = "the fox"
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text
