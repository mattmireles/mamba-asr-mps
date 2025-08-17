from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CharTokenizer:
    """Simple character-level tokenizer with blank=0.

    Vocabulary: space + a-z + apostrophe
    blank id = 0
    ' ' -> 1, 'a'->2, ..., 'z'->27, "'"->28
    vocab_size = 29
    """

    def __post_init__(self):
        self.chars = [' '] + [chr(ord('a') + i) for i in range(26)] + ["'"]
        self.blank_id = 0
        # token ids start at 1 for first char
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.chars)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1

    def normalize(self, text: str) -> str:
        text = text.lower()
        out = []
        for ch in text:
            if ch in self.char_to_id:
                out.append(ch)
            elif ch == '’':
                out.append("'")
            elif ch == '\n':
                out.append(' ')
            # else drop
        return ''.join(out)

    def encode(self, text: str) -> List[int]:
        t = self.normalize(text)
        return [self.char_to_id[ch] for ch in t]

    def decode(self, ids: List[int]) -> str:
        chars: List[str] = []
        for i in ids:
            if i == self.blank_id:
                continue
            if i in self.id_to_char:
                chars.append(self.id_to_char[i])
        return ''.join(chars)
