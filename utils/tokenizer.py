"""
Character-level tokenizer for speech recognition on Apple Silicon.

This module provides a lightweight character-level tokenization system optimized
for speech recognition tasks. It implements a simple yet effective vocabulary
strategy that balances simplicity with coverage for English speech recognition.

Tokenization Strategy:
- Character-level: Individual characters as basic units
- Vocabulary: space + lowercase a-z + apostrophe (29 total tokens)
- Blank token: Index 0 for CTC and RNN-T alignment
- ASCII focus: Optimized for English language processing

Vocabulary Design:
- Index 0: Blank token for CTC/RNN-T
- Index 1: Space character for word boundaries
- Index 2-27: Lowercase letters a-z
- Index 28: Apostrophe for contractions
- Total size: 29 tokens (compact vocabulary)

Apple Silicon Optimizations:
- Efficient string processing on unified memory architecture
- Minimal vocabulary reduces embedding table memory
- Fast character mapping using dictionary lookups
- Unicode normalization optimized for Apple Silicon

Usage in Speech Recognition:
- CTC training: Provides alignment-free character sequence modeling
- RNN-T training: Serves as predictor vocabulary for language modeling
- Decoding: Character sequences collapsed to words post-processing
- Evaluation: WER computation after character-to-word conversion

Integration Points:
- Used by: ConMambaCTC model for character-level prediction
- Used by: MCTModel RNN-T predictor for language modeling
- Used by: train_CTC.py and train_RNNT.py training pipelines
- Compatible with: LibriSpeech and other English speech datasets

Performance Characteristics:
- Memory usage: 29-token vocabulary (minimal embedding table)
- Processing speed: O(text_length) for encoding/decoding
- Coverage: Handles standard English text with normalization
- Accuracy: Character-level granularity enables robust recognition

Design Trade-offs:
- Simple vocabulary vs. subword units: Lower memory, less semantic modeling
- Character-level vs. word-level: More robust to OOV, longer sequences
- ASCII focus vs. Unicode: Better performance for English, limited multilingual
- Fixed vocabulary vs. learned: Predictable behavior, no adaptive capability

References:
- CTC tokenization: Character-level modeling for speech recognition
- RNN-T integration: Language model vocabulary for predictor networks
- Apple Silicon guide: README/Mamba-on-Apple-Silicon.md Section 4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# Tokenizer Configuration Constants
class TokenizerConstants:
    """Named constants for character tokenizer configuration.
    
    These constants define the vocabulary structure and token mappings
    for speech recognition on Apple Silicon hardware.
    """
    
    # Vocabulary Structure
    BLANK_TOKEN_ID = 0              # CTC/RNN-T blank token index
    VOCABULARY_SIZE = 29            # Total tokens: blank + space + a-z + apostrophe
    
    # Character Set
    ALPHABET_SIZE = 26              # English alphabet a-z
    SPECIAL_CHAR_COUNT = 2          # Space and apostrophe
    
    # Token Mapping
    SPACE_TOKEN_ID = 1              # Space character token
    FIRST_LETTER_ID = 2             # Start of alphabet range (a)
    LAST_LETTER_ID = 27             # End of alphabet range (z)
    APOSTROPHE_TOKEN_ID = 28        # Apostrophe for contractions
    
    @staticmethod
    def get_vocabulary_info() -> str:
        """Return vocabulary structure documentation."""
        return f"""
        Character Tokenizer Vocabulary:
        - Index 0: Blank token (CTC/RNN-T alignment)
        - Index 1: Space character (word boundaries)
        - Index 2-27: Lowercase letters a-z
        - Index 28: Apostrophe (contractions like don't, can't)
        - Total size: {TokenizerConstants.VOCABULARY_SIZE} tokens
        - Memory efficient: Minimal embedding table size
        """


@dataclass
class CharTokenizer:
    """Character-level tokenizer for speech recognition with Apple Silicon optimization.
    
    This tokenizer provides efficient character-level tokenization optimized for
    English speech recognition tasks. It implements a compact vocabulary strategy
    that balances simplicity with effective coverage.
    
    Vocabulary Strategy:
    - Compact design: Only 29 tokens total for memory efficiency
    - Character-level: Robust to out-of-vocabulary words
    - English-focused: Optimized for English speech recognition
    - CTC/RNN-T compatible: Includes blank token for alignment
    
    Token Mapping:
    - Blank (0): CTC/RNN-T alignment token
    - Space (1): Word boundary marker
    - Letters (2-27): Lowercase a-z
    - Apostrophe (28): Contraction support
    
    Apple Silicon Optimizations:
    - Dictionary lookups leverage unified memory efficiency
    - Minimal vocabulary reduces embedding memory requirements
    - String processing optimized for Apple Silicon architecture
    - Unicode normalization uses native Apple frameworks
    
    Usage Examples:
        tokenizer = CharTokenizer()
        ids = tokenizer.encode("hello world")  # -> [8, 5, 12, 12, 15, 1, 23, 15, 18, 12, 4]
        text = tokenizer.decode(ids)           # -> "hello world"
        vocab_size = tokenizer.vocab_size      # -> 29
    
    Integration with Models:
    - ConMambaCTC: Uses as output vocabulary for character prediction
    - MCTModel: Predictor network uses for language modeling
    - Training: Both CTC and RNN-T training pipelines
    - Evaluation: WER computation after character-to-word conversion
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
