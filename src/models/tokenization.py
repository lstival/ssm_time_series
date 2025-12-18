"""Compatibility shim.

This module used to contain tokenization utilities. They have been consolidated
into `models/tokenizers.py`.

Keep this file so older imports continue to work.
"""

from __future__ import annotations

try:
    from tokenizers import InputLayout, TokenizeMethod, Tokenizer, as_bft, tokenize_sequence
except Exception:
    from .tokenizers import InputLayout, TokenizeMethod, Tokenizer, as_bft, tokenize_sequence

__all__ = [
    "InputLayout",
    "TokenizeMethod",
    "Tokenizer",
    "tokenize_sequence",
    "as_bft",
]
