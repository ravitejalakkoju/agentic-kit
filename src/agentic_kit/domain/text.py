"""Text as a person actually sees it.

A detector that matches on words can be walked straight past using characters
nobody sees: a zero-width space inside a word, a bidi override reordering it, a
fullwidth lookalike standing in for a letter. Normalising first means the
pattern and the reader are looking at the same string.
"""

from __future__ import annotations

import unicodedata

KEPT_CONTROLS = frozenset("\n\t")
"""Layout the text genuinely needs. Every other control character is noise."""

HIDDEN = frozenset({"Cf", "Cc"})
"""Formatting and control categories, which together are the hiding places."""


def readable(text: str) -> str:
    """Fold lookalikes and drop characters that render as nothing.

    NFKC turns compatibility forms into the plain letters they imitate, and the
    Cf category covers the rest in one rule: zero-width spaces, soft hyphens,
    bidi overrides, and the tag block used to smuggle instructions.

    Confusables across scripts are not folded. A Cyrillic 'a' is a different
    letter rather than a compatibility form, and telling those apart needs a
    table rather than a rule.
    """
    folded = unicodedata.normalize("NFKC", text)
    return "".join(
        char for char in folded if char in KEPT_CONTROLS or unicodedata.category(char) not in HIDDEN
    )
