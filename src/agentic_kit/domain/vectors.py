"""Text as coordinates, and the one measurement taken over them.

Both the passage store and the procedure matcher ask the same question of two
vectors, so the question is asked in one place. It lives in the domain because
it is arithmetic rather than policy: nothing here knows what is being compared
or what the answer will be used for.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

type Vector = Sequence[float]


def cosine(left: Vector, right: Vector) -> float:
    """How far apart the two point, ignoring how long either one is.

    Length is a property of how much text went in, not of what it meant, so
    two vectors are compared by direction alone.
    """
    size = _length(left) * _length(right)
    return sum(a * b for a, b in zip(left, right, strict=True)) / size if size else 0.0


def _length(vector: Vector) -> float:
    return sqrt(sum(value * value for value in vector))
