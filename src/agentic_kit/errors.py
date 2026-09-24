"""Error types, one per boundary rather than one per failure case."""

from __future__ import annotations


class EngineError(Exception):
    """The engine could not complete a turn."""


class GraphValidationError(EngineError):
    """The node set and the edge map disagree."""


class ProviderError(EngineError):
    """An external model provider failed."""


class KnowledgeError(EngineError):
    """The knowledge base was handed something it cannot store or search."""
