"""Turns a procedure into a system prompt.

Retrieved knowledge and CRM context join this function later; they do not need
their own pipeline until there is something to put in it.
"""

from __future__ import annotations

from ...domain.models import SopDefinition


def build_system_prompt(sop: SopDefinition) -> str:
    persona = sop.personality
    sections = [
        f"You are {persona.name}. {persona.identity}",
        f"Speak in this tone: {persona.tone}",
        f"You are handling: {sop.description}",
        f"Follow this procedure:\n{sop.instructions}",
    ]
    return "\n\n".join(sections)
