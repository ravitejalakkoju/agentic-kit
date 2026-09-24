from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from ..composition import Components


def components(request: Request) -> Components:
    return request.app.state.components


Wired = Annotated[Components, Depends(components)]
