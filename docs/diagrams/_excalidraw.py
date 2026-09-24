"""A small Excalidraw document builder: boxes, arrows, text, one JSON file.

This module knows nothing about agentic-kit. It exists so the scripts next to
it can say *what* to draw without restating Excalidraw's element schema at
every call site.

Ids and seeds are derived from a caller-supplied key rather than randomised, so
regenerating an unchanged diagram produces a byte-identical file. That is what
lets a test compare the checked-in diagrams against a fresh run and fail when
someone forgets to regenerate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

INK: Final = "#1e1e1e"
TRANSPARENT: Final = "transparent"

Point = tuple[float, float]

_ID_ALPHABET: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
_ID_LENGTH: Final = 21
_STAMP: Final = 1756280000000
_LINE_HEIGHT: Final = 1.25
_GLYPH_WIDTH: Final = 0.55
_PADDING: Final = 8


def _digest(key: str, size: int) -> bytes:
    return blake2b(key.encode(), digest_size=size).digest()


def _ident(key: str) -> str:
    """A stable id in Excalidraw's own 21-character shape."""
    return "".join(_ID_ALPHABET[byte % len(_ID_ALPHABET)] for byte in _digest(key, _ID_LENGTH))


def _seed(key: str) -> int:
    """Excalidraw wants a positive int; it only affects hand-drawn jitter."""
    return int.from_bytes(_digest(key, 4), "big") or 1


def measure(text: str, size: int) -> tuple[float, float]:
    """Roughly how much room a label needs. Excalidraw's font is near-monospace."""
    lines = text.split("\n")
    width = max(len(line) for line in lines) * size * _GLYPH_WIDTH
    return width, len(lines) * size * _LINE_HEIGHT


@dataclass(slots=True)
class _Box:
    """Where a box sits, and which arrows Excalidraw must re-bind to it."""

    id: str
    x: float
    y: float
    w: float
    h: float
    arrows: list[str] = field(default_factory=list)

    @property
    def centre(self) -> Point:
        return self.x + self.w / 2, self.y + self.h / 2

    def face(self, toward: Point) -> Point:
        """The point on the edge that looks at `toward`, along the dominant axis."""
        centre_x, centre_y = self.centre
        across, down = toward[0] - centre_x, toward[1] - centre_y
        if abs(down) >= abs(across):
            return centre_x, (self.y + self.h if down > 0 else self.y)
        return (self.x + self.w if across > 0 else self.x), centre_y


class Diagram:
    """Collects elements in draw order and writes them as an Excalidraw file.

    Elements render in the order they are added, so draw backdrops such as
    `group_box` before the boxes that sit inside them.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._elements: list[dict[str, Any]] = []
        self._boxes: dict[str, _Box] = {}
        self._drawn = 0

    @property
    def boxes(self) -> Mapping[str, _Box]:
        """The arrow-connectable boxes, for callers that want to check a layout."""
        return MappingProxyType(self._boxes)

    def title(self, key: str, text: str, x: float, y: float, size: int = 28) -> None:
        self.text(key, text, x, y, size=size)

    def text(self, key: str, text: str, x: float, y: float, size: int = 14) -> None:
        """Free-standing label, left aligned, sized to its longest line."""
        width, height = measure(text, size)
        self._add(
            "text",
            key,
            x=x,
            y=y,
            width=max(width, 120),
            height=height,
            **_text_fields(text, size, align="left", vertical="top", resize=True),
        )

    def box(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        bg: str,
        size: int | None = None,
    ) -> str:
        """A labelled rectangle. The returned key is what `arrow` refers to."""
        label_key = f"{key}/label"
        box = self._add(
            "rectangle",
            key,
            x=x,
            y=y,
            width=w,
            height=h,
            backgroundColor=bg,
            roundness={"type": 3},
            boundElements=[{"type": "text", "id": self._id(label_key)}],
        )
        self._boxes[key] = _Box(id=box["id"], x=x, y=y, w=w, h=h)
        self._add(
            "text",
            label_key,
            x=x + _PADDING,
            y=y + _PADDING,
            width=w - 2 * _PADDING,
            height=h - 2 * _PADDING,
            **_text_fields(text, size or _fit(text), container=box["id"]),
        )
        return key

    def group_box(
        self, key: str, text: str, x: float, y: float, w: float, h: float, bg: str
    ) -> None:
        """A hachured backdrop grouping related boxes, labelled at the top left."""
        self._add(
            "rectangle",
            key,
            x=x,
            y=y,
            width=w,
            height=h,
            backgroundColor=bg,
            fillStyle="hachure",
            roundness={"type": 3},
        )
        self.text(f"{key}/caption", text, x + 16, y + 12, size=16)

    def arrow(
        self,
        src: str,
        dst: str,
        label: str | None = None,
        dashed: bool = False,
        via: Sequence[Point] = (),
    ) -> None:
        """Connect two boxes, optionally elbowing through `via` waypoints.

        Waypoints exist for edges that would otherwise cut straight through the
        boxes between their two ends; route those along an empty gutter.
        """
        source, target = self._boxes[src], self._boxes[dst]
        start = source.face(via[0] if via else target.centre)
        end = target.face(via[-1] if via else source.centre)
        points = [start, *via, end]

        key = f"{src}->{dst}:{label or ''}"
        arrow = self._add(
            "arrow",
            key,
            x=start[0],
            y=start[1],
            width=max(x for x, _ in points) - min(x for x, _ in points),
            height=max(y for _, y in points) - min(y for _, y in points),
            strokeStyle="dashed" if dashed else "solid",
            roundness={"type": 2},
            boundElements=[],
            points=[[x - start[0], y - start[1]] for x, y in points],
            lastCommittedPoint=None,
            startBinding={"elementId": source.id, "focus": 0, "gap": 4},
            endBinding={"elementId": target.id, "focus": 0, "gap": 4},
            startArrowhead=None,
            endArrowhead="arrow",
            elbowed=False,
        )
        source.arrows.append(arrow["id"])
        target.arrows.append(arrow["id"])

        if not label:
            return
        label_key = f"{key}/label"
        arrow["boundElements"] = [{"type": "text", "id": self._id(label_key)}]
        anchor = _longest_leg(points)
        width, height = measure(label, 12)
        self._add(
            "text",
            label_key,
            x=anchor[0] - width / 2,
            y=anchor[1] - height / 2,
            width=width,
            height=height,
            **_text_fields(label, 12, container=arrow["id"], resize=True),
        )

    def dump(self, path: Path) -> None:
        self._bind_arrows()
        document = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self._elements,
            "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
            "files": {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n")

    def _id(self, key: str) -> str:
        return _ident(f"{self._name}:{key}")

    def _add(self, kind: str, key: str, **fields: Any) -> dict[str, Any]:
        element_id = self._id(key)
        self._drawn += 1
        element: dict[str, Any] = {
            "id": element_id,
            "type": kind,
            "angle": 0,
            "strokeColor": INK,
            "backgroundColor": TRANSPARENT,
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "index": f"a{self._drawn:04d}",
            "roundness": None,
            "seed": _seed(element_id),
            "version": 1,
            "versionNonce": _seed(f"{element_id}/nonce"),
            "isDeleted": False,
            "boundElements": None,
            "updated": _STAMP,
            "link": None,
            "locked": False,
        }
        element.update(fields)
        self._elements.append(element)
        return element

    def _bind_arrows(self) -> None:
        """Excalidraw keeps bindings on both ends; boxes learn their arrows last."""
        by_id = {element["id"]: element for element in self._elements}
        for box in self._boxes.values():
            element = by_id[box.id]
            bound = list(element.get("boundElements") or [])
            element["boundElements"] = bound + [
                {"id": arrow_id, "type": "arrow"} for arrow_id in box.arrows
            ]


def _longest_leg(points: Sequence[Point]) -> Point:
    """Label the roomiest straight run, so two elbowed arrows do not collide."""
    start, end = max(
        pairwise(points),
        key=lambda leg: abs(leg[1][0] - leg[0][0]) + abs(leg[1][1] - leg[0][1]),
    )
    return (start[0] + end[0]) / 2, (start[1] + end[1]) / 2


def _fit(text: str) -> int:
    """Shrink the label as it grows, so tall boxes stay inside their rectangle."""
    lines = text.count("\n") + 1
    if lines <= 2:
        return 15
    return 13 if lines <= 4 else 11


def _text_fields(
    text: str,
    size: int,
    *,
    container: str | None = None,
    align: str = "center",
    vertical: str = "middle",
    resize: bool = False,
) -> dict[str, Any]:
    return {
        "text": text,
        "originalText": text,
        "fontSize": size,
        "fontFamily": 1,
        "textAlign": align,
        "verticalAlign": vertical,
        "containerId": container,
        "autoResize": resize,
        "lineHeight": _LINE_HEIGHT,
    }
