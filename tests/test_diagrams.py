"""The diagrams are generated, so they can be checked like code.

A picture that quietly disagrees with the engine is worse than no picture, so
these tests fail when someone hand-edits a file, forgets to regenerate after a
change, or adds a node, detector, section or tool without drawing it.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations, pairwise
from pathlib import Path

import pytest

from agentic_kit.composition import build_crm, build_tools
from agentic_kit.engine.graph.state import NodeKey
from agentic_kit.engine.guardrails import default_detectors
from agentic_kit.engine.prompt.sections import DEFAULT_SECTIONS

DIAGRAMS = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
sys.path.insert(0, str(DIAGRAMS))

import generate  # noqa: E402
from _excalidraw import measure  # noqa: E402

FILENAMES = tuple(generate.DIAGRAMS)


def labels(filename: str) -> set[str]:
    document = json.loads((DIAGRAMS / filename).read_text())
    return {element["text"] for element in document["elements"] if element["type"] == "text"}


def drawn_in(filename: str, needle: str) -> bool:
    return any(needle in label for label in labels(filename))


@pytest.mark.parametrize("filename", FILENAMES)
def test_the_checked_in_diagram_matches_a_fresh_run(filename: str, tmp_path: Path) -> None:
    """Ids are derived from keys, so an unchanged diagram regenerates byte for byte."""
    generate.DIAGRAMS[filename]().dump(tmp_path / filename)

    assert (tmp_path / filename).read_text() == (DIAGRAMS / filename).read_text(), (
        f"{filename} is stale; run `python docs/diagrams/generate.py`"
    )


@pytest.mark.parametrize("filename", FILENAMES)
def test_the_file_is_something_excalidraw_will_open(filename: str) -> None:
    document = json.loads((DIAGRAMS / filename).read_text())

    assert document["type"] == "excalidraw"
    assert document["version"] == 2
    assert document["elements"]


@pytest.mark.parametrize("key", list(NodeKey))
def test_every_graph_node_is_drawn(key: NodeKey) -> None:
    assert drawn_in("02-turn-flow.excalidraw", key.value)


@pytest.mark.parametrize("detector", default_detectors(), ids=lambda det: det.id)
def test_every_detector_is_drawn(detector: object) -> None:
    assert drawn_in("03-seams.excalidraw", detector.id)


@pytest.mark.parametrize("section", DEFAULT_SECTIONS, ids=lambda sec: sec.key)
def test_every_prompt_section_is_drawn(section: object) -> None:
    assert drawn_in("03-seams.excalidraw", section.key)


@pytest.mark.parametrize("tool", build_tools(build_crm()).catalog(), ids=lambda tool: tool.name)
def test_every_tool_is_drawn(tool: object) -> None:
    assert drawn_in("03-seams.excalidraw", tool.name)


@pytest.mark.parametrize("filename", FILENAMES)
def test_no_two_boxes_sit_on_top_of_each_other(filename: str) -> None:
    """Catches a layout that broke when someone added a box to a full row."""
    boxes = generate.DIAGRAMS[filename]().boxes

    for (left_key, left), (right_key, right) in combinations(boxes.items(), 2):
        apart = (
            left.x + left.w <= right.x
            or right.x + right.w <= left.x
            or left.y + left.h <= right.y
            or right.y + right.h <= left.y
        )
        assert apart, f"{filename}: {left_key} overlaps {right_key}"


@pytest.mark.parametrize("filename", FILENAMES)
def test_no_arrow_runs_through_a_box_it_is_not_connected_to(filename: str) -> None:
    """An arrow crossing an unrelated box reads as a connection that is not there."""
    diagram = generate.DIAGRAMS[filename]()
    rectangles = {box.id: (key, box) for key, box in diagram.boxes.items()}
    document = json.loads((DIAGRAMS / filename).read_text())

    for arrow in (e for e in document["elements"] if e["type"] == "arrow"):
        ends = {arrow["startBinding"]["elementId"], arrow["endBinding"]["elementId"]}
        corners = [(arrow["x"] + dx, arrow["y"] + dy) for dx, dy in arrow["points"]]
        for start, end in pairwise(corners):
            for box_id, (key, box) in rectangles.items():
                if box_id in ends:
                    continue
                assert not _cuts_through(start, end, box), (
                    f"{filename}: an arrow between {sorted(_names(rectangles, ends))} "
                    f"runs through {key}"
                )


def _names(rectangles: dict, ids: set[str]) -> list[str]:
    return [key for box_id, (key, _) in rectangles.items() if box_id in ids]


def _cuts_through(start: tuple[float, float], end: tuple[float, float], box: object) -> bool:
    """Liang-Barsky, against a box shrunk a little so grazing an edge is allowed."""
    margin = 3
    left, right = box.x + margin, box.x + box.w - margin
    top, bottom = box.y + margin, box.y + box.h - margin
    dx, dy = end[0] - start[0], end[1] - start[1]
    entering, leaving = 0.0, 1.0

    for delta, distance in (
        (-dx, start[0] - left),
        (dx, right - start[0]),
        (-dy, start[1] - top),
        (dy, bottom - start[1]),
    ):
        if delta == 0:
            if distance < 0:
                return False
            continue
        crossing = distance / delta
        if delta < 0:
            entering = max(entering, crossing)
        else:
            leaving = min(leaving, crossing)

    return entering < leaving


@pytest.mark.parametrize("filename", FILENAMES)
def test_no_box_hangs_out_of_the_band_behind_it(filename: str) -> None:
    """A band is a claim about what belongs together, so a box half outside one misleads."""
    document = json.loads((DIAGRAMS / filename).read_text())
    bands = [
        element
        for element in document["elements"]
        if element["type"] == "rectangle" and element.get("fillStyle") == "hachure"
    ]

    for key, box in generate.DIAGRAMS[filename]().boxes.items():
        middle = (box.x + box.w / 2, box.y + box.h / 2)
        for band in bands:
            if not _holds(band, *middle):
                continue
            assert _holds(band, box.x, box.y) and _holds(band, box.x + box.w, box.y + box.h), (
                f"{filename}: {key} hangs out of the band behind it"
            )


def _holds(band: dict, x: float, y: float) -> bool:
    return (
        band["x"] <= x <= band["x"] + band["width"] and band["y"] <= y <= band["y"] + band["height"]
    )


@pytest.mark.parametrize("filename", FILENAMES)
def test_no_label_spills_out_of_its_box(filename: str) -> None:
    document = json.loads((DIAGRAMS / filename).read_text())
    shapes = {element["id"]: element for element in document["elements"]}

    for element in document["elements"]:
        container = shapes.get(element.get("containerId") or "")
        if element["type"] != "text" or container is None or container["type"] != "rectangle":
            continue
        width, height = measure(element["text"], element["fontSize"])
        assert width <= container["width"] - 16, f"{filename}: {element['text']!r} is too wide"
        assert height <= container["height"] - 16, f"{filename}: {element['text']!r} is too tall"
