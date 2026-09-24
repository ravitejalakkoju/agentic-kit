# Diagrams

Four pictures of the engine. They are generated, not drawn, so they can be
checked like code.

| File | Answers |
| --- | --- |
| `01-layers.excalidraw` | What depends on what, and where `build()` wires it |
| `02-turn-flow.excalidraw` | What one turn does, including the tool loop |
| `03-seams.excalidraw` | Which protocols exist and what implements them |
| `04-roadmap.excalidraw` | What is built, what is planned, what was deferred |

## Opening one

Go to [excalidraw.com](https://excalidraw.com), then **File -> Open** and pick
the `.excalidraw` file. Everything is editable once it is open.

## Changing one

Edit [`generate.py`](generate.py) and regenerate:

```sh
python docs/diagrams/generate.py
```

Do not hand-edit the `.excalidraw` files. `tests/test_diagrams.py` regenerates
them and compares, so a hand edit fails the suite on the next run.

## What keeps them honest

Node names come from `EDGES`, detector ids from `default_detectors()`, section
keys from `DEFAULT_SECTIONS` and tool names from the `ToolRegistry`. Add a node
or a tool and it appears the next time you regenerate; forget to regenerate and
the tests say so.

`tests/test_diagrams.py` also checks the layout, not just the contents: no two
boxes overlap, no label spills outside its box, and no arrow runs through a box
it is not connected to. Those three caught real mistakes while these diagrams
were being laid out, and they are the reason a new box cannot quietly land on
top of an old one.

Element ids are derived from their keys rather than randomised, so regenerating
an unchanged diagram produces a byte-identical file and a git diff only shows
what actually moved.

## Layout notes

[`_excalidraw.py`](_excalidraw.py) knows nothing about this project; it is
boxes, arrows and JSON. An arrow leaves and enters through whichever edges face
each other. When that would cut through the boxes in between, pass `via` with
waypoints and route it along an empty gutter:

```python
d.arrow("guard_input", "persist_state", via=[(360, 443), (360, 1248)])
```
