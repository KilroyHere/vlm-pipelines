"""Write JSON Schemas for the contract: ``python -m roomlayout.schema.export [outdir]``.

The committed files in ``schemas/`` are what non-Python consumers (the iOS app,
the VLM structured-output call) validate against. A test fails if they are stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from .asset import Asset
from .layout import LayoutProblem, SolveResult

MODELS: dict[str, type[BaseModel]] = {
    "asset": Asset,
    "layout_problem": LayoutProblem,
    "solve_result": SolveResult,
}


def render() -> dict[str, str]:
    return {
        f"{name}.schema.json": json.dumps(model.model_json_schema(), indent=2) + "\n"
        for name, model in MODELS.items()
    }


def main(outdir: str = "schemas") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, text in render().items():
        (out / filename).write_text(text)
        print(out / filename)


if __name__ == "__main__":
    main(*sys.argv[1:])
