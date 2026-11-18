"""CLI for the web viewer: ``export`` writes a page, ``serve`` serves one with editing.

    python -m roomlayout.viewer export PROBLEM.json RESULT.json --out DIR [--fit FIT.json]
        [--meshes DIR] [--request TEXT] [--title TEXT]
    python -m roomlayout.viewer serve DIR [--port 8000]

Served, the page can re-solve: POST /api/solve {"locks": [{"id", "x", "y", "yaw"}]}
pins the locked pieces where the user put them (hard ``fixed``, origin "user") and
returns new layouts; POST /api/evaluate {"placements": {id: {x, y, yaw}}} scores a
hand-edited arrangement and explains it.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from roomlayout.schema import Layout, LayoutProblem, Placement, Pose2D, SolveResult

from . import _layout_json, export

RESOLVE_SECONDS = 6.0


def check_lock(problem: LayoutProblem, lock: dict[str, float]) -> None:
    """Refuse a lock that puts a piece outside the room or into an obstacle. Pinned pieces
    are facts to the solver (ADR-0008: no inside-room check on them), so a piece dragged
    through a wall and locked would otherwise be accepted as is."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon

    a = next(x for x in problem.assets if x.id == lock["id"])
    fp = Polygon(a.effective_footprint().points)
    fp = translate(rotate(fp, lock["yaw"], origin=(0, 0), use_radians=True), lock["x"], lock["y"])
    floor = Polygon(problem.room.floor.points)
    outside = fp.difference(floor.buffer(0.01)).area
    if outside > 0.005:
        raise ValueError(f"the {a.label or a.id} sticks out of the room ({outside:.2f} m²); "
                         "move it inside before locking it")  # fmt: skip
    for ob in problem.room.obstacles:
        if fp.intersection(Polygon(ob.footprint.points)).area > 0.005:
            raise ValueError(f"the {a.label or a.id} overlaps the {ob.label}; move it first")


def locked_problem(problem: LayoutProblem, locks: list[dict[str, float]]) -> LayoutProblem:
    """``problem`` with each locked asset pinned at the user's pose (hard ``fixed``)."""
    data = problem.model_dump(mode="json")
    ids = {a["id"] for a in data["assets"]}
    for lk in locks:
        if lk["id"] not in ids:
            raise KeyError(f"unknown asset {lk['id']!r}")
        check_lock(problem, lk)
        data["initial"][lk["id"]] = {"x": lk["x"], "y": lk["y"], "yaw": lk["yaw"]}
        data["constraints"] = [c for c in data["constraints"] if c["id"] != f"lock:{lk['id']}"]
        data["constraints"].append({
            "type": "fixed", "id": f"lock:{lk['id']}", "a": lk["id"], "hard": True,
            "origin": "user", "rationale": "you placed it here",
        })  # fmt: skip
    return LayoutProblem.model_validate(data)


def resolve(
    problem: LayoutProblem, locks: list[dict[str, float]], seconds: float = RESOLVE_SECONDS
) -> dict[str, Any]:
    from roomlayout.solver import solve

    p = locked_problem(problem, locks)
    res = solve(p, k=3, seed=0, time_budget_s=seconds)
    return {"layouts": [_layout_json(p, lay) for lay in res.layouts], "notes": res.notes}


def evaluate_placements(problem: LayoutProblem, placements: dict[str, dict]) -> dict[str, Any]:
    from roomlayout.solver import evaluate

    poses = {k: Pose2D(**v) for k, v in placements.items()}
    results, feasible, cost = evaluate(problem, poses)
    merged = {**problem.initial, **poses}
    lay = Layout(
        id="edited", placements=[Placement(asset_id=k, pose=v) for k, v in merged.items()],
        cost=cost, feasible=feasible, results=results,
    )  # fmt: skip
    return _layout_json(problem, lay)


class Handler(SimpleHTTPRequestHandler):
    problem: LayoutProblem

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter
        sys.stderr.write("viewer: " + fmt % args + "\n")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/api/ping":
            data = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/solve":
                out = resolve(self.problem, body.get("locks", []))
            elif self.path == "/api/evaluate":
                out = evaluate_placements(self.problem, body["placements"])
            else:
                self.send_error(404)
                return
            data, code = json.dumps(out).encode(), 200
        except (KeyError, ValueError, TypeError) as e:
            data, code = json.dumps({"error": str(e)}).encode(), 400
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(directory: Path, port: int = 8000, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    problem = LayoutProblem.model_validate_json((directory / "problem.json").read_text())
    handler = type("BoundHandler", (Handler,), {"problem": problem})
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(directory)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roomlayout.viewer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a viewer page for a problem + result")
    ex.add_argument("problem", type=Path)
    ex.add_argument("result", type=Path)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--fit", type=Path)
    ex.add_argument("--meshes", type=Path)
    ex.add_argument("--request")
    ex.add_argument("--title")
    sv = sub.add_parser("serve", help="serve a viewer directory with drag / lock / re-solve")
    sv.add_argument("dir", type=Path)
    sv.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        problem = LayoutProblem.model_validate_json(args.problem.read_text())
        result = SolveResult.model_validate_json(args.result.read_text())
        fit = json.loads(args.fit.read_text()) if args.fit else None
        page = export(problem, result, args.out, fit=fit, mesh_dir=args.meshes,
                      request=args.request, title=args.title)  # fmt: skip
        print(page)
        return 0
    httpd = serve(args.dir, args.port)
    print(f"viewer: http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
