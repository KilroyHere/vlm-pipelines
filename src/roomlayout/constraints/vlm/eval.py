"""Eval for constraint extraction: ``python -m roomlayout.constraints.vlm.eval``.

Modes:
  --live            call the API for every case (needs credentials), save raw responses
                    to --out, then score.
  --replay DIR      re-run the post-processing on saved raw responses (no network), then
                    score. Cases without a saved file are skipped and reported.

Scores (see docs/vlm.md):
  constraint precision / recall / F1   match = type + resolved ids (+ numbers within
                                       tolerance); a/b order ignored for symmetric types
  hard agreement                       hard/soft right, among matched items that specify it
  hallucinated-id rate                 model items naming an asset/opening/wall that isn't
                                       in the room, over all items returned (all attempts)
  unexpressed recall                   expected "couldn't express" wishes that were reported
  cost / latency / cache reads         from usage, per case (live or recorded)

Case format (evals/vlm/cases.jsonl, one JSON object per line):
  {"id", "problem" (examples/problems/<name>.json), "text", "tags": [...],
   "keep_existing": false,           # default: the problem's constraints are removed
   "expected": {
     "constraints": [{"type", "a"?, "b"?, "wall_id"?, "opening_id"?, <number>?, "hard"?,
                      "optional"?, "tol"?: {<field>: abs tolerance}}],
     "unexpressed": [{"any": [keywords], "kind"?}]}}
Keys left out of an expected constraint are not checked; ``"wall_id": null`` requires
"any wall". Optional expected items are neither required nor counted as false positives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roomlayout.schema import LayoutProblem

from .extract import DEFAULT_EFFORT, DEFAULT_MODEL, ParseResult, build_request, parse_request
from .prompt import PROMPT_VERSION

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CASES = REPO_ROOT / "evals" / "vlm" / "cases.jsonl"
DEFAULT_PROBLEMS = REPO_ROOT / "examples" / "problems"

_SYMMETRIC = {"near", "distance_at_least", "parallel"}
_REF_KEYS = ("wall_id", "opening_id")
_NUM_KEYS = ("max_distance", "min_distance", "distance", "min_width", "max_angle_deg", "max_gap")
_HALLUCINATION_CODES = {"unknown_asset", "unknown_opening", "unknown_wall"}


def default_tolerance(key: str, expected: float) -> float:
    if key == "max_angle_deg":
        return 10.0
    return max(0.1, 0.15 * abs(expected))


# --------------------------------------------------------------------------- cases


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    cases = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return cases


def load_problem(case: dict[str, Any], problems_dir: Path = DEFAULT_PROBLEMS) -> LayoutProblem:
    problem = LayoutProblem.model_validate_json(
        (problems_dir / f"{case['problem']}.json").read_text()
    )
    if not case.get("keep_existing", False):
        problem = problem.model_copy(update={"constraints": []})
    return problem


# --------------------------------------------------------------------------- matching


def _matches(exp: dict[str, Any], c: dict[str, Any]) -> bool:
    if exp["type"] != c["type"]:
        return False
    if c["type"] in _SYMMETRIC and "a" in exp and "b" in exp:
        if {exp["a"], exp["b"]} != {c.get("a"), c.get("b")}:
            return False
    else:
        for k in ("a", "b"):
            if k in exp and exp[k] != c.get(k):
                return False
    for k in _REF_KEYS:
        if k in exp and exp[k] != c.get(k):
            return False
    tol = exp.get("tol", {})
    for k in _NUM_KEYS:
        if k in exp:
            v = c.get(k)
            if v is None or abs(v - exp[k]) > tol.get(k, default_tolerance(k, exp[k])):
                return False
    return True


@dataclass
class CaseScore:
    case_id: str
    tags: list[str]
    status: str
    n_pred: int = 0
    n_required: int = 0
    tp_required: int = 0  # required expected items matched
    tp_pred: int = 0  # predicted items matching any expected (required or optional)
    hard_checked: int = 0
    hard_right: int = 0
    n_items_raw: int = 0
    n_hallucinated: int = 0
    unexp_expected: int = 0
    unexp_found: int = 0
    unexp_kind_right: int = 0
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    missed: list[dict[str, Any]] = field(default_factory=list)
    missed_unexpressed: list[list[str]] = field(default_factory=list)
    latency_s: float | None = None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    repaired: bool = False

    @property
    def exact(self) -> bool:
        return (
            self.status == "ok"
            and self.tp_required == self.n_required
            and self.tp_pred == self.n_pred
            and self.unexp_found == self.unexp_expected
        )


def score_case(case: dict[str, Any], result: ParseResult, latency_s: float | None) -> CaseScore:
    """Score one case. ``latency_s``: measured wall time (None when unknown, e.g. replay
    of a hand-authored fixture)."""
    exp = case.get("expected", {})
    expected = exp.get("constraints", [])
    preds = [c.model_dump() for c in result.constraints]
    s = CaseScore(case_id=case["id"], tags=case.get("tags", []), status=result.status)
    s.n_pred = len(preds)
    s.n_required = sum(1 for e in expected if not e.get("optional"))

    # Greedy matching, required items first so optional ones can't steal their match.
    used: set[int] = set()
    for e in sorted(expected, key=lambda e: bool(e.get("optional"))):
        hit = next((i for i, c in enumerate(preds) if i not in used and _matches(e, c)), None)
        if hit is None:
            if not e.get("optional"):
                s.missed.append(e)
            continue
        used.add(hit)
        s.tp_pred += 1
        if not e.get("optional"):
            s.tp_required += 1
        if "hard" in e:
            s.hard_checked += 1
            s.hard_right += int(preds[hit]["hard"] == e["hard"])
    s.false_positives = [c for i, c in enumerate(preds) if i not in used]

    for a in result.attempts:
        s.n_items_raw += a.n_items
        s.n_hallucinated += sum(1 for d in a.dropped if d.code in _HALLUCINATION_CODES)

    for u in exp.get("unexpressed", []):
        s.unexp_expected += 1
        words = [w.lower() for w in u["any"]]
        found = next(
            (
                p
                for p in result.unexpressed
                if any(w in f"{p.text} {p.reason}".lower() for w in words)
            ),
            None,
        )
        if found is None:
            s.missed_unexpressed.append(u["any"])
            continue
        s.unexp_found += 1
        s.unexp_kind_right += int("kind" not in u or u["kind"] == found.kind)

    s.latency_s = latency_s
    s.cost_usd = result.cost_usd
    s.input_tokens = result.usage.input_tokens
    s.output_tokens = result.usage.output_tokens
    s.cache_read = result.usage.cache_read_input_tokens
    s.cache_write = result.usage.cache_creation_input_tokens
    s.repaired = len(result.attempts) > 1
    return s


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def aggregate(scores: list[CaseScore], *, timing: bool) -> dict[str, Any]:
    tp_pred = sum(s.tp_pred for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    tp_req = sum(s.tp_required for s in scores)
    n_req = sum(s.n_required for s in scores)
    precision = _ratio(tp_pred, n_pred)
    recall = _ratio(tp_req, n_req)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    statuses: dict[str, int] = {}
    for s in scores:
        statuses[s.status] = statuses.get(s.status, 0) + 1
    out: dict[str, Any] = {
        "cases": len(scores),
        "status": statuses,
        "exact_cases": sum(s.exact for s in scores),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hard_agreement": _ratio(
            sum(s.hard_right for s in scores), sum(s.hard_checked for s in scores)
        ),
        "hallucinated_id_rate": _ratio(
            sum(s.n_hallucinated for s in scores), sum(s.n_items_raw for s in scores)
        ),
        "unexpressed_recall": _ratio(
            sum(s.unexp_found for s in scores), sum(s.unexp_expected for s in scores)
        ),
        "unexpressed_kind_agreement": _ratio(
            sum(s.unexp_kind_right for s in scores), sum(s.unexp_found for s in scores)
        ),
        "repair_rounds": sum(s.repaired for s in scores),
    }
    if timing:
        lat = sorted(s.latency_s for s in scores if s.latency_s is not None)
        costs = [s.cost_usd for s in scores if s.cost_usd is not None]
        out["latency_s"] = (
            {
                "mean": statistics.fmean(lat),
                "p50": lat[len(lat) // 2],
                "p95": lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)],
            }
            if lat
            else None
        )
        out["cost_usd"] = (
            {"total": sum(costs), "mean_per_case": statistics.fmean(costs)} if costs else None
        )
        out["tokens_mean"] = {
            k: statistics.fmean(getattr(s, k) for s in scores) if scores else 0
            for k in ("input_tokens", "output_tokens", "cache_read", "cache_write")
        }
        out["cases_with_cache_read"] = sum(1 for s in scores if s.cache_read > 0)
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({t for s in scores for t in s.tags}):
        sub = [s for s in scores if tag in s.tags]
        by_tag[tag] = {
            "cases": len(sub),
            "exact": sum(s.exact for s in sub),
            "recall": _ratio(sum(s.tp_required for s in sub), sum(s.n_required for s in sub)),
            "precision": _ratio(sum(s.tp_pred for s in sub), sum(s.n_pred for s in sub)),
        }
    out["by_tag"] = by_tag
    return out


# --------------------------------------------------------------------------- runners


class _RecordingClient:
    """Wraps a client; keeps the raw JSON of every response (for replay)."""

    def __init__(self, client: Any):
        self._client = client
        self.responses: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        resp = self._client.beta.messages.create(**kwargs)
        self.responses.append(resp.to_dict())
        return resp


def replay(
    replay_dir: Path,
    cases: list[dict[str, Any]],
    problems_dir: Path = DEFAULT_PROBLEMS,
    *,
    repair: bool = True,
) -> tuple[list[CaseScore], list[str], list[str]]:
    """Score saved raw responses. Returns (scores, missing case ids, notes)."""
    from .offline import OfflineExhausted, offline_client

    scores, missing, notes = [], [], []
    for case in cases:
        path = replay_dir / f"{case['id']}.json"
        if not path.exists():
            missing.append(case["id"])
            continue
        saved = json.loads(path.read_text())
        if saved.get("prompt_version") != PROMPT_VERSION:
            notes.append(
                f"{case['id']}: recorded with {saved.get('prompt_version')}, "
                f"current prompt is {PROMPT_VERSION}"
            )
        client = offline_client(saved["responses"])
        try:
            result = parse_request(
                load_problem(case, problems_dir),
                case["text"],
                client=client,
                model=saved.get("model", DEFAULT_MODEL),
                effort=saved.get("effort", DEFAULT_EFFORT),
                repair=repair,
            )
        except OfflineExhausted:
            notes.append(f"{case['id']}: needs a repair response that wasn't recorded")
            continue
        latencies = saved.get("latency_s")
        scores.append(score_case(case, result, sum(latencies) if latencies else None))
    return scores, missing, notes


def _has_credentials_hint() -> bool:
    env = os.environ
    if any(env.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    if env.get("ANTHROPIC_FEDERATION_RULE_ID"):
        return True
    profile_dir = Path(env.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic"
    return profile_dir.is_dir() and any(profile_dir.iterdir())


_NO_CREDS = (
    "No Anthropic API credentials found. --live needs ANTHROPIC_API_KEY (or "
    "ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile). Nothing was sent. "
    "Use --replay DIR to score saved responses offline."
)


def live(
    cases: list[dict[str, Any]],
    out_dir: Path,
    problems_dir: Path = DEFAULT_PROBLEMS,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    repair: bool = True,
    workers: int = 4,
) -> tuple[list[CaseScore], dict[str, Any]]:
    import anthropic

    if not _has_credentials_hint():
        raise SystemExit(_NO_CREDS)
    client = anthropic.Anthropic()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact token counts for one request and for the static prefix (count_tokens is free).
    first = cases[0]
    kwargs = build_request(load_problem(first, problems_dir), first["text"], model=model)
    count_kwargs = {k: kwargs[k] for k in ("model", "system", "messages", "output_config")}
    try:
        full = client.beta.messages.count_tokens(**count_kwargs).input_tokens
        tiny = dict(count_kwargs, messages=[{"role": "user", "content": "x"}])
        prefix = client.beta.messages.count_tokens(**tiny).input_tokens
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise SystemExit(f"{_NO_CREDS}\n({type(e).__name__}: {e})") from e
    except TypeError as e:  # raised by the SDK when no auth method resolves
        if "authentication" not in str(e).lower():
            raise
        raise SystemExit(_NO_CREDS) from e
    meta = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "effort": effort,
        "repair": repair,
        "count_tokens": {"first_case_request": full, "static_prefix_plus_tiny_message": prefix},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def run(case: dict[str, Any]) -> CaseScore:
        rec = _RecordingClient(client)
        problem = load_problem(case, problems_dir)
        try:
            result = parse_request(
                problem, case["text"], client=rec, model=model, effort=effort, repair=repair
            )
        except anthropic.APIError as e:
            result = ParseResult(status="error", message=f"{e}", model=model, effort=effort)
        record = {
            "case_id": case["id"],
            "synthetic": False,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "effort": effort,
            "latency_s": [a.latency_s for a in result.attempts],
            "responses": rec.responses,
            "result": result.model_dump(mode="json"),
        }
        (out_dir / f"{case['id']}.json").write_text(json.dumps(record, indent=1))
        return score_case(case, result, result.latency_s)

    # First case alone so it writes the prompt cache before the others read it.
    scores = [run(cases[0])]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scores += list(pool.map(run, cases[1:]))
    return scores, meta


# --------------------------------------------------------------------------- CLI


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def report(metrics: dict[str, Any], scores: list[CaseScore], *, verbose: bool) -> str:
    lines = [f"cases scored: {metrics['cases']}   status: {metrics['status']}"]
    for k in (
        "precision",
        "recall",
        "f1",
        "hard_agreement",
        "hallucinated_id_rate",
        "unexpressed_recall",
        "unexpressed_kind_agreement",
        "exact_cases",
        "repair_rounds",
    ):
        lines.append(f"  {k:28s} {_fmt(metrics[k])}")
    for k in ("latency_s", "cost_usd", "tokens_mean", "cases_with_cache_read"):
        if k in metrics:
            lines.append(f"  {k:28s} {metrics[k]}")
    lines.append("by tag:")
    for tag, m in metrics["by_tag"].items():
        lines.append(
            f"  {tag:14s} cases={m['cases']:3d} exact={m['exact']:3d} "
            f"recall={_fmt(m['recall'])} precision={_fmt(m['precision'])}"
        )
    if verbose:
        for s in scores:
            if s.exact:
                continue
            lines.append(f"- {s.case_id} [{s.status}]")
            for e in s.missed:
                lines.append(f"    missed     {e}")
            for c in s.false_positives:
                brief = {k: v for k, v in c.items() if k not in ("rationale", "origin", "weight")}
                lines.append(f"    extra      {brief}")
            for u in s.missed_unexpressed:
                lines.append(f"    unexpressed not reported: {u}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roomlayout.constraints.vlm.eval")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="call the API (needs credentials)")
    mode.add_argument("--replay", type=Path, metavar="DIR", help="score saved responses")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--out", type=Path, help="live: where to save responses")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", type=Path, help="also write metrics JSON here")
    ap.add_argument("-v", "--verbose", action="store_true", help="list per-case misses")
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    meta: dict[str, Any] = {}
    if args.live:
        out = args.out or REPO_ROOT / "out" / "vlm_eval" / time.strftime(
            f"%Y%m%dT%H%M%SZ-{PROMPT_VERSION}-{args.effort}", time.gmtime()
        )
        scores, meta = live(
            cases,
            out,
            args.problems,
            model=args.model,
            effort=args.effort,
            repair=not args.no_repair,
            workers=args.workers,
        )
        print(f"saved raw responses to {out}")
    else:
        scores, missing, notes = replay(
            args.replay, cases, args.problems, repair=not args.no_repair
        )
        for n in notes:
            print(f"note: {n}")
        if missing:
            print(f"{len(missing)} case(s) have no saved response in {args.replay} (skipped)")
        if not scores:
            print("nothing to score", file=sys.stderr)
            return 2

    metrics = aggregate(scores, timing=True)
    metrics["meta"] = meta
    print(report(metrics, scores, verbose=args.verbose))
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
