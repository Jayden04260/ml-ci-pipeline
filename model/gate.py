"""
gate.py

The deployment gate: compares a candidate's metrics (evaluate.py's
output) against the committed baseline in metrics/baseline.json, and
decides whether the candidate is allowed to ship.

Rule: every metric in GATED_METRICS must be no more than MAX_DROP below
the baseline. Deliberately a single, legible comparison - someone reading
a red build should be able to see exactly which number moved and by how
much, without trusting a black-box score.

Baseline updates ratchet (see next_baseline): the baseline only moves
when a candidate is at least as good on every gated metric. A candidate
that passes the gate with a small, within-tolerance drop ships, but the
baseline stays where it was - otherwise a series of individually
tolerable 0.9-point drops could quietly walk accuracy down with every
PR green. A deliberate tradeoff (e.g. a slower-but-better model that
loses a little top-5) is accepted explicitly, via allow_regression,
which resets the baseline to the candidate's numbers.

    python -m model.gate check --latest metrics/latest.json
    python -m model.gate update-baseline --latest metrics/latest.json [--reset]
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASELINE_PATH = Path("metrics/baseline.json")

# Metrics the gate enforces. top1 is what a user sees first; top5 is
# whether the right code is on screen at all; MRR summarizes the whole
# ranking. heading/chapter accuracy are reported but not gated - they
# mostly move with top1 and would add noise to the pass/fail decision.
GATED_METRICS = ("top1", "top5", "mrr")

# Maximum allowed drop, in absolute points (0.01 = one percentage point =
# 5 of the 500 eval queries). Absolute rather than relative so the rule
# means the same thing for a 0.42 metric as a 0.68 one. The eval is
# deterministic, so this tolerance isn't absorbing run-to-run randomness -
# it's the size of change considered too small to block a PR over.
MAX_DROP = 0.01


@dataclass(frozen=True)
class MetricResult:
    name: str
    baseline: float
    candidate: float
    gated: bool

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def regressed(self) -> bool:
        # Rounded so a delta of exactly -MAX_DROP (-0.0100000001 in float)
        # counts as "at the limit", not "over it".
        return self.gated and round(self.delta, 6) < -MAX_DROP


@dataclass(frozen=True)
class GateResult:
    metrics: list[MetricResult]
    comparable: bool
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.comparable and not any(m.regressed for m in self.metrics)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(candidate: dict, baseline: dict) -> GateResult:
    if candidate["n_queries"] != baseline["n_queries"]:
        # Different eval set sizes mean the numbers aren't measuring the
        # same thing - refuse to call that a pass or a regression.
        return GateResult(
            metrics=[],
            comparable=False,
            reason=f"eval set changed: {baseline['n_queries']} -> {candidate['n_queries']} queries",
        )

    results = [
        MetricResult(name=name, baseline=value, candidate=candidate["metrics"][name], gated=name in GATED_METRICS)
        for name, value in baseline["metrics"].items()
        if name in candidate["metrics"]
    ]
    missing = [name for name in GATED_METRICS if name not in candidate["metrics"]]
    if missing:
        return GateResult(metrics=results, comparable=False, reason=f"candidate is missing gated metrics: {missing}")
    return GateResult(metrics=results, comparable=True)


def next_baseline(candidate: dict, baseline: dict, allow_regression: bool = False) -> dict:
    """
    The baseline to commit after `candidate` ships. Ratchets: only
    replaced when the candidate is >= baseline on every gated metric,
    or when a regression was explicitly accepted.
    """
    if allow_regression:
        return candidate
    result = compare(candidate, baseline)
    if result.comparable and all(m.delta >= 0 for m in result.metrics if m.gated):
        return candidate
    return baseline


def render_summary(result: GateResult, allow_regression: bool = False) -> str:
    if not result.comparable:
        verdict = f"**Not comparable** - {result.reason}"
    elif result.passed:
        verdict = f"**Passed** - no gated metric dropped more than {MAX_DROP:.2f}"
    else:
        verdict = f"**Regression** - at least one gated metric dropped more than {MAX_DROP:.2f}"
    if allow_regression and not result.passed:
        verdict += " (accepted via `accept-regression` - baseline will be reset on merge)"

    lines = [
        "## Model regression gate",
        "",
        verdict,
        "",
        "| metric | baseline | candidate | delta | gated | |",
        "|---|---|---|---|---|---|",
    ]
    for m in result.metrics:
        status = "FAIL" if m.regressed else ("up" if m.delta > 0 else ("down" if m.delta < 0 else "="))
        lines.append(
            f"| {m.name} | {m.baseline:.4f} | {m.candidate:.4f} | {m.delta:+.4f} | "
            f"{'yes' if m.gated else 'no'} | {status} |"
        )
    return "\n".join(lines) + "\n"


def _write_step_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="fail (exit 1) if the candidate regresses against the baseline")
    check.add_argument("--latest", type=Path, required=True)
    check.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    check.add_argument("--allow-regression", action="store_true")

    update = sub.add_parser("update-baseline", help="ratchet the baseline forward after a successful ship")
    update.add_argument("--latest", type=Path, required=True)
    update.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    update.add_argument("--reset", action="store_true", help="accept the candidate as the new baseline outright")

    args = parser.parse_args(argv)
    candidate = load(args.latest)

    if args.command == "check":
        result = compare(candidate, load(args.baseline))
        summary = render_summary(result, allow_regression=args.allow_regression)
        print(summary)
        _write_step_summary(summary)
        return 0 if result.passed or args.allow_regression else 1

    baseline = load(args.baseline)
    new = next_baseline(candidate, baseline, allow_regression=args.reset)
    if new is baseline:
        print("Baseline unchanged (candidate was not >= baseline on every gated metric).")
    else:
        args.baseline.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline updated -> {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
