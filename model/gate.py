"""
gate.py

The deployment gate: compares a candidate's metrics (evaluate.py's
output) against the committed baseline in metrics/baseline.json, and
decides whether the candidate is allowed to ship.

Rule: a gated metric counts as regressed only if BOTH
  1. it dropped more than MAX_DROP below the baseline, and
  2. the drop is statistically significant - the 95% paired-bootstrap
     confidence interval for the change lies entirely below zero.
Either on its own isn't enough. A 1-point drop on 500 queries is 5
questions, which can easily be a handful of borderline queries flipping
rather than a genuinely worse model; and a significant-but-tiny drop
isn't worth blocking a PR over. Someone reading a red build can see
exactly which number moved, by how much, and how sure we are.

Baseline updates ratchet (see next_baseline): the baseline only moves
when a candidate is at least as good on every gated metric. A candidate
that passes with a small or not-yet-significant drop ships, but the
baseline stays where it was - so repeated small drops keep being
measured against the same bar, and add up until they're significant
instead of each one quietly resetting it. A deliberate tradeoff is
accepted explicitly, via allow_regression, which resets the baseline.

Two runs are only compared if they were scored on the exact same query
list (eval_set_sha256): the significance test pairs queries by
position, and a refreshed data snapshot can change which queries are
held out without changing how many there are.

    python -m model.gate check --latest metrics/latest.json [--summary-out gate.md]
    python -m model.gate update-baseline --latest metrics/latest.json [--reset]
"""

import argparse
import json
import os
import random
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
# means the same thing for a 0.42 metric as a 0.68 one.
MAX_DROP = 0.01

# Paired bootstrap settings. Seeded, so the gate's verdict for a given
# pair of result files never changes between runs.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95

# How to score one query from its rank (None = correct code not in the
# top 5), for each metric the significance test can be run on.
PER_QUERY = {
    "top1": lambda r: 1.0 if r == 1 else 0.0,
    "top3": lambda r: 1.0 if r is not None and r <= 3 else 0.0,
    "top5": lambda r: 1.0 if r is not None and r <= 5 else 0.0,
    "mrr": lambda r: 1.0 / r if r is not None else 0.0,
}


def paired_bootstrap_ci(
    baseline_ranks: list[int | None],
    candidate_ranks: list[int | None],
    metric: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Confidence interval for (candidate - baseline) on `metric`. Resamples
    queries with replacement and keeps each query's baseline/candidate
    pair together - the pairing is what makes this sensitive: most
    queries score the same under both models, so only the ones that
    actually changed contribute to the difference.
    """
    score = PER_QUERY[metric]
    diffs = [score(c) - score(b) for b, c in zip(baseline_ranks, candidate_ranks)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - CONFIDENCE) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


@dataclass(frozen=True)
class MetricResult:
    name: str
    baseline: float
    candidate: float
    gated: bool
    ci: tuple[float, float] | None = None  # None: no per-query data to test with

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def exceeds_tolerance(self) -> bool:
        # Rounded so a delta of exactly -MAX_DROP (-0.0100000001 in float)
        # counts as "at the limit", not "over it".
        return round(self.delta, 6) < -MAX_DROP

    @property
    def significant_drop(self) -> bool:
        # Without per-query data there's nothing to test - fall back to
        # the tolerance alone rather than silently passing.
        return self.ci is None or self.ci[1] < 0

    @property
    def regressed(self) -> bool:
        return self.gated and self.exceeds_tolerance and self.significant_drop


@dataclass(frozen=True)
class GateResult:
    metrics: list[MetricResult]
    comparable: bool
    reason: str = ""
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.comparable and not any(m.regressed for m in self.metrics)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _not_comparable_reason(candidate: dict, baseline: dict) -> str | None:
    if candidate["n_queries"] != baseline["n_queries"]:
        return f"eval set changed: {baseline['n_queries']} -> {candidate['n_queries']} queries"
    if candidate.get("eval_set_sha256") != baseline.get("eval_set_sha256"):
        return (
            "eval set changed: same number of queries, but different ones "
            "(eval_set_sha256 differs - e.g. a refreshed data snapshot)"
        )
    missing = [name for name in GATED_METRICS if name not in candidate["metrics"]]
    if missing:
        return f"candidate is missing gated metrics: {missing}"
    return None


def compare(candidate: dict, baseline: dict) -> GateResult:
    reason = _not_comparable_reason(candidate, baseline)
    if reason:
        # Numbers from different eval sets aren't measuring the same thing -
        # refuse to call that a pass or a regression.
        return GateResult(metrics=[], comparable=False, reason=reason)

    base_ranks = baseline.get("per_query_ranks")
    cand_ranks = candidate.get("per_query_ranks")
    have_ranks = base_ranks is not None and cand_ranks is not None and len(base_ranks) == len(cand_ranks)

    results = [
        MetricResult(
            name=name,
            baseline=value,
            candidate=candidate["metrics"][name],
            gated=name in GATED_METRICS,
            ci=paired_bootstrap_ci(base_ranks, cand_ranks, name) if have_ranks and name in PER_QUERY else None,
        )
        for name, value in baseline["metrics"].items()
        if name in candidate["metrics"]
    ]

    notes = []
    if not have_ranks:
        notes.append("No per-query ranks to test significance with - gating on the tolerance alone.")
    base_data = baseline.get("model", {}).get("data_sha256")
    cand_data = candidate.get("model", {}).get("data_sha256")
    if base_data and cand_data and base_data != cand_data:
        # Same queries, different index corpus: still a fair comparison
        # (the corpus is part of the model), but worth flagging.
        notes.append("Index data changed since the baseline (data_sha256 differs); eval queries are unchanged.")
    return GateResult(metrics=results, comparable=True, notes=tuple(notes))


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


def _status(m: MetricResult) -> str:
    if m.regressed:
        return "FAIL"
    if m.gated and m.exceeds_tolerance:
        return "down, not significant"
    return "up" if m.delta > 0 else ("down" if m.delta < 0 else "=")


def render_summary(result: GateResult, allow_regression: bool = False) -> str:
    if not result.comparable:
        verdict = f"**Not comparable** - {result.reason}"
    elif result.passed:
        verdict = f"**Passed** - no gated metric had a significant drop of more than {MAX_DROP:.2f}"
    else:
        verdict = f"**Regression** - at least one gated metric had a significant drop of more than {MAX_DROP:.2f}"
    if allow_regression and not result.passed:
        verdict += " (accepted via `accept-regression` - baseline will be reset on merge)"

    lines = ["## Model regression gate", "", verdict, ""]
    if result.metrics:
        lines += [
            f"| metric | baseline | candidate | delta | {CONFIDENCE:.0%} CI of delta | gated | |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in result.metrics:
            ci = f"[{m.ci[0]:+.4f}, {m.ci[1]:+.4f}]" if m.ci else "-"
            lines.append(
                f"| {m.name} | {m.baseline:.4f} | {m.candidate:.4f} | {m.delta:+.4f} | {ci} | "
                f"{'yes' if m.gated else 'no'} | {_status(m)} |"
            )
        lines.append("")
    lines += [f"_{note}_" for note in result.notes]
    return "\n".join(lines).rstrip() + "\n"


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
    check.add_argument("--summary-out", type=Path, help="also write the markdown summary here (for a PR comment)")

    update = sub.add_parser("update-baseline", help="ratchet the baseline forward after a successful ship")
    update.add_argument("--latest", type=Path, required=True)
    update.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    update.add_argument("--reset", action="store_true", help="accept the candidate as the new baseline outright")

    args = parser.parse_args(argv)

    if args.command == "check":
        result = compare(load(args.latest), load(args.baseline))
        summary = render_summary(result, allow_regression=args.allow_regression)
        print(summary)
        _write_step_summary(summary)
        if args.summary_out:
            args.summary_out.write_text(summary, encoding="utf-8")
        return 0 if result.passed or args.allow_regression else 1

    baseline = load(args.baseline)
    new = next_baseline(load(args.latest), baseline, allow_regression=args.reset)
    if new is baseline:
        print("Baseline unchanged (candidate was not >= baseline on every gated metric).")
    else:
        # Copied byte-for-byte rather than re-serialized, to keep
        # evaluate.py's layout (per-query ranks on one line).
        args.baseline.write_text(args.latest.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Baseline updated -> {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
