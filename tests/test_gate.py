import json

import pytest

from model import gate
from model.gate import MAX_DROP, compare, next_baseline, paired_bootstrap_ci, render_summary

EVAL_SET = "a" * 64


def _result(ranks, eval_set=EVAL_SET, data="d" * 64):
    """An evaluate.py-shaped result whose metrics are computed from `ranks`,
    so the headline numbers and the per-query data always agree."""
    n = len(ranks)
    return {
        "metrics": {
            "top1": sum(1 for r in ranks if r == 1) / n,
            "top5": sum(1 for r in ranks if r is not None and r <= 5) / n,
            "mrr": sum(1 / r for r in ranks if r is not None) / n,
        },
        "n_queries": n,
        "eval_set_sha256": eval_set,
        "model": {"data_sha256": data},
        "per_query_ranks": ranks,
    }


# 500 queries: 210 right at rank 1 (top1 = 0.42), 130 at rank 3, 160 misses.
BASE = [1] * 210 + [3] * 130 + [None] * 160


def _flip(ranks, n, frm, to):
    """Change the first n queries currently at rank `frm` to rank `to`."""
    out, done = list(ranks), 0
    for i, r in enumerate(out):
        if done < n and r == frm:
            out[i], done = to, done + 1
    return out


def test_identical_results_pass():
    assert compare(_result(BASE), _result(BASE)).passed


def test_improvement_passes():
    assert compare(_result(_flip(BASE, 20, None, 1)), _result(BASE)).passed


def test_large_real_drop_fails():
    # 30 queries lose their rank-1 hit entirely: top1 -0.06, top5 -0.06
    result = compare(_result(_flip(BASE, 30, 1, None)), _result(BASE))
    assert not result.passed
    assert {m.name for m in result.metrics if m.regressed} == {"top1", "top5", "mrr"}


def test_drop_within_tolerance_passes():
    assert compare(_result(_flip(BASE, 4, 1, None)), _result(BASE)).passed


def test_drop_exactly_at_tolerance_passes():
    # 5 of 500 = exactly MAX_DROP - "at the limit" is allowed
    result = compare(_result(_flip(BASE, 5, 1, 3)), _result(BASE))
    top1 = next(m for m in result.metrics if m.name == "top1")
    assert top1.delta == pytest.approx(-MAX_DROP)
    assert result.passed


def test_drop_over_tolerance_but_not_significant_passes():
    # 10 queries improve (rank 3 -> 1) and 16 get worse (1 -> 3): net top1
    # -0.012, over the tolerance - but a 16-vs-10 split of changed queries
    # is well within what chance produces (exact McNemar p ~= 0.33).
    # (Improvements flipped first, so the second flip can't re-pick them.)
    cand = _flip(_flip(BASE, 10, 3, 1), 16, 1, 3)
    result = compare(_result(cand), _result(BASE))
    top1 = next(m for m in result.metrics if m.name == "top1")
    assert top1.exceeds_tolerance
    assert not top1.significant_drop
    assert result.passed
    assert "down, not significant" in render_summary(result)


def test_without_per_query_ranks_the_tolerance_alone_decides():
    cand, base = _result(_flip(BASE, 7, 1, 3)), _result(BASE)
    del cand["per_query_ranks"], base["per_query_ranks"]
    result = compare(cand, base)
    assert not result.passed
    assert "tolerance alone" in render_summary(result)


def test_different_eval_set_size_is_not_comparable():
    result = compare(_result(BASE[:400]), _result(BASE))
    assert not result.comparable
    assert not result.passed
    assert "500 -> 400" in result.reason


def test_same_size_but_different_queries_is_not_comparable():
    result = compare(_result(BASE, eval_set="b" * 64), _result(BASE))
    assert not result.comparable
    assert "different ones" in result.reason


def test_index_data_change_is_compared_but_flagged():
    result = compare(_result(BASE, data="e" * 64), _result(BASE))
    assert result.passed
    assert "Index data changed" in render_summary(result)


def test_missing_gated_metric_is_not_comparable():
    candidate = _result(BASE)
    del candidate["metrics"]["top5"]
    assert not compare(candidate, _result(BASE)).passed


def test_bootstrap_ci_is_deterministic_and_brackets_the_true_change():
    cand = _flip(BASE, 30, 1, None)
    lo, hi = paired_bootstrap_ci(BASE, cand, "top1")
    assert (lo, hi) == paired_bootstrap_ci(BASE, cand, "top1")
    assert lo < -0.06 < hi < 0


def test_bootstrap_ci_of_identical_runs_is_zero():
    assert paired_bootstrap_ci(BASE, BASE, "mrr") == (0.0, 0.0)


def test_baseline_ratchets_up_when_every_gated_metric_is_at_least_as_good():
    candidate = _result(_flip(BASE, 10, None, 1))
    assert next_baseline(candidate, _result(BASE)) is candidate


def test_baseline_holds_after_a_drop_that_passed_the_gate():
    # passes the gate, ships - but must not lower the bar for the next PR
    baseline = _result(BASE)
    candidate = _result(_flip(BASE, 4, 1, None))
    assert compare(candidate, baseline).passed
    assert next_baseline(candidate, baseline) is baseline


def test_allow_regression_resets_baseline():
    candidate = _result(_flip(BASE, 100, 1, None))
    assert next_baseline(candidate, _result(BASE), allow_regression=True) is candidate


def test_summary_marks_the_failing_metric_with_its_confidence_interval():
    summary = render_summary(compare(_result(_flip(BASE, 30, 1, None)), _result(BASE)))
    assert "**Regression**" in summary
    assert "| top1 | 0.4200 | 0.3600 | -0.0600 | [-" in summary
    assert "| yes | FAIL |" in summary


def _write(path, result):
    path.write_text(json.dumps(result))
    return str(path)


def test_cli_check_exit_codes_and_summary_file(tmp_path):
    baseline = _write(tmp_path / "baseline.json", _result(BASE))
    good = _write(tmp_path / "good.json", _result(_flip(BASE, 5, None, 1)))
    bad = _write(tmp_path / "bad.json", _result(_flip(BASE, 30, 1, None)))
    summary = tmp_path / "summary.md"

    assert gate.main(["check", "--latest", good, "--baseline", baseline]) == 0
    assert gate.main(["check", "--latest", bad, "--baseline", baseline, "--summary-out", str(summary)]) == 1
    assert "**Regression**" in summary.read_text()
    assert gate.main(["check", "--latest", bad, "--baseline", baseline, "--allow-regression"]) == 0


def test_cli_update_baseline_writes_only_when_ratcheting(tmp_path):
    baseline = tmp_path / "baseline.json"
    _write(baseline, _result(BASE))
    worse = _write(tmp_path / "worse.json", _result(_flip(BASE, 2, 1, None)))
    better = _write(tmp_path / "better.json", _result(_flip(BASE, 25, None, 1)))

    gate.main(["update-baseline", "--latest", worse, "--baseline", str(baseline)])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.42

    gate.main(["update-baseline", "--latest", better, "--baseline", str(baseline)])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.47

    gate.main(["update-baseline", "--latest", worse, "--baseline", str(baseline), "--reset"])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.416
