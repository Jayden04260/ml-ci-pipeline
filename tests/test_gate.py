import json

from model import gate
from model.gate import MAX_DROP, compare, next_baseline, render_summary


def _result(top1=0.42, top5=0.68, mrr=0.52, heading_top1=0.56, n_queries=500):
    return {
        "metrics": {"top1": top1, "top5": top5, "mrr": mrr, "heading_top1": heading_top1},
        "n_queries": n_queries,
    }


def test_identical_metrics_pass():
    assert compare(_result(), _result()).passed


def test_improvement_passes():
    assert compare(_result(top1=0.50), _result()).passed


def test_drop_within_tolerance_passes():
    assert compare(_result(top1=0.42 - MAX_DROP / 2), _result()).passed


def test_drop_exactly_at_tolerance_passes():
    # 0.42 - 0.01 is 0.41000000000000003 in float - must not count as over the limit
    assert compare(_result(top1=0.42 - MAX_DROP), _result()).passed


def test_drop_beyond_tolerance_on_a_gated_metric_fails():
    result = compare(_result(mrr=0.52 - MAX_DROP - 0.001), _result())
    assert not result.passed
    assert [m.name for m in result.metrics if m.regressed] == ["mrr"]


def test_large_drop_on_an_ungated_metric_does_not_fail():
    assert compare(_result(heading_top1=0.10), _result()).passed


def test_different_eval_set_size_is_not_comparable():
    result = compare(_result(n_queries=400), _result())
    assert not result.comparable
    assert not result.passed
    assert "400" in result.reason


def test_missing_gated_metric_is_not_comparable():
    candidate = _result()
    del candidate["metrics"]["top5"]
    assert not compare(candidate, _result()).passed


def test_baseline_ratchets_up_when_every_gated_metric_is_at_least_as_good():
    candidate = _result(top1=0.45)
    assert next_baseline(candidate, _result()) is candidate


def test_baseline_holds_after_a_within_tolerance_drop():
    # passes the gate, ships - but must not lower the bar for the next PR
    baseline = _result()
    candidate = _result(top1=0.45, top5=0.68 - MAX_DROP / 2)
    assert compare(candidate, baseline).passed
    assert next_baseline(candidate, baseline) is baseline


def test_allow_regression_resets_baseline():
    candidate = _result(top1=0.20)
    assert next_baseline(candidate, _result(), allow_regression=True) is candidate


def test_summary_marks_the_failing_metric():
    summary = render_summary(compare(_result(top1=0.30), _result()))
    assert "Regression" in summary
    assert "| top1 | 0.4200 | 0.3000 | -0.1200 | yes | FAIL |" in summary


def test_cli_check_exit_codes(tmp_path):
    baseline = tmp_path / "baseline.json"
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    baseline.write_text(json.dumps(_result()))
    good.write_text(json.dumps(_result(top1=0.43)))
    bad.write_text(json.dumps(_result(top1=0.30)))

    assert gate.main(["check", "--latest", str(good), "--baseline", str(baseline)]) == 0
    assert gate.main(["check", "--latest", str(bad), "--baseline", str(baseline)]) == 1
    assert gate.main(["check", "--latest", str(bad), "--baseline", str(baseline), "--allow-regression"]) == 0


def test_cli_update_baseline_writes_only_when_ratcheting(tmp_path):
    baseline = tmp_path / "baseline.json"
    worse = tmp_path / "worse.json"
    better = tmp_path / "better.json"
    baseline.write_text(json.dumps(_result()))
    worse.write_text(json.dumps(_result(top1=0.415)))
    better.write_text(json.dumps(_result(top1=0.47)))

    gate.main(["update-baseline", "--latest", str(worse), "--baseline", str(baseline)])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.42

    gate.main(["update-baseline", "--latest", str(better), "--baseline", str(baseline)])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.47

    gate.main(["update-baseline", "--latest", str(worse), "--baseline", str(baseline), "--reset"])
    assert json.loads(baseline.read_text())["metrics"]["top1"] == 0.415
