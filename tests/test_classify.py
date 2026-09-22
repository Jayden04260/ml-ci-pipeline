import numpy as np

from model import classify
from model.classify import Candidate, _dedupe_by_code, confidence_gap
from model.references import Reference


def _ref(code, title, level="Heading"):
    return Reference(title=title, code=code, level=level)


def test_dedupe_by_code_keeps_highest_scoring_synonym_per_code():
    candidates = [
        (_ref("0307000000", "abalone"), 0.5),
        (_ref("0307000000", "abalone (fresh)"), 0.8),
        (_ref("1605570000", "abalone, prepared or preserved"), 0.6),
    ]
    result = _dedupe_by_code(candidates)

    assert len(result) == 2
    assert result[0][0].title == "abalone (fresh)"  # higher-scoring synonym of 0307000000 wins
    assert result[0][1] == 0.8
    assert result[1][0].code == "1605570000"


def test_dedupe_by_code_sorts_by_score_descending():
    candidates = [(_ref("1", "a"), 0.2), (_ref("2", "b"), 0.9), (_ref("3", "c"), 0.5)]
    result = _dedupe_by_code(candidates)
    assert [ref.code for ref, _ in result] == ["2", "3", "1"]


class _FakeIndex:
    def __init__(self, results):
        self._results = results

    def search(self, query_embedding, top_k):
        return self._results


def test_classify_without_rerank_returns_deduped_candidates(monkeypatch):
    monkeypatch.setattr(classify, "embed_query", lambda q: np.zeros(3, dtype=np.float32))
    candidates_raw = [
        (_ref("0307000000", "abalone"), 0.5),
        (_ref("0307000000", "abalone (fresh)"), 0.8),
    ]
    index = _FakeIndex(candidates_raw)

    result = classify.classify("shellfish", index, top_k=5, use_rerank=False)

    assert result == [Candidate(code="0307000000", title="abalone (fresh)", level="Heading", score=0.8)]


def test_classify_with_rerank_uses_rerank_scores(monkeypatch):
    monkeypatch.setattr(classify, "embed_query", lambda q: np.zeros(3, dtype=np.float32))
    raw = [(_ref("1", "a"), 0.9), (_ref("2", "b"), 0.1)]
    index = _FakeIndex(raw)
    # rerank flips the order relative to cosine score
    flipped = [(_ref("2", "b"), 5.0), (_ref("1", "a"), 1.0)]
    monkeypatch.setattr(classify, "rerank_candidates", lambda q, cands, top_k: flipped)

    result = classify.classify("query", index, top_k=5, use_rerank=True)

    assert [c.code for c in result] == ["2", "1"]


def test_classify_respects_top_k(monkeypatch):
    monkeypatch.setattr(classify, "embed_query", lambda q: np.zeros(3, dtype=np.float32))
    raw = [(_ref(str(i), f"item {i}"), 1.0 - i * 0.01) for i in range(10)]
    index = _FakeIndex(raw)

    result = classify.classify("query", index, top_k=3, use_rerank=False)
    assert len(result) == 3


def test_confidence_gap_is_difference_between_top_two_scores():
    candidates = [
        Candidate(code="1", title="a", level="Heading", score=5.0),
        Candidate(code="2", title="b", level="Heading", score=2.0),
        Candidate(code="3", title="c", level="Heading", score=1.0),
    ]
    assert confidence_gap(candidates) == 3.0


def test_confidence_gap_none_with_fewer_than_two_candidates():
    assert confidence_gap([]) is None
    assert confidence_gap([Candidate(code="1", title="a", level="Heading", score=5.0)]) is None
