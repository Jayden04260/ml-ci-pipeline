from model.references import Reference
from model.rerank import rerank


def _ref(code, title):
    return Reference(title=title, code=code, level="Heading")


def test_rerank_promotes_the_most_relevant_reference_to_top():
    query = "wireless bluetooth headphones"
    candidates = [
        (_ref("1", "abrasive stones, natural"), 0.3),
        (_ref("2", "headphones and earphones"), 0.3),
        (_ref("3", "abaci (abacus)"), 0.3),
    ]

    results = rerank(query, candidates, top_k=2)

    assert len(results) == 2
    assert results[0][0].title == "headphones and earphones"
    assert results[0][1] >= results[1][1]


def test_rerank_empty_candidates_returns_empty_list():
    assert rerank("anything", [], top_k=5) == []


def test_rerank_top_k_larger_than_candidates_returns_all():
    candidates = [(_ref("1", "only one"), 0.5)]
    results = rerank("query", candidates, top_k=10)
    assert len(results) == 1
