import numpy as np
import pytest

from model.index import ReferenceIndex
from model.references import Reference


def _ref(code, title="x"):
    return Reference(title=title, code=code, level="Heading")


def test_search_returns_top_k_ordered_by_similarity():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0] / np.linalg.norm([0.9, 0.1, 0.0]),
        ],
        dtype=np.float32,
    )
    refs = [_ref("0000000001", "cat"), _ref("0000000002", "dog"), _ref("0000000003", "cat-ish")]
    index = ReferenceIndex(embeddings, refs)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = index.search(query, top_k=2)

    assert len(results) == 2
    assert results[0][0].title == "cat"
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][0].title == "cat-ish"
    assert results[0][1] >= results[1][1]


def test_search_top_k_larger_than_index_returns_everything():
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    index = ReferenceIndex(embeddings, [_ref("0000000001")])
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)
    assert len(results) == 1


def test_search_on_empty_index_returns_empty_list():
    index = ReferenceIndex(np.zeros((0, 3), dtype=np.float32), [])
    assert index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=4) == []


def test_len_matches_reference_count():
    embeddings = np.zeros((3, 2), dtype=np.float32)
    refs = [_ref(f"{i:010d}") for i in range(3)]
    index = ReferenceIndex(embeddings, refs)
    assert len(index) == 3


def test_mismatched_embeddings_and_references_raises():
    embeddings = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        ReferenceIndex(embeddings, [_ref("0000000001")])
