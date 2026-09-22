"""
Sanity, shape, and edge-case checks on the train -> evaluate path.

Embeddings are faked (deterministic per-title random unit vectors) so
these run in milliseconds without downloading a model; the real models
are exercised by test_rerank.py and, end to end, by the deploy-gate
workflow's evaluate step.
"""

import hashlib

import numpy as np
import pytest

from model import evaluate, train
from model.index import ReferenceIndex
from model.references import DEFAULT_SEED, Reference, load_references, split_train_eval

DIM = 8


def _fake_embed_texts(texts):
    rows = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
        rows.append(v / np.linalg.norm(v))
    return np.array(rows, dtype=np.float32).reshape(len(texts), DIM)


def _ref(code, title="x"):
    return Reference(title=title, code=code, level="Heading")


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(train.embeddings, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(evaluate, "embed_texts", _fake_embed_texts)


# --- real data sanity -------------------------------------------------------


def test_real_data_codes_are_ten_digits():
    for ref in load_references():
        assert len(ref.code) == 10 and ref.code.isdigit(), ref


def test_real_split_leaves_enough_eval_queries_for_the_gate():
    _index, eval_refs = split_train_eval(load_references(), seed=DEFAULT_SEED)
    assert len(eval_refs) >= evaluate.MAX_EVAL_QUERIES


def test_eval_queries_are_deterministic_and_capped():
    first = evaluate.eval_queries()
    assert len(first) == evaluate.MAX_EVAL_QUERIES
    assert first == evaluate.eval_queries()


def test_index_corpus_never_contains_an_eval_query():
    indexed = {(r.title, r.code) for r in train.build_index_references()}
    assert not any((r.title, r.code) in indexed for r in evaluate.eval_queries())


# --- artifact shape ---------------------------------------------------------


def test_train_writes_a_loadable_artifact_with_matching_shapes(tmp_path, fake_embeddings):
    manifest = train.train(tmp_path)
    index, loaded_manifest = train.load_artifact(tmp_path)

    assert loaded_manifest == manifest
    assert index.embeddings.shape == (manifest["n_references"], DIM)
    assert index.embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-5)
    assert len(manifest["data_sha256"]) == 64


def test_load_artifact_rejects_mismatched_rows(tmp_path):
    train.save_artifact(tmp_path, np.zeros((3, DIM), dtype=np.float32), [_ref("0000000001")], {})
    with pytest.raises(ValueError):
        train.load_artifact(tmp_path)


# --- metrics edge cases -----------------------------------------------------


def test_compute_metrics_all_correct_at_rank_one():
    refs = [_ref("0101000000"), _ref("0202000000")]
    metrics = evaluate.compute_metrics([1, 1], refs, refs)
    assert metrics == {"top1": 1.0, "top3": 1.0, "top5": 1.0, "mrr": 1.0, "heading_top1": 1.0, "chapter_top1": 1.0}


def test_compute_metrics_misses_and_partial_matches():
    expected = [_ref("0101100000"), _ref("0202000000"), _ref("0303000000"), _ref("0404000000")]
    top1 = [
        _ref("0101100000"),  # exact
        _ref("0202990000"),  # same heading, wrong code
        _ref("0399000000"),  # same chapter only
        None,  # nothing came back
    ]
    metrics = evaluate.compute_metrics([1, 2, 4, None], expected, top1)

    assert metrics["top1"] == 0.25
    assert metrics["top3"] == 0.5
    assert metrics["top5"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 4)
    assert metrics["heading_top1"] == 0.5
    assert metrics["chapter_top1"] == 0.75


def test_compute_metrics_rejects_empty_eval_set():
    with pytest.raises(ValueError):
        evaluate.compute_metrics([], [], [])


def test_leakage_check_catches_an_eval_query_in_the_index():
    leaked = _ref("0101000000", "horses")
    index = ReferenceIndex(np.zeros((1, DIM), dtype=np.float32), [leaked])
    with pytest.raises(ValueError, match="present in the index"):
        evaluate.check_no_leakage(index, [leaked])


def test_evaluate_perfect_index_scores_one(fake_embeddings):
    # Every query's own title is in the index, so cosine search must put
    # it first - an end-to-end check of the evaluate loop's plumbing.
    refs = [_ref(f"{i:02d}01000000", f"product number {i}") for i in range(10)]
    index = ReferenceIndex(_fake_embed_texts([r.title for r in refs]), refs)

    metrics = evaluate.evaluate(index, refs, use_rerank=False)
    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
