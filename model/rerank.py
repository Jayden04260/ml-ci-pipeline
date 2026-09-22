"""
rerank.py

Cross-encoder re-ranking of candidate references before final code
selection. Cosine similarity search (index.py) embeds the query and
each reference title independently, so it can miss the difference
between a reference that's topically similar and one that's actually
the right match. A cross-encoder scores a (query, reference title) pair
jointly through one transformer pass - too slow to run against the
whole corpus, but exactly right for re-scoring the small candidate set
cosine search already narrowed down. Same design as this author's
local-rag project's rerank.py.
"""

from model.references import Reference

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_model_name = None


def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(model_name)
        _model_name = model_name
    return _model


def rerank(
    query: str, candidates: list[tuple[Reference, float]], top_k: int, model_name: str = DEFAULT_MODEL_NAME
) -> list[tuple[Reference, float]]:
    """
    Re-score `candidates` against `query` with a cross-encoder and
    return the top_k, highest cross-encoder score first, as
    (Reference, rerank_score) pairs. The original cosine score isn't
    carried through - callers that need it should keep their own
    mapping from the input candidates.
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, ref.title) for ref, _ in candidates]
    scores = model.predict(pairs)

    scored = [(ref, float(score)) for (ref, _), score in zip(candidates, scores)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
