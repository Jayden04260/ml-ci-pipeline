"""
embeddings.py

Thin wrapper around sentence-transformers for turning text into
L2-normalized embedding vectors. All-MiniLM-L6-v2 (the default) is a
~80MB model that runs comfortably on CPU - no GPU required. Same model
and wrapper shape as this author's local-rag project, reused here
because the underlying problem is the same: embed text, compare by
cosine similarity.

L2-normalizing here (rather than at the call site) means every
embedding this module ever produces is guaranteed comparable via plain
dot product, regardless of which code path created it.
"""

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_model_name = None


def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """
    Embed a list of strings into an (n, dim) array of L2-normalized
    float32 vectors. Empty input returns an (0, dim)-shaped array rather
    than erroring, so callers don't need to special-case an empty index.
    """
    model = _get_model(model_name)
    if not texts:
        return np.zeros((0, model.get_embedding_dimension()), dtype=np.float32)

    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Embed a single query string, returning a 1D (dim,) vector."""
    return embed_texts([query], model_name)[0]
