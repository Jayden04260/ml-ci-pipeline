"""
index.py

Hand-rolled top-k cosine-similarity search over embedded reference
titles, instead of pulling in a vector-database dependency. At this
scale - a few thousand reference titles, not a web-scale corpus - a
plain numpy matrix multiply is fast enough, has no server to run, and
is simple enough to unit test directly. Same design as this author's
local-rag project's retrieve.py.
"""

import numpy as np

from model.references import Reference


class ReferenceIndex:
    """Holds one embedding vector per Reference. Embeddings are expected
    to already be L2-normalized (embeddings.embed_texts does this) so
    cosine similarity reduces to a plain dot product."""

    def __init__(self, embeddings: np.ndarray, references: list[Reference]):
        if embeddings.shape[0] != len(references):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but {len(references)} references were given"
            )
        self.embeddings = embeddings
        self.references = references

    def __len__(self) -> int:
        return len(self.references)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[Reference, float]]:
        """
        Return the top_k references most similar to query_embedding, as
        (Reference, score) pairs, highest score first. query_embedding
        must be L2-normalized, same as the index's own embeddings.
        """
        if len(self) == 0:
            return []

        scores = self.embeddings @ query_embedding
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.references[i], float(scores[i])) for i in top_indices]
