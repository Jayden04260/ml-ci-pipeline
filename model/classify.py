"""
classify.py

Orchestrates the pipeline: build a retrieval index over reference
titles, then for a free-text product description, retrieve and
optionally re-rank candidate commodity codes.

Deduplication matters here in a way it doesn't for a plain document RAG
pipeline: 1,624 of the 3,071 distinct codes in this data have more than
one reference title ("abalone" and "abalone, prepared or preserved" are
different titles that could both legitimately match a query, but they
point at different codes so that's fine; more often, near-duplicate
synonyms of the *same* code shouldn't each occupy a separate slot in
the top-k). Candidates are deduplicated by code, keeping each code's
best-scoring synonym match, so the final top_k is k distinct candidate
codes, not k possibly-repeated ones.
"""

from dataclasses import dataclass

from model.embeddings import embed_query, embed_texts
from model.index import ReferenceIndex
from model.references import Reference
from model.rerank import rerank as rerank_candidates

# How many candidates cosine search retrieves (and, if reranking, how
# many survive reranking) before deduplication narrows down to top_k
# distinct codes. Wider than top_k on purpose - several of these raw
# hits often collapse into the same code once deduplicated, so top_k
# alone wouldn't leave enough room for top_k *distinct* codes to survive.
SEARCH_POOL = 5

# Below this score gap between the top and second candidate, treat the
# classification as low-confidence. Picked from eval_classification.py's
# confidence-gap quartile breakdown (run against the full search_refs +
# nomenclature index, re-ranked - the configuration this actually ships
# with): queries in the bottom two quartiles (gap < ~1.6) averaged 31%
# top-1 accuracy; the top two quartiles (gap >= ~1.6) averaged 53% - a
# real, measured split, not a guessed number. Same reasoning as
# local-rag's NO_ANSWER_THRESHOLD: don't present a low-confidence guess
# with the same visual weight as a confident one.
CONFIDENCE_GAP_THRESHOLD = 1.6


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str  # the reference title that produced this match
    level: str  # Chapter/Heading/Subheading/Commodity - see references.Reference
    score: float


def build_index(references: list[Reference]) -> ReferenceIndex:
    vectors = embed_texts([r.title for r in references])
    return ReferenceIndex(vectors, references)


def _dedupe_by_code(candidates: list[tuple[Reference, float]]) -> list[tuple[Reference, float]]:
    best_per_code: dict[str, tuple[Reference, float]] = {}
    for ref, score in candidates:
        current = best_per_code.get(ref.code)
        if current is None or score > current[1]:
            best_per_code[ref.code] = (ref, score)
    return sorted(best_per_code.values(), key=lambda pair: pair[1], reverse=True)


def classify(
    query: str,
    index: ReferenceIndex,
    top_k: int = 5,
    use_rerank: bool = True,
) -> list[Candidate]:
    """
    Return up to top_k distinct candidate commodity codes for `query`,
    highest confidence first.
    """
    query_vector = embed_query(query)
    raw = index.search(query_vector, top_k=SEARCH_POOL)

    if use_rerank:
        raw = rerank_candidates(query, raw, top_k=SEARCH_POOL)

    deduped = _dedupe_by_code(raw)[:top_k]
    return [Candidate(code=ref.code, title=ref.title, level=ref.level, score=score) for ref, score in deduped]


def confidence_gap(candidates: list[Candidate]) -> float | None:
    """
    Score gap between the top and second candidate - see
    eval_classification.py's confidence-bucket report for why this is a
    reasonable proxy for "how sure should you be about the top result":
    a large gap means the top candidate stood out clearly; a small one
    means the model was choosing between near-equally-plausible codes.
    None if there's only one (or zero) candidates - nothing to compare
    the top one against.
    """
    if len(candidates) < 2:
        return None
    return candidates[0].score - candidates[1].score
