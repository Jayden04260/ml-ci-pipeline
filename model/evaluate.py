"""
evaluate.py

Scores a trained index artifact (see train.py) against the held-out
eval split and writes the metrics as JSON - the input to gate.py's
regression check.

Same evaluation as hs-code-classifier's eval_classification.py, in the
configuration that project actually ships (reference titles +
nomenclature, cross-encoder re-ranking): top-1/3/5 exact-code accuracy,
MRR, and heading/chapter-level top-1 accuracy, over a fixed 500-query
subsample of the held-out titles.

Fully deterministic: the split, the subsample, and CPU inference are all
seeded or seed-free, so running this twice on the same commit gives the
same numbers. That matters for a gate - any metric movement a PR
causes is caused by the PR, not by a different random draw.

Run it directly (after python -m model.train):

    python -m model.evaluate --out metrics/latest.json
"""

import argparse
import json
import random
from pathlib import Path

from model.classify import SEARCH_POOL, _dedupe_by_code
from model.embeddings import embed_texts
from model.index import ReferenceIndex
from model.references import DEFAULT_SEED, Reference, load_references, split_train_eval
from model.rerank import rerank as rerank_candidates
from model.train import ARTIFACT_DIR, load_artifact

MAX_EVAL_QUERIES = 500
MAX_K = 5


def eval_queries(max_queries: int = MAX_EVAL_QUERIES) -> list[Reference]:
    _index_refs, eval_refs = split_train_eval(load_references(), seed=DEFAULT_SEED)
    if len(eval_refs) > max_queries:
        random.Random(DEFAULT_SEED).shuffle(eval_refs)
        eval_refs = eval_refs[:max_queries]
    return eval_refs


def check_no_leakage(index: ReferenceIndex, eval_refs: list[Reference]) -> None:
    """
    Fail loudly if any eval query appears verbatim (same title, same code)
    in the index. That would mean the artifact was built from the wrong
    side of the split, and every metric below would be inflated.
    """
    indexed = {(r.title, r.code) for r in index.references}
    leaked = [r for r in eval_refs if (r.title, r.code) in indexed]
    if leaked:
        raise ValueError(f"{len(leaked)} eval queries are present in the index, e.g. {leaked[0]}")


def rank_of_correct_code(candidates: list[tuple[Reference, float]], expected_code: str) -> int | None:
    for rank, (ref, _score) in enumerate(candidates, start=1):
        if ref.code == expected_code:
            return rank
    return None


def compute_metrics(
    ranks: list[int | None], expected: list[Reference], top1: list[Reference | None]
) -> dict[str, float]:
    """
    ranks[i] is the 1-based rank of the correct code for query i within
    the top MAX_K distinct candidates, or None if it wasn't there at all.
    top1[i] is the top candidate's Reference (None if nothing came back).
    """
    n = len(ranks)
    if n == 0:
        raise ValueError("no eval queries to compute metrics over")

    def hit_rate(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    return {
        "top1": hit_rate(1),
        "top3": hit_rate(3),
        "top5": hit_rate(5),
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
        "heading_top1": sum(1 for e, t in zip(expected, top1) if t is not None and t.heading == e.heading) / n,
        "chapter_top1": sum(1 for e, t in zip(expected, top1) if t is not None and t.chapter == e.chapter) / n,
    }


def evaluate(index: ReferenceIndex, eval_refs: list[Reference], use_rerank: bool = True) -> dict[str, float]:
    queries = [ref.title for ref in eval_refs]
    query_vectors = embed_texts(queries)

    ranks: list[int | None] = []
    top1: list[Reference | None] = []
    for expected, query, query_vector in zip(eval_refs, queries, query_vectors):
        candidates = index.search(query_vector, top_k=SEARCH_POOL)
        if use_rerank:
            candidates = rerank_candidates(query, candidates, top_k=SEARCH_POOL)
        deduped = _dedupe_by_code(candidates)[:MAX_K]

        ranks.append(rank_of_correct_code(deduped, expected.code))
        top1.append(deduped[0][0] if deduped else None)

    return compute_metrics(ranks, eval_refs, top1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--out", type=Path, default=Path("metrics/latest.json"))
    parser.add_argument("--max-queries", type=int, default=MAX_EVAL_QUERIES)
    args = parser.parse_args()

    index, manifest = load_artifact(args.artifact)
    eval_refs = eval_queries(args.max_queries)
    check_no_leakage(index, eval_refs)

    metrics = evaluate(index, eval_refs)
    result = {
        # Rounded so the committed baseline diffs cleanly and last-bit
        # float noise between machines can't masquerade as a change.
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "n_queries": len(eval_refs),
        "model": {k: manifest.get(k) for k in ("embedding_model", "rerank_model", "data_sha256", "git_sha")},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
