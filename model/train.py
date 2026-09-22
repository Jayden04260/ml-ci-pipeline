"""
train.py

"Training" for a retrieval model: embed the index corpus and save it as
a self-contained artifact that evaluate.py scores and the deploy step
ships. There are no learned weights here beyond the pretrained embedding
and cross-encoder models - what changes between versions is the corpus,
the embedding model, and the retrieval/rerank logic, and those are
exactly what this pipeline needs to catch regressions in.

The index is built from the train side of references.split_train_eval
plus the official nomenclature text - never from the held-out eval
titles, so evaluate.py can't be scoring queries the index already
contains verbatim.

Output (ARTIFACT_DIR, default artifacts/):
    embeddings.npy   (n, dim) float32, L2-normalized
    references.json  the n Reference rows, same order as embeddings
    manifest.json    which models/data produced this artifact

Run it directly:

    python -m model.train
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from model import embeddings, rerank
from model.index import ReferenceIndex
from model.references import (
    DEFAULT_DATA_PATH,
    DEFAULT_NOMENCLATURE_PATH,
    DEFAULT_SEED,
    Reference,
    load_references,
    split_train_eval,
)

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"


def data_fingerprint(paths: list[Path]) -> str:
    """sha256 over the input data files, so a metrics change can be traced
    to a data change rather than a code change (or ruled out as one)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_index_references(
    data_path: Path = DEFAULT_DATA_PATH, nomenclature_path: Path = DEFAULT_NOMENCLATURE_PATH
) -> list[Reference]:
    index_refs, _eval_refs = split_train_eval(load_references(data_path), seed=DEFAULT_SEED)
    return index_refs + load_references(nomenclature_path)


def save_artifact(out_dir: Path, vectors: np.ndarray, references: list[Reference], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vectors)
    (out_dir / "references.json").write_text(json.dumps([asdict(r) for r in references]), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_artifact(artifact_dir: Path = ARTIFACT_DIR) -> tuple[ReferenceIndex, dict]:
    vectors = np.load(artifact_dir / "embeddings.npy")
    raw = json.loads((artifact_dir / "references.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    references = [Reference(**r) for r in raw]
    # ReferenceIndex itself rejects a row-count mismatch, which is the
    # failure a half-written or hand-edited artifact would produce.
    return ReferenceIndex(vectors, references), manifest


def train(out_dir: Path = ARTIFACT_DIR) -> dict:
    references = build_index_references()
    vectors = embeddings.embed_texts([r.title for r in references])

    manifest = {
        "embedding_model": embeddings.DEFAULT_MODEL_NAME,
        "rerank_model": rerank.DEFAULT_MODEL_NAME,
        "n_references": len(references),
        "embedding_dim": int(vectors.shape[1]),
        "data_sha256": data_fingerprint([DEFAULT_DATA_PATH, DEFAULT_NOMENCLATURE_PATH]),
        "split_seed": DEFAULT_SEED,
        # Set by GitHub Actions; absent for local runs.
        "git_sha": os.environ.get("GITHUB_SHA"),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_artifact(out_dir, vectors, references, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()

    manifest = train(args.out)
    print(f"Built index artifact: {manifest['n_references']} references -> {args.out}")


if __name__ == "__main__":
    main()
