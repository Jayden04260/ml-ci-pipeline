"""
references.py

Loads the local snapshot of UK Trade Tariff reference data (a
snapshot copied from the hs-code-classifier project's data/) and splits it into a retrieval index plus a held-out
eval set, without leaking eval queries into the index.

Splitting logic: many commodity codes have several reference titles
("abalone" and "abalone, prepared or preserved" might both point near
the same code) - synonyms of the same underlying product. Only codes
with more than one synonym are eligible to contribute an eval query,
and exactly one synonym is held out per eligible code; the rest stay in
the index. A code with only one known name can't be tested fairly - if
its one synonym were held out, the correct answer wouldn't even be
retrievable, since nothing in the index would represent it anymore.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "search_references.json"
# Official chapter/heading nomenclature text (hs-code-classifier's fetch_nomenclature.py) -
# same {title, code, level} shape as search_references.json, so
# load_references() reads either file. Only ever added to the retrieval
# index, never eligible as an eval query - these are legal-nomenclature
# descriptions, not the kind of short product-name text a real query
# looks like, so testing against them wouldn't measure the right thing.
DEFAULT_NOMENCLATURE_PATH = Path(__file__).parent.parent / "data" / "nomenclature.json"

# Deterministic - same eval set every run, so accuracy numbers are
# reproducible and comparable across changes to the retrieval/rerank
# logic rather than shifting because of a different random draw.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Reference:
    title: str
    code: str
    level: str

    @property
    def chapter(self) -> str:
        """First 2 digits of the 10-digit code - the broadest classification level."""
        return self.code[:2]

    @property
    def heading(self) -> str:
        """First 4 digits - the level most of this data is actually classified to."""
        return self.code[:4]


def load_references(path: Path = DEFAULT_DATA_PATH) -> list[Reference]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Reference(title=r["title"], code=r["code"], level=r["level"]) for r in raw]


def split_train_eval(
    references: list[Reference], seed: int = DEFAULT_SEED
) -> tuple[list[Reference], list[Reference]]:
    """
    Returns (index_references, eval_references). eval_references holds
    exactly one reference per code that has more than one synonym in
    the input; index_references is everything else, including every
    reference for codes with only one synonym.
    """
    by_code: dict[str, list[Reference]] = {}
    for ref in references:
        by_code.setdefault(ref.code, []).append(ref)

    rng = random.Random(seed)
    index_refs: list[Reference] = []
    eval_refs: list[Reference] = []

    for code, group in by_code.items():
        if len(group) > 1:
            group = list(group)
            rng.shuffle(group)
            eval_refs.append(group[0])
            index_refs.extend(group[1:])
        else:
            index_refs.extend(group)

    return index_refs, eval_refs
