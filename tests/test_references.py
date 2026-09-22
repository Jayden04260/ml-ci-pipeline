import json

from model.references import Reference, load_references, split_train_eval


def test_reference_chapter_and_heading_properties():
    ref = Reference(title="3d printer", code="8485000000", level="Heading")
    assert ref.chapter == "84"
    assert ref.heading == "8485"


def test_load_references_reads_json(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text(
        json.dumps([{"title": "abalone", "code": "0307000000", "level": "Heading"}]), encoding="utf-8"
    )
    refs = load_references(path)
    assert refs == [Reference(title="abalone", code="0307000000", level="Heading")]


def test_split_train_eval_holds_out_one_synonym_per_multi_synonym_code():
    refs = [
        Reference(title="abalone", code="0307000000", level="Heading"),
        Reference(title="abalone, prepared or preserved", code="1605570000", level="Commodity"),
        Reference(title="abalone (fresh)", code="0307000000", level="Heading"),
    ]
    index_refs, eval_refs = split_train_eval(refs, seed=1)

    # code 0307000000 has 2 synonyms -> one held out for eval, one stays in the index
    eval_codes = [r.code for r in eval_refs]
    assert eval_codes == ["0307000000"]

    # the single-synonym code (1605570000) is never eligible for eval - it always stays in the index
    index_codes = [r.code for r in index_refs]
    assert "1605570000" in index_codes
    assert index_codes.count("0307000000") == 1


def test_split_train_eval_single_synonym_codes_never_held_out():
    refs = [Reference(title="abaci (abacus)", code="9503000000", level="Heading")]
    index_refs, eval_refs = split_train_eval(refs, seed=1)
    assert index_refs == refs
    assert eval_refs == []


def test_split_train_eval_covers_every_input_reference_exactly_once():
    refs = [
        Reference(title=f"item {i}", code=f"{i:010d}", level="Heading") for i in range(5)
    ] + [
        Reference(title="dup a", code="9999999999", level="Heading"),
        Reference(title="dup b", code="9999999999", level="Heading"),
    ]
    index_refs, eval_refs = split_train_eval(refs, seed=1)
    assert len(index_refs) + len(eval_refs) == len(refs)
    assert set(r.title for r in index_refs) | set(r.title for r in eval_refs) == {r.title for r in refs}


def test_split_train_eval_is_deterministic_for_a_given_seed():
    refs = [
        Reference(title="a1", code="1111111111", level="Heading"),
        Reference(title="a2", code="1111111111", level="Heading"),
        Reference(title="b1", code="2222222222", level="Heading"),
        Reference(title="b2", code="2222222222", level="Heading"),
    ]
    result1 = split_train_eval(refs, seed=7)
    result2 = split_train_eval(refs, seed=7)
    assert result1 == result2
