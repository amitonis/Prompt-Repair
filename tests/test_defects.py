from src.defects import changed_sections, inject_defect, load_defects, load_prompt


def test_oracle_prompt_has_four_sections():
    oracle = load_prompt("oracle")
    assert set(oracle) == {"instruction", "context", "constraints", "output_format"}
    assert all(isinstance(text, str) and text.strip() for text in oracle.values())


def test_defect_catalog_loads_eight_defects():
    defects = load_defects()
    assert [d["id"] for d in defects] == [f"D{i}" for i in range(1, 9)]


def test_injector_changes_exactly_one_section():
    oracle = load_prompt("oracle")
    for defect in load_defects():
        defective = inject_defect(oracle, defect)
        assert changed_sections(oracle, defective) == [defect["section"]]
        for section, text in oracle.items():
            if section != defect["section"]:
                assert defective[section] == text
        assert defective[defect["section"]] == defect["defective_section_text"]


def test_injected_text_differs_from_oracle_section():
    oracle = load_prompt("oracle")
    for defect in load_defects():
        assert defect["defective_section_text"] != oracle[defect["section"]]


def test_injector_rejects_noop_defect():
    oracle = load_prompt("oracle")
    noop = {
        "id": "D0",
        "section": "constraints",
        "defective_section_text": oracle["constraints"],
    }
    try:
        inject_defect(oracle, noop)
    except ValueError:
        pass
    else:
        raise AssertionError("A defect that changes nothing must be rejected")
