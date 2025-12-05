import json
from pathlib import Path

EXPECTED_LABEL = "Gross:O\nprofit:O\nfor:O\nthe:O\nfirst:O\nquarter:O\nof:O\n2019:O\nincludes:O\n$:O\n11:B-RestructuringAndRelatedCostIncurredCost\nof:O\ncharges:O\nrelated:O\nto:O\nthe:O\nGlobal:O\nGrowth:O\nand:O\nEfficiency:O\nProgram:O\n.:O"
ALLOWED_CHOICES = ["Gross:O\nprofit:O\nfor:O\nthe:O\nfirst:O\nquarter:O\nof:O\n2019:O\nincludes:O\n$:O\n11:B-RestructuringAndRelatedCostIncurredCost\nof:O\ncharges:O\nrelated:O\nto:O\nthe:O\nGlobal:O\nGrowth:O\nand:O\nEfficiency:O\nProgram:O\n.:O"]
PIXIU_ID = "fnxl134"
LABEL_TYPE = "token labels"

ANSWER_FILE = Path("/app/answer.txt")
ITEM_FILE = Path("/tests/data/item.json")


def test_item_file_exists():
    assert ITEM_FILE.exists(), f"Missing data file for PIXIU sample {PIXIU_ID}"
    data = json.loads(ITEM_FILE.read_text())
    assert data.get("id") == PIXIU_ID
    assert data.get("label_type") == LABEL_TYPE


def _read_answer():
    assert ANSWER_FILE.exists(), "answer.txt was not created"
    return ANSWER_FILE.read_text().strip()


def test_answer_is_valid_choice():
    answer = _read_answer()
    assert (
        answer in ALLOWED_CHOICES
    ), f"Answer {answer!r} not in allowed choices {ALLOWED_CHOICES}"


def test_answer_matches_reference():
    answer = _read_answer()
    assert (
        answer == EXPECTED_LABEL
    ), f"Expected {EXPECTED_LABEL!r} but found {answer!r}"


