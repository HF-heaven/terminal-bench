import json
from pathlib import Path

EXPECTED_LABEL = "Additionally:O\n,:O\nthe:O\nCompany:O\nexpensed:O\n$:O\n8.6:B-InterestExpense\nmillion:O\nof:O\nthe:O\npremium:O\npaid:O\nfor:O\nthe:O\nderivatives:O\nas:O\na:O\ncomponent:O\nof:O\ninterest:O\nexpense:O\nfor:O\nthe:O\nyear:O\nended:O\nDecember:O\n31:O\n,:O\n2018:O\n.:O"
ALLOWED_CHOICES = ["Additionally:O\n,:O\nthe:O\nCompany:O\nexpensed:O\n$:O\n8.6:B-InterestExpense\nmillion:O\nof:O\nthe:O\npremium:O\npaid:O\nfor:O\nthe:O\nderivatives:O\nas:O\na:O\ncomponent:O\nof:O\ninterest:O\nexpense:O\nfor:O\nthe:O\nyear:O\nended:O\nDecember:O\n31:O\n,:O\n2018:O\n.:O"]
PIXIU_ID = "fnxl282"
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


