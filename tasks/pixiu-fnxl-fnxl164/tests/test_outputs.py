import json
from pathlib import Path

EXPECTED_LABEL = "Trade:O\naccounts:O\nreceivable:O\nas:O\nof:O\nDecember:O\n31:O\n,:O\n2018:O\nis:O\nnet:O\nof:O\nan:O\nallowance:O\nfor:O\ndoubtful:O\naccounts:O\nof:O\n$:O\n3:B-AllowanceForDoubtfulAccountsReceivable\nmillion:O\n.:O"
ALLOWED_CHOICES = ["Trade:O\naccounts:O\nreceivable:O\nas:O\nof:O\nDecember:O\n31:O\n,:O\n2018:O\nis:O\nnet:O\nof:O\nan:O\nallowance:O\nfor:O\ndoubtful:O\naccounts:O\nof:O\n$:O\n3:B-AllowanceForDoubtfulAccountsReceivable\nmillion:O\n.:O"]
PIXIU_ID = "fnxl164"
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


