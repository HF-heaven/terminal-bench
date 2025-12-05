import json
from pathlib import Path

EXPECTED_LABEL = "Total:O\nCompany:O\ninterest:O\npayments:O\nwere:O\n$:O\n616:B-InterestPaid\n,:O\n$:O\n527:B-InterestPaid\nand:O\n$:O\n523:B-InterestPaid\nfor:O\nthe:O\nyears:O\nended:O\nDecember:O\n31:O\n,:O\n2018:O\n,:O\n2017:O\nand:O\n2016:O\n,:O\nrespectively:O\n.:O"
ALLOWED_CHOICES = ["Total:O\nCompany:O\ninterest:O\npayments:O\nwere:O\n$:O\n616:B-InterestPaid\n,:O\n$:O\n527:B-InterestPaid\nand:O\n$:O\n523:B-InterestPaid\nfor:O\nthe:O\nyears:O\nended:O\nDecember:O\n31:O\n,:O\n2018:O\n,:O\n2017:O\nand:O\n2016:O\n,:O\nrespectively:O\n.:O"]
PIXIU_ID = "fnxl271"
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


