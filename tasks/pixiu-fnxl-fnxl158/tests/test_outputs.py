import json
from pathlib import Path

EXPECTED_LABEL = "At:O\nyear:O\n-:O\nend:O\n2020:O\n,:O\nwe:O\nhad:O\napproximately:O\n$:O\n3,938:O\nmillion:O\nof:O\nprimarily:O\nstate:O\nand:O\nforeign:O\nnet:O\noperating:O\nlosses:O\n,:O\nof:O\nwhich:O\n$:O\n2,315:B-DeferredTaxAssetsOperatingLossCarryforwardsSubjectToExpiration\nmillion:O\nwill:O\nexpire:O\nthrough:O\n2040:O\n.:O"
ALLOWED_CHOICES = ["At:O\nyear:O\n-:O\nend:O\n2020:O\n,:O\nwe:O\nhad:O\napproximately:O\n$:O\n3,938:O\nmillion:O\nof:O\nprimarily:O\nstate:O\nand:O\nforeign:O\nnet:O\noperating:O\nlosses:O\n,:O\nof:O\nwhich:O\n$:O\n2,315:B-DeferredTaxAssetsOperatingLossCarryforwardsSubjectToExpiration\nmillion:O\nwill:O\nexpire:O\nthrough:O\n2040:O\n.:O"]
PIXIU_ID = "fnxl158"
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


