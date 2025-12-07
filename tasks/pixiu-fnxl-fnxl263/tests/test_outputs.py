import json
from pathlib import Path

EXPECTED_LABEL = "(:O\n1)The:O\n$:O\n15.8:B-GoodwillAcquiredDuringPeriod\nmillion:O\naddition:O\nto:O\ngoodwill:O\nin:O\n2020:O\nwas:O\ndue:O\nto:O\nthe:O\npreliminary:O\nallocation:O\nof:O\nthe:O\npurchase:O\nprice:O\nto:O\nacquire:O\nPartsmaster:O\n.:O"
ALLOWED_CHOICES = ["(:O\n1)The:O\n$:O\n15.8:B-GoodwillAcquiredDuringPeriod\nmillion:O\naddition:O\nto:O\ngoodwill:O\nin:O\n2020:O\nwas:O\ndue:O\nto:O\nthe:O\npreliminary:O\nallocation:O\nof:O\nthe:O\npurchase:O\nprice:O\nto:O\nacquire:O\nPartsmaster:O\n.:O"]
PIXIU_ID = "fnxl263"
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


