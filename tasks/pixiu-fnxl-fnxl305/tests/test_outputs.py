import json
from pathlib import Path

EXPECTED_LABEL = "Other:O\nEquity:O\n\u2014:O\nXcel:O\nEnergy:O\nissued:O\n$:O\n40million:O\nand:O\n$:O\n39:B-ProceedsFromIssuanceOfCommonStock\nmillion:O\nof:O\nequity:O\nannually:O\nthrough:O\nthe:O\nDRIP:O\nprogram:O\nduring:O\nthe:O\nyears:O\nended:O\nDec.:O\n31:O\n,:O\n2020:O\nand:O\n2019:O\nrespectively:O\n.:O"
ALLOWED_CHOICES = ["Other:O\nEquity:O\n—:O\nXcel:O\nEnergy:O\nissued:O\n$:O\n40million:O\nand:O\n$:O\n39:B-ProceedsFromIssuanceOfCommonStock\nmillion:O\nof:O\nequity:O\nannually:O\nthrough:O\nthe:O\nDRIP:O\nprogram:O\nduring:O\nthe:O\nyears:O\nended:O\nDec.:O\n31:O\n,:O\n2020:O\nand:O\n2019:O\nrespectively:O\n.:O"]
PIXIU_ID = "fnxl305"
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


