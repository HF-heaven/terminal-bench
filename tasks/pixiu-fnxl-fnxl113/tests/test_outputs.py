import json
from pathlib import Path

EXPECTED_LABEL = "For:O\nthe:O\nyear:O\nended:O\nDecember:O\n31:O\n,:O\n2019:O\n,:O\nrealized:O\nforeign:O\nexchange:O\ngain:O\nwas:O\n$:O\n0.2:B-ForeignCurrencyTransactionGainLossRealized\nmillion:O\nand:O\nwas:O\nrecognized:O\nin:O\nthe:O\nCompany:O\n\u2019s:O\nconsolidated:O\nstatement:O\nof:O\noperations:O\nin:O\ninterest:O\nexpense:O\n,:O\nnet:O\n.:O"
ALLOWED_CHOICES = ["For:O\nthe:O\nyear:O\nended:O\nDecember:O\n31:O\n,:O\n2019:O\n,:O\nrealized:O\nforeign:O\nexchange:O\ngain:O\nwas:O\n$:O\n0.2:B-ForeignCurrencyTransactionGainLossRealized\nmillion:O\nand:O\nwas:O\nrecognized:O\nin:O\nthe:O\nCompany:O\n’s:O\nconsolidated:O\nstatement:O\nof:O\noperations:O\nin:O\ninterest:O\nexpense:O\n,:O\nnet:O\n.:O"]
PIXIU_ID = "fnxl113"
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


