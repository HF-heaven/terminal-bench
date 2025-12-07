import json
from pathlib import Path

EXPECTED_LABEL = "At:O\nDecember31,2020:O\n,:O\nthe:O\nfair:O\nvalue:O\nof:O\nthe:O\nCorporation:O\n\u2019s:O\ninvestment:O\nin:O\nsmall:O\nbusiness:O\ninvestment:O\ncompanies:O\n,:O\nbased:O\non:O\nnet:O\nasset:O\nvalue:O\n,:O\nwas:O\n$:O\n1.48:B-EquitySecuritiesFvNi\nmillion:O\n.:O"
ALLOWED_CHOICES = ["At:O\nDecember31,2020:O\n,:O\nthe:O\nfair:O\nvalue:O\nof:O\nthe:O\nCorporation:O\n’s:O\ninvestment:O\nin:O\nsmall:O\nbusiness:O\ninvestment:O\ncompanies:O\n,:O\nbased:O\non:O\nnet:O\nasset:O\nvalue:O\n,:O\nwas:O\n$:O\n1.48:B-EquitySecuritiesFvNi\nmillion:O\n.:O"]
PIXIU_ID = "fnxl219"
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


