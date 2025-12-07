import json
from pathlib import Path

EXPECTED_LABEL = "This:O\namount:O\nincludes:O\n$:O\n8:B-DerivativeGainLossOnDerivativeNet\nmillion:O\nof:O\nrealized:O\nlosses:O\nrelated:O\nto:O\ncrop:O\nderivatives:O\nwhich:O\nare:O\nreported:O\nin:O\nNet:O\nrealized:O\ngains:O\n(:O\nlosses:O\n):O\nincluding:O\nOTTI:O\nin:O\nthe:O\nCorporate:O\ncolumn:O\nbelow:O\n.:O"
ALLOWED_CHOICES = ["This:O\namount:O\nincludes:O\n$:O\n8:B-DerivativeGainLossOnDerivativeNet\nmillion:O\nof:O\nrealized:O\nlosses:O\nrelated:O\nto:O\ncrop:O\nderivatives:O\nwhich:O\nare:O\nreported:O\nin:O\nNet:O\nrealized:O\ngains:O\n(:O\nlosses:O\n):O\nincluding:O\nOTTI:O\nin:O\nthe:O\nCorporate:O\ncolumn:O\nbelow:O\n.:O"]
PIXIU_ID = "fnxl42"
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


