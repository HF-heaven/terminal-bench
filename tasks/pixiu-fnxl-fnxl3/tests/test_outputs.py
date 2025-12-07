import json
from pathlib import Path

EXPECTED_LABEL = "Equity:O\nin:O\nearnings:O\nof:O\ncertain:O\nof:O\nour:O\njoint:O\nventures:O\nincludes:O\nthe:O\namortization:O\nof:O\nthe:O\nCompany:O\n\u2019s:O\nexcess:O\npurchase:O\nprice:O\nof:O\n$:O\n25,251:B-EquityMethodInvestmentDifferenceBetweenCarryingAmountAndUnderlyingEquity\nof:O\nthese:O\nequity:O\ninvestments:O\nover:O\nits:O\noriginal:O\nbasis:O\n.:O"
ALLOWED_CHOICES = ["Equity:O\nin:O\nearnings:O\nof:O\ncertain:O\nof:O\nour:O\njoint:O\nventures:O\nincludes:O\nthe:O\namortization:O\nof:O\nthe:O\nCompany:O\n’s:O\nexcess:O\npurchase:O\nprice:O\nof:O\n$:O\n25,251:B-EquityMethodInvestmentDifferenceBetweenCarryingAmountAndUnderlyingEquity\nof:O\nthese:O\nequity:O\ninvestments:O\nover:O\nits:O\noriginal:O\nbasis:O\n.:O"]
PIXIU_ID = "fnxl3"
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


