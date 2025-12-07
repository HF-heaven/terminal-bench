import json
from pathlib import Path

EXPECTED_LABEL = "The:O\nSubsidiary:O\nIssuer:O\nis:O\na:O\nLuxembourg:O\npublic:O\nlimited:O\nliability:O\ncompany:O\nformed:O\nin:O\nJanuary:O\n2012:O\nand:O\n100:B-VariableInterestEntityOwnershipPercentage\npercent:O\n-:O\nowned:O\nsubsidiary:O\nof:O\nthe:O\nSubsidiary:O\nGuarantor:O\n.:O"
ALLOWED_CHOICES = ["The:O\nSubsidiary:O\nIssuer:O\nis:O\na:O\nLuxembourg:O\npublic:O\nlimited:O\nliability:O\ncompany:O\nformed:O\nin:O\nJanuary:O\n2012:O\nand:O\n100:B-VariableInterestEntityOwnershipPercentage\npercent:O\n-:O\nowned:O\nsubsidiary:O\nof:O\nthe:O\nSubsidiary:O\nGuarantor:O\n.:O"]
PIXIU_ID = "fnxl195"
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


