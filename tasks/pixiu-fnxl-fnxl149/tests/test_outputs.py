import json
from pathlib import Path

EXPECTED_LABEL = "Xilinx:O\nwill:O\nmatch:O\nup:O\nto:O\n50:O\n%:O\nof:O\nthe:O\nfirst:O\n8:B-DefinedContributionPlanEmployerMatchingContributionPercent\n%:O\nof:O\nan:O\nemployee:O\n's:O\ncompensation:O\nthat:O\nthe:O\nemployee:O\ncontributed:O\nto:O\ntheir:O\n401(k:O\n):O\naccounts:O\n.:O"
ALLOWED_CHOICES = ["Xilinx:O\nwill:O\nmatch:O\nup:O\nto:O\n50:O\n%:O\nof:O\nthe:O\nfirst:O\n8:B-DefinedContributionPlanEmployerMatchingContributionPercent\n%:O\nof:O\nan:O\nemployee:O\n's:O\ncompensation:O\nthat:O\nthe:O\nemployee:O\ncontributed:O\nto:O\ntheir:O\n401(k:O\n):O\naccounts:O\n.:O"]
PIXIU_ID = "fnxl149"
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


