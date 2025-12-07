import json
from pathlib import Path

EXPECTED_LABEL = "In:O\naddition:O\n,:O\nwe:O\nexpect:O\nto:O\nderecognize:O\na:O\nbuild:O\n-:O\nto:O\n-:O\nsuit:O\narrangement:O\nin:O\naccordance:O\nwith:O\nthe:O\ntransition:O\nrequirements:O\n,:O\nwhich:O\nwill:O\nresult:O\nin:O\nan:O\nadjustment:O\nto:O\nretained:O\nearnings:O\nof:O\napproximately:O\n$:O\n7,000:B-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption\n.:O"
ALLOWED_CHOICES = ["In:O\naddition:O\n,:O\nwe:O\nexpect:O\nto:O\nderecognize:O\na:O\nbuild:O\n-:O\nto:O\n-:O\nsuit:O\narrangement:O\nin:O\naccordance:O\nwith:O\nthe:O\ntransition:O\nrequirements:O\n,:O\nwhich:O\nwill:O\nresult:O\nin:O\nan:O\nadjustment:O\nto:O\nretained:O\nearnings:O\nof:O\napproximately:O\n$:O\n7,000:B-CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption\n.:O"]
PIXIU_ID = "fnxl222"
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


