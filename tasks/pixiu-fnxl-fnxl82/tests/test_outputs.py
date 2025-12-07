import json
from pathlib import Path

EXPECTED_LABEL = "Undrawn:O\nbalances:O\navailable:O\nunder:O\nthe:O\nrevolving:O\ncredit:O\nfacility:O\nare:O\nsubject:O\nto:O\ncommitment:O\nfees:O\nat:O\nthe:O\napplicable:O\nrate:O\nranging:O\nfrom:O\n0.07:B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage\n%:O\nto:O\n0.20:B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage\n%:O\n.:O"
ALLOWED_CHOICES = ["Undrawn:O\nbalances:O\navailable:O\nunder:O\nthe:O\nrevolving:O\ncredit:O\nfacility:O\nare:O\nsubject:O\nto:O\ncommitment:O\nfees:O\nat:O\nthe:O\napplicable:O\nrate:O\nranging:O\nfrom:O\n0.07:B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage\n%:O\nto:O\n0.20:B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage\n%:O\n.:O"]
PIXIU_ID = "fnxl82"
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


