import json
from pathlib import Path

EXPECTED_LABEL = "Following:O\na:O\ncourt:O\n-:O\nordered:O\nmediation:O\non:O\nMarch:O\n28:O\n,:O\n2019:O\n,:O\nwe:O\nreached:O\nan:O\nagreement:O\nin:O\nprinciple:O\nto:O\nsettle:O\nthis:O\nmatter:O\nfor:O\na:O\ntotal:O\nof:O\n$:O\n13:B-LitigationSettlementAmountAwardedToOtherParty\nmillion:O\n,:O\nincluding:O\nattorneys:O\n\u2019:O\nfees:O\n.:O"
ALLOWED_CHOICES = ["Following:O\na:O\ncourt:O\n-:O\nordered:O\nmediation:O\non:O\nMarch:O\n28:O\n,:O\n2019:O\n,:O\nwe:O\nreached:O\nan:O\nagreement:O\nin:O\nprinciple:O\nto:O\nsettle:O\nthis:O\nmatter:O\nfor:O\na:O\ntotal:O\nof:O\n$:O\n13:B-LitigationSettlementAmountAwardedToOtherParty\nmillion:O\n,:O\nincluding:O\nattorneys:O\n’:O\nfees:O\n.:O"]
PIXIU_ID = "fnxl298"
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


