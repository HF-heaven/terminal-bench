import json
from pathlib import Path

EXPECTED_LABEL = "Our:O\nindefinite:O\n-:O\nlived:O\nintangible:O\nasset:O\nbalance:O\nprimarily:O\nconsists:O\nof:O\na:O\nnumber:O\nof:O\nindividual:O\nbrands:O\n,:O\nwhich:O\nhad:O\nan:O\naggregate:O\ncarrying:O\namount:O\nof:O\n$:O\n43.4:B-IndefiniteLivedIntangibleAssetsExcludingGoodwill\nbillion:O\nas:O\nof:O\nDecember28:O\n,:O\n2019:O\n.:O"
ALLOWED_CHOICES = ["Our:O\nindefinite:O\n-:O\nlived:O\nintangible:O\nasset:O\nbalance:O\nprimarily:O\nconsists:O\nof:O\na:O\nnumber:O\nof:O\nindividual:O\nbrands:O\n,:O\nwhich:O\nhad:O\nan:O\naggregate:O\ncarrying:O\namount:O\nof:O\n$:O\n43.4:B-IndefiniteLivedIntangibleAssetsExcludingGoodwill\nbillion:O\nas:O\nof:O\nDecember28:O\n,:O\n2019:O\n.:O"]
PIXIU_ID = "fnxl177"
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


