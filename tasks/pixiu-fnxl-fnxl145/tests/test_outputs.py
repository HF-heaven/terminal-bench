import json
from pathlib import Path

EXPECTED_LABEL = "The:O\nremaining:O\nnet:O\nbook:O\nvalue:O\nof:O\nthe:O\ntradenames:O\nattributable:O\nto:O\nthe:O\nTransportation:O\n&:O\nIndustrial:O\nsegment:O\nat:O\nDecember31:O\n,:O\n2020:O\nwas:O\napproximately:O\n$:O\n289:B-IndefiniteLivedIntangibleAssetsExcludingGoodwill\nmillion:O\n,:O\nwhich:O\nrepresents:O\nfair:O\nvalue:O\n.:O"
ALLOWED_CHOICES = ["The:O\nremaining:O\nnet:O\nbook:O\nvalue:O\nof:O\nthe:O\ntradenames:O\nattributable:O\nto:O\nthe:O\nTransportation:O\n&:O\nIndustrial:O\nsegment:O\nat:O\nDecember31:O\n,:O\n2020:O\nwas:O\napproximately:O\n$:O\n289:B-IndefiniteLivedIntangibleAssetsExcludingGoodwill\nmillion:O\n,:O\nwhich:O\nrepresents:O\nfair:O\nvalue:O\n.:O"]
PIXIU_ID = "fnxl145"
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


