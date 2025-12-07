import json
from pathlib import Path

EXPECTED_LABEL = "Sales:B-QUANT\nin:O\nthe:B-TIME\nlatest:I-TIME\nperiod:I-TIME\nwere:O\n$:B-VALUE\n1.25:I-VALUE\nbillion:I-VALUE\n,:O\ndown:O\nfrom:O\n$:B-VALUE\n1.36:I-VALUE\nbillion:I-VALUE\nin:O\nthe:B-TIME\n1988:I-TIME\nquarter:I-TIME\n.:O"
ALLOWED_CHOICES = ["Sales:B-QUANT\nin:O\nthe:B-TIME\nlatest:I-TIME\nperiod:I-TIME\nwere:O\n$:B-VALUE\n1.25:I-VALUE\nbillion:I-VALUE\n,:O\ndown:O\nfrom:O\n$:B-VALUE\n1.36:I-VALUE\nbillion:I-VALUE\nin:O\nthe:B-TIME\n1988:I-TIME\nquarter:I-TIME\n.:O"]
PIXIU_ID = "fsrl69"
LABEL_TYPE = "semantic role labels"

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


