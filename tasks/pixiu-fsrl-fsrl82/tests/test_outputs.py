import json
from pathlib import Path

EXPECTED_LABEL = "LONDON:B-QUANT\nINTERBANK:I-QUANT\nOFFERED:I-QUANT\nRATES:I-QUANT\n-LRB-:O\nLIBOR:B-QUANT\n-RRB-:O\n::O\n8:B-VALUE\n11\\/16:I-VALUE\n%:I-VALUE\none:B-TIME\nmonth:I-TIME\n;:O\n8:B-VALUE\n11\\/16:I-VALUE\n%:I-VALUE\nthree:B-TIME\nmonths:I-TIME\n;:O\n8:B-VALUE\n1\\/2:I-VALUE\n%:I-VALUE\nsix:B-TIME\nmonths:I-TIME\n;:O\n8:B-VALUE\n1\\/2:I-VALUE\n%:I-VALUE\none:B-TIME\nyear:I-TIME\n.:O"
ALLOWED_CHOICES = ["LONDON:B-QUANT\nINTERBANK:I-QUANT\nOFFERED:I-QUANT\nRATES:I-QUANT\n-LRB-:O\nLIBOR:B-QUANT\n-RRB-:O\n::O\n8:B-VALUE\n11\\/16:I-VALUE\n%:I-VALUE\none:B-TIME\nmonth:I-TIME\n;:O\n8:B-VALUE\n11\\/16:I-VALUE\n%:I-VALUE\nthree:B-TIME\nmonths:I-TIME\n;:O\n8:B-VALUE\n1\\/2:I-VALUE\n%:I-VALUE\nsix:B-TIME\nmonths:I-TIME\n;:O\n8:B-VALUE\n1\\/2:I-VALUE\n%:I-VALUE\none:B-TIME\nyear:I-TIME\n.:O"]
PIXIU_ID = "fsrl82"
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


