import json
from pathlib import Path

EXPECTED_LABEL = "In:O\naddition:O\n,:O\nif:O\n(:O\nx)the:O\nCompany:O\nissues:O\nadditional:O\nClassA:O\nordinary:O\nshares:O\nor:O\nequity:O\n-:O\nlinked:O\nsecurities:O\nfor:O\ncapital:O\nraising:O\npurposes:O\nin:O\nconnection:O\nwith:O\nthe:O\nclosing:O\nof:O\na:O\nBusiness:O\nCombination:O\nat:O\nan:O\nissue:O\nprice:O\nor:O\neffective:O\nissue:O\nprice:O\nof:O\nless:O\nthan:O\n$:O\n9.20:B-SaleOfStockPricePerShare\nper:O\nClassA:O\nordinary:O\nshare:O\n(:O\n\u201c:O\nNewly:O\nIssued:O\nPrice:O\n\u201d:O\n):O\n(:O\nwith:O\nsuch:O\nissue:O\nprice:O\nor:O\neffective:O\nissue:O\nprice:O\nto:O\nbe:O\ndetermined:O\nin:O\ngood:O\nfaith:O\nby:O\nthe:O\nCompany:O\n\u2019s:O\nboard:O\nof:O\ndirectors:O\nand:O\n,:O\nin:O\nthe:O\ncase:O\nof:O\nany:O\nsuch:O\nissuance:O\nto:O\nthe:O\nSponsor:O\nor:O\nits:O\naffiliates:O\n,:O\nwithout:O\ntaking:O\ninto:O\naccount:O\nany:O\nFounder:O\nShares:O\nheld:O\nby:O\nthe:O\nSponsor:O\n,:O\nAnchor:O\nInvestor:O\nor:O\nsuch:O\naffiliates:O\n,:O\nas:O\napplicable:O\n,:O\nprior:O\nto:O\nsuch:O\nissuance:O\n):O\n,:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\n(:O\ny)the:O\naggregate:O\ngross:O\nproceeds:O\nfrom:O\nsuch:O\nissuances:O\nrepresent:O\nmore:O\nthan:O\n60:O\n%:O\nof:O\nthe:O\ntotal:O\nequity:O\nproceeds:O\n,:O\nand:O\ninterest:O\nthereon:O\n,:O\navailable:O\nfor:O\nthe:O\nfunding:O\nof:O\na:O\nBusiness:O\nCombination:O\non:O\nthe:O\ndate:O\nof:O\nthe:O\nconsummation:O\nof:O\na:O\nBusiness:O\nCombination:O\n(:O\nnet:O\nof:O\nredemptions:O\n):O\n,:O\nand:O\n(:O\nz)the:O\nvolume:O\nweighted:O\naverage:O\ntrading:O\nprice:O\nof:O\nits:O\nClassA:O\nordinary:O\nshares:O\nduring:O\nthe:O\n20:O\ntrading:O\nday:O\nperiod:O\nstarting:O\non:O\nthe:O\ntrading:O\nday:O\nprior:O\nto:O\nthe:O\nday:O\non:O\nwhich:O\nthe:O\nCompany:O\nconsummates:O\nits:O\nBusiness:O\nCombination:O\n(:O\nsuch:O\nprice:O\n,:O\nthe:O\n\u201c:O\nMarket:O\nValue:O\n\u201d:O\n):O\n,:O\nis:O\nbelow:O\n$:O\n9.20:O\nper:O\nshare:O\n,:O\nthe:O\nexercise:O\nprice:O\nof:O\nthe:O\nwarrants:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\n115:O\n%:O\nof:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\nthe:O\n$:O\n18.00:O\nper:O\nshare:O\nredemption:O\ntrigger:O\nprice:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\n180:O\n%:O\nof:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\nand:O\nthe:O\n$:O\n10.00:O\nper:O\nshare:O\nredemption:O\ntrigger:O\nprice:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n.:O"
ALLOWED_CHOICES = ["In:O\naddition:O\n,:O\nif:O\n(:O\nx)the:O\nCompany:O\nissues:O\nadditional:O\nClassA:O\nordinary:O\nshares:O\nor:O\nequity:O\n-:O\nlinked:O\nsecurities:O\nfor:O\ncapital:O\nraising:O\npurposes:O\nin:O\nconnection:O\nwith:O\nthe:O\nclosing:O\nof:O\na:O\nBusiness:O\nCombination:O\nat:O\nan:O\nissue:O\nprice:O\nor:O\neffective:O\nissue:O\nprice:O\nof:O\nless:O\nthan:O\n$:O\n9.20:B-SaleOfStockPricePerShare\nper:O\nClassA:O\nordinary:O\nshare:O\n(:O\n“:O\nNewly:O\nIssued:O\nPrice:O\n”:O\n):O\n(:O\nwith:O\nsuch:O\nissue:O\nprice:O\nor:O\neffective:O\nissue:O\nprice:O\nto:O\nbe:O\ndetermined:O\nin:O\ngood:O\nfaith:O\nby:O\nthe:O\nCompany:O\n’s:O\nboard:O\nof:O\ndirectors:O\nand:O\n,:O\nin:O\nthe:O\ncase:O\nof:O\nany:O\nsuch:O\nissuance:O\nto:O\nthe:O\nSponsor:O\nor:O\nits:O\naffiliates:O\n,:O\nwithout:O\ntaking:O\ninto:O\naccount:O\nany:O\nFounder:O\nShares:O\nheld:O\nby:O\nthe:O\nSponsor:O\n,:O\nAnchor:O\nInvestor:O\nor:O\nsuch:O\naffiliates:O\n,:O\nas:O\napplicable:O\n,:O\nprior:O\nto:O\nsuch:O\nissuance:O\n):O\n,:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\n(:O\ny)the:O\naggregate:O\ngross:O\nproceeds:O\nfrom:O\nsuch:O\nissuances:O\nrepresent:O\nmore:O\nthan:O\n60:O\n%:O\nof:O\nthe:O\ntotal:O\nequity:O\nproceeds:O\n,:O\nand:O\ninterest:O\nthereon:O\n,:O\navailable:O\nfor:O\nthe:O\nfunding:O\nof:O\na:O\nBusiness:O\nCombination:O\non:O\nthe:O\ndate:O\nof:O\nthe:O\nconsummation:O\nof:O\na:O\nBusiness:O\nCombination:O\n(:O\nnet:O\nof:O\nredemptions:O\n):O\n,:O\nand:O\n(:O\nz)the:O\nvolume:O\nweighted:O\naverage:O\ntrading:O\nprice:O\nof:O\nits:O\nClassA:O\nordinary:O\nshares:O\nduring:O\nthe:O\n20:O\ntrading:O\nday:O\nperiod:O\nstarting:O\non:O\nthe:O\ntrading:O\nday:O\nprior:O\nto:O\nthe:O\nday:O\non:O\nwhich:O\nthe:O\nCompany:O\nconsummates:O\nits:O\nBusiness:O\nCombination:O\n(:O\nsuch:O\nprice:O\n,:O\nthe:O\n“:O\nMarket:O\nValue:O\n”:O\n):O\n,:O\nis:O\nbelow:O\n$:O\n9.20:O\nper:O\nshare:O\n,:O\nthe:O\nexercise:O\nprice:O\nof:O\nthe:O\nwarrants:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\n115:O\n%:O\nof:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\nthe:O\n$:O\n18.00:O\nper:O\nshare:O\nredemption:O\ntrigger:O\nprice:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\n180:O\n%:O\nof:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n,:O\nand:O\nthe:O\n$:O\n10.00:O\nper:O\nshare:O\nredemption:O\ntrigger:O\nprice:O\nwill:O\nbe:O\nadjusted:O\n(:O\nto:O\nthe:O\nnearest:O\ncent:O\n):O\nto:O\nbe:O\nequal:O\nto:O\nthe:O\nhigher:O\nof:O\nthe:O\nMarket:O\nValue:O\nand:O\nthe:O\nNewly:O\nIssued:O\nPrice:O\n.:O"]
PIXIU_ID = "fnxl218"
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


