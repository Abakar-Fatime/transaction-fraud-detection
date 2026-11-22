from fraud import notes

ALERT = {"id": 7, "type": "TRANSFER", "amount": 181.0, "nameOrig": "C840083671", "nameDest": "C38997010",
         "step": 1, "score": 0.9876,
         "reasons": [{"text": "the transfer was 100% of the sender's balance"},
                     {"text": "the receiving account was empty before"}]}
FACTS = notes.facts_for(ALERT)


def test_grounded_note_passes():
    n = "A TRANSFER of 181.00 from C840083671 emptied the sender's balance (100%). Score 0.988 - please review."
    assert notes.check_grounded(n, FACTS) == []


def test_invented_number_is_rejected():
    n = "The transfer of 9,500 looks odd, please review."
    assert any("9500" in p for p in notes.check_grounded(n, FACTS))


def test_invented_account_is_rejected():
    assert any("C1234567" in p for p in notes.check_grounded("Account C1234567 is suspicious, review it.", FACTS))


def test_note_without_review_or_too_long_is_rejected():
    assert "does not recommend review" in notes.check_grounded("Looks bad.", FACTS)
    assert "too long" in notes.check_grounded("review " + "word " * 100, FACTS)


def test_generate_retries_a_bad_note_then_accepts_a_good_one():
    answers = iter(["Invented 77777 amount, review.", "Transfer of 181.00 emptied the account, please review."])
    out = notes.generate([ALERT], "m", "h", ask=lambda *a: next(answers))
    assert out["notes"]["7"]["attempts"] == 2
    assert out["stats"] == {"requested": 1, "accepted": 1, "first_try": 0, "rejected": 0}


def test_generate_drops_a_note_that_never_passes():
    out = notes.generate([ALERT], "m", "h", ask=lambda *a: "Made up 12345.", retries=1)
    assert out["notes"] == {} and out["stats"]["rejected"] == 1
