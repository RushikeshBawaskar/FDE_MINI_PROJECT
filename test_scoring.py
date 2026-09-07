#!/usr/bin/env python3
"""Self-check for the deterministic scoring core. No framework - plain
asserts, run directly: `python test_scoring.py`."""
import datetime as dt

import pandas as pd

import main as m

CFG = m.load_config("config.yaml")
TOKENS = CFG["data_quality"]["missing_tokens"]


def test_missing_tokens():
    for tok in ["", "NA", "N/A", "null", "None", "Unknown", "-"]:
        assert m.is_missing(tok, TOKENS), tok
    assert not m.is_missing("SaaS", TOKENS)


def test_source_scoring():
    known, status = m.score_source("Referral", CFG)
    assert status == "known" and known == CFG["source_scores"]["Referral"]
    unk, status = m.score_source("Carrier pigeon", CFG)
    assert status == "unknown" and unk == CFG["source_scores"]["_unknown"]
    miss, status = m.score_source("", CFG)
    assert status == "missing" and miss == CFG["source_scores"]["_missing"]


def test_size_scoring():
    score, val, status = m.score_size("120", CFG)
    assert status == "known" and val == 120
    # comma-formatted extreme value must not crash and must clamp into a real band
    score, val, status = m.score_size("50,000", CFG)
    assert status == "known" and val == 50000
    score, val, status = m.score_size("NA", CFG)
    assert status == "missing" and score == CFG["company_size_bands"]["missing_score"]
    score, val, status = m.score_size("0", CFG)
    assert status == "unparseable"
    score, val, status = m.score_size("-5", CFG)
    assert status == "unparseable"


def test_industry_scoring():
    score, status = m.score_industry("SaaS", CFG)
    assert status == "known_core" and score == CFG["industry_fit"]["core_score"]
    score, status = m.score_industry("Government", CFG)
    assert status == "known_low_fit" and score == CFG["industry_fit"]["low_fit_score"]
    score, status = m.score_industry("QuantumFarming", CFG)
    assert status == "unrecognized" and score == CFG["industry_fit"]["_default"]
    score, status = m.score_industry("", CFG)
    assert status == "missing" and score == CFG["industry_fit"]["_default"]


def test_recency_scoring():
    ref = dt.date(2024, 1, 20)
    score, days, status = m.score_recency("2024-01-13", "Referral", ref, CFG)
    assert status == "known" and days == 7 and score == 10

    # blank date, high-intent source -> brand-new lead, scored high
    score, days, status = m.score_recency("", "Referral", ref, CFG)
    assert status == "missing_date_high_intent"
    assert score == CFG["recency_bands"]["missing_date"]["high_intent_score"]

    # blank date, cold source -> untouched record, scored low-neutral
    score, days, status = m.score_recency("", "LinkedIn outreach", ref, CFG)
    assert status == "missing_date_default"
    assert score == CFG["recency_bands"]["missing_date"]["default_score"]

    # unparseable date treated the same as missing, never crashes
    score, days, status = m.score_recency("not-a-date", "Referral", ref, CFG)
    assert status == "missing_date_high_intent"

    # future-dated row clamps to 0 days instead of going negative
    score, days, status = m.score_recency("2024-02-01", "Referral", ref, CFG)
    assert days == 0 and score == 10


def test_reference_date_uses_file_not_wallclock():
    """The single most important guardrail in the brief: recency must be
    computed against max(last_interaction_date) in the file, not today()."""
    df = pd.DataFrame({"last_interaction_date": ["2023-08-30", "2024-01-20", "2023-12-01"]})
    ref = m.compute_reference_date(df, CFG)
    assert ref == dt.date(2024, 1, 20)
    # sanity: this must NOT equal the wall-clock date on an old dataset
    assert ref != dt.date.today()


def test_reject_upgrades_to_review_on_missing_data():
    """A row that would score below review_min purely because factors were
    missing/unknown must never come out as a hard reject."""
    row = pd.Series({
        "source": "",            # missing -> neutral, but flagged
        "company_size": "NA",    # missing -> neutral, but flagged
        "industry": "",          # missing -> neutral, but flagged
        "last_interaction_date": "2022-01-01",  # genuinely stale -> low score
    })
    ref = dt.date(2024, 1, 20)
    result = m.score_row(row, CFG, ref)
    assert len(result["flags"]) >= CFG["review_upgrade_rules"]["min_flags_to_force_review"]
    assert result["decision"] != "rejected"


def test_reject_upgrades_on_missing_date_alone():
    """Regression: a blank last_interaction_date produces status
    'missing_date_default'/'missing_date_high_intent', not a bare 'missing' -
    the upgrade check must still catch it (substring, not endswith)."""
    row = pd.Series({
        "source": "LinkedIn outreach",  # known, low score
        "company_size": "50",           # known
        "industry": "SaaS",             # known, core
        "last_interaction_date": "",    # missing -> only ignorance signal
    })
    ref = dt.date(2024, 1, 20)
    result = m.score_row(row, CFG, ref)
    assert result["flags"] == ["recency:missing_date_default"]
    assert result["decision"] != "rejected", result


def test_strong_row_qualifies():
    row = pd.Series({
        "source": "Referral",
        "company_size": "120",
        "industry": "SaaS",
        "last_interaction_date": "2024-01-18",
    })
    ref = dt.date(2024, 1, 20)
    result = m.score_row(row, CFG, ref)
    assert result["decision"] == "qualified"
    assert result["flags"] == []


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"All {len(tests)} checks passed.")


if __name__ == "__main__":
    run_all()
