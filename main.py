#!/usr/bin/env python3
"""Lead Intelligence System.

Scores leads deterministically against config.yaml, batches the ambiguous
"review" band and outreach-message writing out to an LLM (with a
deterministic fallback if the LLM is unavailable), ranks the result, and
writes output_report.json.

Usage:
    python main.py leads.csv [--config config.yaml] [--output output_report.json] [--no-llm]
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = ["name", "company", "company_size", "industry", "source", "last_interaction_date"]
KNOWN_STATUSES = {"known", "known_core", "known_low_fit"}


# ---------------------------------------------------------------- env / config

def load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't add a dependency for one job. Existing
    env vars win (a real `export` should always beat a stale .env file)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- normalization

def is_missing(value, missing_tokens) -> bool:
    if value is None:
        return True
    return str(value).strip() in missing_tokens


# ---------------------------------------------------------------- per-factor scoring
# Each scorer returns a score plus a status. status in KNOWN_STATUSES means the
# value was usable as-is; anything else ("missing", "unknown", "unrecognized",
# "unparseable") is an ignorance signal that can trigger the review-upgrade rule.

def score_source(source_raw, cfg):
    tokens = cfg["data_quality"]["missing_tokens"]
    sc = cfg["source_scores"]
    if is_missing(source_raw, tokens):
        return sc["_missing"], "missing"
    s = str(source_raw).strip()
    if s in sc:
        return sc[s], "known"
    return sc["_unknown"], "unknown"


def score_size(size_raw, cfg):
    tokens = cfg["data_quality"]["missing_tokens"]
    band_cfg = cfg["company_size_bands"]
    if is_missing(size_raw, tokens):
        return band_cfg["missing_score"], None, "missing"
    cleaned = str(size_raw).strip().replace(",", "")
    try:
        n = int(float(cleaned))
    except ValueError:
        return band_cfg["missing_score"], None, "unparseable"
    if n <= 0:
        return band_cfg["missing_score"], None, "unparseable"
    n = min(n, cfg["data_quality"]["max_plausible_company_size"])
    for band in band_cfg["bands"]:
        if band["max"] is None or n <= band["max"]:
            return band["score"], n, "known"
    return band_cfg["bands"][-1]["score"], n, "known"  # unreachable safety net


def score_industry(industry_raw, cfg):
    tokens = cfg["data_quality"]["missing_tokens"]
    fit = cfg["industry_fit"]
    if is_missing(industry_raw, tokens):
        return fit["_default"], "missing"
    s = str(industry_raw).strip()
    if s in fit["core"]:
        return fit["core_score"], "known_core"
    if s in fit["low_fit"]:
        return fit["low_fit_score"], "known_low_fit"
    return fit["_default"], "unrecognized"


def score_recency(date_raw, source_raw, ref_date, cfg):
    tokens = cfg["data_quality"]["missing_tokens"]
    rc = cfg["recency_bands"]
    missing = is_missing(date_raw, tokens)
    parsed = None
    if not missing:
        try:
            parsed = dt.date.fromisoformat(str(date_raw).strip())
        except ValueError:
            missing = True  # unparseable date is treated the same as no date

    if missing:
        md = rc["missing_date"]
        # Same blank cell, opposite meaning: a brand-new inbound/referral lead
        # vs. an untouched cold-outreach record. Disambiguate by source.
        src = None if is_missing(source_raw, tokens) else str(source_raw).strip()
        if src in md["high_intent_sources"]:
            return md["high_intent_score"], None, "missing_date_high_intent"
        return md["default_score"], None, "missing_date_default"

    days = (ref_date - parsed).days
    if days < 0:
        days = 0  # future-dated row (clock skew / bad entry) - clamp, don't penalize or crash
    for band in rc["bands"]:
        if band["max_days"] is None or days <= band["max_days"]:
            return band["score"], days, "known"
    return rc["bands"][-1]["score"], days, "known"  # unreachable safety net


# ---------------------------------------------------------------- row scoring

def build_reasoning(source_raw, size_val, industry_raw, src_score, size_score, rec_score, ind_score,
                     days, total, decision, flags, upgraded, weights):
    size_txt = f"{size_val} employees" if size_val is not None else "size unknown"
    rec_txt = f"{days}d since last touch" if days is not None else "no interaction date"
    bits = [
        f"source={source_raw or '(none)'} ({src_score}/10, w{weights['source']:.0%})",
        f"{size_txt} ({size_score}/10, w{weights['company_size']:.0%})",
        f"{rec_txt} ({rec_score}/10, w{weights['recency']:.0%})",
        f"industry={industry_raw or '(none)'} ({ind_score}/10, w{weights['industry']:.0%})",
    ]
    reasoning = "; ".join(bits) + f" -> fit {total}/10, {decision}."
    if flags:
        reasoning += f" Uncertain factors: {', '.join(flags)}."
    if upgraded:
        reasoning += " Reject upgraded to review: verdict relied on missing/unknown data."
    return reasoning


REJECTION_REASON_LABELS = {
    "source": "low-intent source",
    "company_size": "poor company-size fit",
    "recency": "stale interaction",
    "industry": "low-fit industry",
}


def dominant_reject_reason(src_score, size_score, rec_score, ind_score, weights) -> str:
    """Which single factor drove a reject, for the 'common rejection reasons'
    aggregate stat. Lowest raw score wins; ties broken toward the more
    heavily-weighted factor, since that one moved the total further."""
    factors = {"source": src_score, "company_size": size_score, "recency": rec_score, "industry": ind_score}
    floor = min(factors.values())
    candidates = [k for k, v in factors.items() if v == floor]
    dominant = max(candidates, key=lambda k: weights[k])
    return REJECTION_REASON_LABELS[dominant]


def score_row(row, cfg, ref_date):
    w = cfg["weights"]
    source_raw = row.get("source", "")
    size_raw = row.get("company_size", "")
    date_raw = row.get("last_interaction_date", "")
    industry_raw = row.get("industry", "")

    src_score, src_status = score_source(source_raw, cfg)
    size_score, size_val, size_status = score_size(size_raw, cfg)
    rec_score, days, rec_status = score_recency(date_raw, source_raw, ref_date, cfg)
    ind_score, ind_status = score_industry(industry_raw, cfg)

    total = round(
        w["source"] * src_score
        + w["company_size"] * size_score
        + w["recency"] * rec_score
        + w["industry"] * ind_score,
        2,
    )

    flags = []
    if src_status not in KNOWN_STATUSES:
        flags.append(f"source:{src_status}")
    if size_status not in KNOWN_STATUSES:
        flags.append(f"company_size:{size_status}")
    if rec_status not in KNOWN_STATUSES:
        flags.append(f"recency:{rec_status}")
    if ind_status not in KNOWN_STATUSES:
        flags.append(f"industry:{ind_status}")

    th = cfg["thresholds"]
    if total >= th["qualified_min"]:
        decision = "qualified"
    elif total >= th["review_min"]:
        decision = "review"
    else:
        decision = "rejected"

    upg = cfg["review_upgrade_rules"]
    upgraded = False
    if decision == "rejected" and len(flags) >= upg["min_flags_to_force_review"]:
        # substring, not endswith: recency's missing-date statuses are
        # "missing_date_high_intent"/"missing_date_default", not a bare "missing"
        missing_data = any("missing" in f or "unparseable" in f for f in flags)
        unknown_source = "source:unknown" in flags
        if (upg["upgrade_reject_on_missing_data"] and missing_data) or (
            upg["upgrade_reject_on_unknown_source"] and unknown_source
        ):
            decision = "review"
            upgraded = True

    reasoning = build_reasoning(
        source_raw, size_val, industry_raw, src_score, size_score, rec_score, ind_score,
        days, total, decision, flags, upgraded, w,
    )
    rejection_reason = (
        dominant_reject_reason(src_score, size_score, rec_score, ind_score, w) if decision == "rejected" else None
    )

    return pd.Series({
        "fit_score": total,
        "decision": decision,
        "reasoning": reasoning,
        "flags": flags,
        "rejection_reason": rejection_reason,
        "days_since_interaction": days,
        "source_score": src_score,
        "company_size_score": size_score,
        "company_size_parsed": size_val,
        "recency_score": rec_score,
        "industry_score": ind_score,
    })


# ---------------------------------------------------------------- reference date

def compute_reference_date(df: pd.DataFrame, cfg: dict) -> dt.date:
    override = cfg["reference_date"].get("override")
    if override:
        return dt.date.fromisoformat(str(override))

    tokens = cfg["data_quality"]["missing_tokens"]
    dates = []
    for raw in df["last_interaction_date"]:
        if is_missing(raw, tokens):
            continue
        try:
            dates.append(dt.date.fromisoformat(str(raw).strip()))
        except ValueError:
            continue
    if not dates:
        # No usable date anywhere in the file - fall back to wall-clock time
        # rather than crash. Rare: every row would need a blank/bad date.
        return dt.date.today()
    return max(dates)


# ---------------------------------------------------------------- LLM (batched, optional)

def get_llm_client(cfg):
    if not cfg["llm"]["enabled"]:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _extract_json(text: str):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[start : end + 1])


def call_with_retries(fn, cfg):
    """Runs fn() up to max_retries times with exponential backoff.
    Returns fn()'s result, or None if every attempt failed (caller falls
    back to deterministic behaviour - an LLM outage must degrade, not crash)."""
    max_retries = cfg["llm"]["max_retries"]
    backoff = cfg["llm"]["backoff_seconds"]
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any API/parse failure triggers backoff+fallback
            if attempt == max_retries - 1:
                print(f"  [llm] giving up after {max_retries} attempts: {exc}", file=sys.stderr)
                return None
            time.sleep(backoff * (2 ** attempt))
    return None


def build_review_prompt(batch: pd.DataFrame, cfg) -> str:
    """Exact prompt text sent for the review-band tiebreak. Split out from
    llm_review_tiebreak so it can be reused (e.g. to preview what the LLM
    stage would do without a live API key) without touching the real call path."""
    lines = []
    for row_id, r in batch.iterrows():
        lines.append(
            f'{row_id}: company="{r["company"]}", industry="{r["industry"]}", '
            f'company_size={r["company_size"]}, source="{r["source"]}", '
            f'fit_score={r["fit_score"]}, reasoning="{r["reasoning"]}"'
        )
    return (
        "You are tie-breaking B2B SaaS leads that scored in the ambiguous review band "
        "(not a clear qualify or reject). For each lead below, decide \"qualified\" "
        "(worth a sales rep's time) or \"rejected\", or keep \"review\" if it is genuinely "
        "still too close to call. Respond with ONLY a JSON array, one object per lead:\n"
        '[{"id": <id>, "verdict": "qualified"|"rejected"|"review", "note": "<one short sentence why>"}]\n\n'
        + "\n".join(lines)
    )


def llm_review_tiebreak(client, model, batch: pd.DataFrame, cfg):
    """Asks the LLM to make the close call on review-band leads. Returns
    {row_id: {"verdict": ..., "note": ...}} or None on total failure."""
    prompt = build_review_prompt(batch, cfg)

    def _call():
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(resp.content[0].text)

    result = call_with_retries(_call, cfg)
    if result is None:
        return None
    out = {}
    for item in result:
        try:
            out[int(item["id"])] = {"verdict": item["verdict"], "note": item.get("note", "")}
        except (KeyError, ValueError, TypeError):
            continue
    return out


def build_outreach_prompt(batch: pd.DataFrame, cfg) -> str:
    """Exact prompt text sent for outreach-message generation. Split out from
    llm_outreach_messages for the same reason as build_review_prompt above."""
    n = cfg["llm"]["outreach_variants"]
    lines = []
    for row_id, r in batch.iterrows():
        lines.append(
            f'{row_id}: name="{r["name"]}", company="{r["company"]}", industry="{r["industry"]}", '
            f'source="{r["source"]}", reasoning="{r["reasoning"]}"'
        )
    return (
        f"Write {n} short (2-3 sentence), DISTINCT outreach email openers for each qualified "
        "B2B SaaS lead below - a rep will pick whichever variant fits their voice, so vary the "
        "angle across the options (e.g. lead with their source/trigger event, their industry "
        "pain point, or a direct value prop) rather than rephrasing the same sentence. Reference "
        "their company/industry/source naturally. No subject line, no signature. Respond with "
        "ONLY a JSON array:\n"
        f'[{{"id": <id>, "messages": ["<variant 1>", ..., "<variant {n}>"]}}]\n\n' + "\n".join(lines)
    )


def llm_outreach_messages(client, model, batch: pd.DataFrame, cfg):
    """Returns {row_id: [message, ...]} (N variants per lead, N = config
    llm.outreach_variants) or None on total failure."""
    n = cfg["llm"]["outreach_variants"]
    prompt = build_outreach_prompt(batch, cfg)

    def _call():
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(resp.content[0].text)

    result = call_with_retries(_call, cfg)
    if result is None:
        return None
    out = {}
    for item in result:
        try:
            msgs = [str(m) for m in item["messages"]][:n]
            if msgs:
                out[int(item["id"])] = msgs
        except (KeyError, ValueError, TypeError):
            continue
    return out


def fallback_outreach_messages(row, n: int) -> list:
    """Deterministic templates used when the LLM is unavailable. Not
    personalised prose, but every qualified lead still gets N actionable,
    angle-varied messages - never a crash, never zero output."""
    name = row["name"] or "there"
    company = row["company"] or "your company"
    source = row["source"] or "your recent activity"
    industry = row["industry"] or "team"
    templates = [
        f"Hi {name}, saw {company}'s interest via {source} - worth 15 minutes this week to see "
        f"if we're a fit for your {industry} team?",
        f"Hi {name}, we work with a lot of {industry} companies facing the same scaling questions "
        f"{company} is likely hitting right now - open to a quick intro call?",
        f"Hi {name}, following up on {company}'s {source.lower()} - happy to walk through how "
        f"similar teams got value in under a week, if useful?",
    ]
    return templates[:n] if n <= len(templates) else (templates + templates[: n - len(templates)])


def chunk_index(idx, size):
    idx = list(idx)
    for i in range(0, len(idx), size):
        yield idx[i : i + size]


def apply_llm_stages(df: pd.DataFrame, cfg: dict, use_llm: bool) -> bool:
    """Mutates df in place: review-band tiebreak, then outreach messages for
    whatever ends up 'qualified'. Returns whether the LLM was actually used."""
    df["llm_note"] = ""
    df["outreach_messages"] = [[] for _ in range(len(df))]

    client = get_llm_client(cfg) if use_llm else None
    model = os.environ.get("ANTHROPIC_MODEL") or cfg["llm"]["model"]
    batch_size = cfg["llm"]["batch_size"]
    n_variants = cfg["llm"]["outreach_variants"]
    llm_used = False

    review_ids = df.index[df["decision"] == "review"]
    if client is not None:
        for chunk in chunk_index(review_ids, batch_size):
            result = llm_review_tiebreak(client, model, df.loc[chunk], cfg)
            if result is None:
                for rid in chunk:
                    df.at[rid, "llm_note"] = "LLM unavailable - kept deterministic 'review' verdict"
                continue
            llm_used = True
            for rid in chunk:
                r = result.get(rid)
                if not r or r["verdict"] not in ("qualified", "rejected", "review"):
                    df.at[rid, "llm_note"] = "LLM response incomplete - kept deterministic 'review' verdict"
                    continue
                if r["verdict"] != "review":
                    df.at[rid, "decision"] = r["verdict"]
                df.at[rid, "llm_note"] = f"LLM tiebreak: {r['note']}" if r["note"] else "LLM tiebreak applied"
    else:
        for rid in review_ids:
            df.at[rid, "llm_note"] = "LLM disabled/unavailable - kept deterministic 'review' verdict"

    qualified_ids = df.index[df["decision"] == "qualified"]
    if client is not None:
        for chunk in chunk_index(qualified_ids, batch_size):
            result = llm_outreach_messages(client, model, df.loc[chunk], cfg)
            if result is None:
                for rid in chunk:
                    df.at[rid, "outreach_messages"] = fallback_outreach_messages(df.loc[rid], n_variants)
                continue
            llm_used = True
            for rid in chunk:
                df.at[rid, "outreach_messages"] = result.get(rid) or fallback_outreach_messages(df.loc[rid], n_variants)
    else:
        for rid in qualified_ids:
            df.at[rid, "outreach_messages"] = fallback_outreach_messages(df.loc[rid], n_variants)

    return llm_used


# ---------------------------------------------------------------- priority ranking

_PRIORITY_SCOPES = {
    "qualified_and_review": ("qualified", "review"),
    "qualified_only": ("qualified",),
    "all": ("qualified", "review", "rejected"),
}
_SORT_KEY_COLUMNS = {"score": "fit_score", "freshness": "_freshness"}


def assign_priority(df: pd.DataFrame, cfg: dict) -> None:
    """Rank is not the score: sort the configured scope by config's sort_by
    order (score desc, then freshness asc - most recent contact first;
    unknown dates rank last within a tie), both driven by config.yaml."""
    p = cfg["priority"]
    df["priority_rank"] = pd.NA
    scope_mask = df["decision"].isin(_PRIORITY_SCOPES[p["scope"]])
    freshness = pd.to_numeric(df["days_since_interaction"], errors="coerce").fillna(10**9)
    sort_cols = [_SORT_KEY_COLUMNS[k] for k in p["sort_by"]]
    ascending = [c != "fit_score" for c in sort_cols]
    ranked = df[scope_mask].assign(_freshness=freshness[scope_mask]).sort_values(
        by=sort_cols, ascending=ascending
    )
    df.loc[ranked.index, "priority_rank"] = range(1, len(ranked) + 1)


# ---------------------------------------------------------------- pipeline

def load_leads(path: str) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: we do our own missing-value handling
    # per config.yaml, so pandas must not silently turn "NA"/"" into NaN first.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    df["_row_id"] = range(len(df))
    df = df.set_index("_row_id", drop=False)
    return df


def run(input_path: str, config_path: str, output_path: str, use_llm: bool) -> dict:
    cfg = load_config(config_path)
    df = load_leads(input_path)
    ref_date = compute_reference_date(df, cfg)

    scored = df.apply(lambda row: score_row(row, cfg, ref_date), axis=1)
    df = pd.concat([df, scored], axis=1)

    llm_used = apply_llm_stages(df, cfg, use_llm)
    assign_priority(df, cfg)

    counts = df["decision"].value_counts().to_dict()
    total = len(df)

    reject_reasons = df.loc[df["decision"] == "rejected", "rejection_reason"].dropna().value_counts()
    common_rejection_reasons = [{"reason": k, "count": int(v)} for k, v in reject_reasons.items()]

    report = {
        "meta": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "input_file": input_path,
            "config_file": config_path,
            "reference_date": ref_date.isoformat(),
            "total_leads": total,
            "counts": {k: counts.get(k, 0) for k in ("qualified", "review", "rejected")},
            "rates": {
                k: round(counts.get(k, 0) / total, 4) if total else 0.0
                for k in ("qualified", "review", "rejected")
            },
            "common_rejection_reasons": common_rejection_reasons,
            "llm_used": llm_used,
        },
        "leads": [],
        "sample_outreach_messages": [],
    }

    ranked_first = df.sort_values(
        by=["priority_rank"], key=lambda s: pd.to_numeric(s, errors="coerce").fillna(10**9)
    )
    for _, r in ranked_first.iterrows():
        entry = {
            "name": r["name"],
            "company": r["company"],
            "company_size": r["company_size"],
            "industry": r["industry"],
            "source": r["source"],
            "last_interaction_date": r["last_interaction_date"],
            "fit_score": r["fit_score"],
            "decision": r["decision"],
            "priority_rank": None if pd.isna(r["priority_rank"]) else int(r["priority_rank"]),
            "reasoning": r["reasoning"],
            "factor_scores": {
                "source": r["source_score"],
                "company_size": r["company_size_score"],
                "recency": r["recency_score"],
                "industry": r["industry_score"],
            },
            "flags": r["flags"],
        }
        if r["llm_note"]:
            entry["llm_note"] = r["llm_note"]
        if r["decision"] == "qualified":
            entry["outreach_messages"] = r["outreach_messages"]
        report["leads"].append(entry)

    # Deliverable: "sample outreach messages (3-5 examples)" - top-ranked
    # qualified leads, pulled straight from what was already generated above.
    sample_n = cfg["llm"].get("sample_outreach_count", 5)
    report["sample_outreach_messages"] = [
        {"name": e["name"], "company": e["company"], "messages": e["outreach_messages"]}
        for e in report["leads"] if e["decision"] == "qualified"
    ][:sample_n]

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Score and rank leads per config.yaml")
    parser.add_argument("input_csv", help="Path to a leads CSV")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="output_report.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls even if a key is configured")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")

    report = run(args.input_csv, args.config, args.output, use_llm=not args.no_llm)
    m = report["meta"]
    print(f"Reference date: {m['reference_date']}")
    print(f"Total leads: {m['total_leads']}")
    print(
        f"Qualified: {m['counts']['qualified']} ({m['rates']['qualified']:.1%})  "
        f"Review: {m['counts']['review']} ({m['rates']['review']:.1%})  "
        f"Rejected: {m['counts']['rejected']} ({m['rates']['rejected']:.1%})"
    )
    print(f"LLM used: {m['llm_used']}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
