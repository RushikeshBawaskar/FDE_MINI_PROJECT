# Lead Intelligence System

CLI that ingests a leads CSV, scores + gates + ranks every lead deterministically, and
batches an LLM for the two jobs code can't do: judgment calls on the ambiguous band and
personalized outreach copy.

## How to run

```bash
uv sync                        # installs pandas / PyYAML / anthropic into .venv, from pyproject.toml
cp .env.example .env           # optional - fill ANTHROPIC_API_KEY to enable the LLM stages

uv run python main.py leads_full_scale_100.csv --output output_report.json
uv run python test_scoring.py  # self-check for the scoring engine, no framework
```
No API key? The run still completes end-to-end: the review-band tiebreak keeps its
deterministic verdict and outreach messages fall back to templates — that's the required
"LLM outage degrades the output, doesn't crash it" behavior, not a special mode. `--no-llm`
forces that path even with a key present, for reproducible runs.

## The rubric (config.yaml, not code)

Four weighted factors: **source** (0.30, who initiated), **company size** (0.30, banded to
pricing tiers, not "bigger is better"), **recency** (0.25, days since last touch measured
against the *newest date in the input file*, never wall-clock `today` — this training data
is ~2.5y old and `today` collapses every lead to maximally stale), **industry** (0.15, a
small core/low-fit list, everything else neutral — 26 industries in 29 training rows makes
an exhaustive map pointless). Score ≥8.0 → `qualified`, ≥7.0 → `review`, else `rejected` —
except a reject built on missing/unknown data always upgrades to `review` (a false reject is
invisible and permanent; a false approve costs a rep 15 minutes). Deterministic code scores
and gates every lead identically on every run; the LLM only tie-breaks the review band and
writes 3 outreach-message variants per qualified lead (creative options, a rep picks one),
batched, never scores.

## Calibration (honest, and thin)

Sweeping cutoffs on all 29 `leads_training.csv` rows with this exact rubric gives
**7 qualified (24%) / 6 review (21%) / 16 rejected (55%)** at qualified_min=8.0,
review_min=7.0. 24% is ~5x the company's current 5% pass rate; a 21% review queue is ~250
leads/month at their volume — workable for a small human team. **29 rows is a thin basis for
cutoffs** — these bands are noisy, not a statistically sound threshold, and should be
revisited once real conversion data exists. That's a limit of the constraint (no tuning
against the held-out files), not something a bigger training sample would route around.

## Known limitations

- **No BANT signal.** The six columns carry no title, seniority, geography, or budget field.
  `qualified` means *"worth a rep contacting"*, not "qualified opportunity" — company size is
  at best a weak budget proxy, and there's no authority signal at all.
- **`Sales call` is ambiguous** (booked call vs. outbound cold dial) — scored neutral (5/10)
  in `config.yaml` rather than guessed either way.
- **Unrecognized source/industry values never reject** — only fall to a neutral score plus a
  flag. An unknown category is missing information, not a negative signal.
- Fallback outreach messages (no LLM) are templated, not personalized prose — 3 angle-varied
  templates per lead, but not on-brand copywriting. That gap is what the LLM path closes.
- The 100-lead and 20-lead edge-case runs were produced without ever inspecting those files'
  rows directly, per this project's constraint on the held-out evaluation sets.

## What's in `output_report.json`

Per lead: `fit_score`, `decision`, `priority_rank`, `reasoning` (built from which factors
fired, not templated prose), `factor_scores`, `flags`, and (if qualified) 3
`outreach_messages` variants. At the top: processed/qualified/review/rejected counts and
rates, `common_rejection_reasons` (which factor most often drove a reject), and a
`sample_outreach_messages` block with 5 worked examples.

## Files

`main.py` · `config.yaml` (rubric + thresholds, one-line rationale per value) ·
`pyproject.toml` / `uv.lock` (env, via `uv`) · `output_report.json` (100-lead run) ·
`leads_sample.csv` (35 self-authored leads, mixed wins/rejects/unclear) ·
`fixtures_edge_cases.csv` + `test_scoring.py` (missing-data / malformed-row coverage,
written before the held-out files were ever touched).
