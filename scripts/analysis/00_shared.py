"""
Shared data loading for the July 2026 ARR rebuttal checks.

Source of truth: experiments/may24_rewrite/v8_corpus_classifications.csv
(the merged V8 output used for every number in the resubmitted paper:
6,531 keyword-extracted speeches, stance for all, sexism flags for the
2,942 non-irrelevant ones).

Two derived columns are added here:

  tier         'tier1' / 'tier2', re-derived from target_text with the exact
               logic of scripts/analysis/extract_suffrage_reliable.py (the
               extractor that built the corpus). Tier 1 = the explicit
               suffrage regex; Tier 2 = women/female within 25 words of a
               voting-term substring, checked only when Tier 1 misses.
               'no_match' is reported if neither fires (should be ~0; any
               residue means target_text differs from the text the extractor
               saw, and is surfaced rather than hidden).

  speaker_key  normalized speaker name for one-vote-per-speaker aggregation:
               lowercased, punctuation-trimmed, whitespace-collapsed.
               This is string normalization, not entity resolution: distinct
               MPs sharing a surname across decades can merge. The scripts
               report how many unique keys this yields so the effect is
               inspectable.
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "may24_rewrite" / "v8_corpus_classifications.csv"

EXPECTED_TOTAL = 6531
EXPECTED_RELEVANT = 2942

# Verbatim from scripts/analysis/extract_suffrage_reliable.py (high_pattern)
TIER1_PATTERN = re.compile(
    "women.*suffrage|female suffrage|suffrage.*women|"
    "votes for women|suffragette|suffragist|"
    "enfranchise.*women|women.*enfranchise|"
    "equal franchise|"
    "representation of the people.*women|"
    "sex disqualification|"
    "women.*social.*political.*union",
    re.IGNORECASE,
)

# Verbatim from extract_suffrage_reliable.py (_is_medium_confidence)
TIER2_VOTING_PATTERNS = [
    r"vote", r"voting", r"voter", r"voters",
    r"electoral", r"electorate",
    r"franchise", r"enfranchise",
    r"representation",
]
TIER2_WINDOW = 25


def is_tier2(text: str) -> bool:
    """Port of ReliableSuffrageExtractor._is_medium_confidence, unchanged."""
    text_lower = text.lower()
    words = text_lower.split()
    for i, word in enumerate(words):
        if "women" in word or "female" in word:
            start = max(0, i - TIER2_WINDOW)
            end = min(len(words), i + TIER2_WINDOW)
            context = " ".join(words[start:end])
            for pattern in TIER2_VOTING_PATTERNS:
                if re.search(pattern, context):
                    return True
    return False


def derive_tier(text) -> str:
    if not isinstance(text, str) or not text:
        return "no_match"
    if TIER1_PATTERN.search(text):
        return "tier1"
    if is_tier2(text):
        return "tier2"
    return "no_match"


def normalize_speaker(name) -> str | None:
    if not isinstance(name, str):
        return None
    key = re.sub(r"\s+", " ", name.strip().strip(".").strip()).lower()
    return key or None


def load_corpus() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if len(df) != EXPECTED_TOTAL:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL} rows in {CSV_PATH}, got {len(df)}"
        )

    for col in ("llm_hostile", "llm_benevolent"):
        df[col] = df[col].map({True: True, False: False, "True": True, "False": False})

    df["is_irrelevant"] = df["llm_stance"] == "irrelevant"
    n_relevant = int((~df["is_irrelevant"]).sum())
    if n_relevant != EXPECTED_RELEVANT:
        raise AssertionError(
            f"Expected {EXPECTED_RELEVANT} relevant speeches, got {n_relevant}"
        )

    df["tier"] = df["target_text"].map(derive_tier)
    df["speaker_key"] = df["speaker"].map(normalize_speaker)
    return df


def era_order(df: pd.DataFrame) -> list[str]:
    """Eras sorted by their earliest year, so tables read chronologically."""
    return list(df.groupby("era")["year"].min().sort_values().index)
