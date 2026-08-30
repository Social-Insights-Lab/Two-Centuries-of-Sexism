"""
Create a random validation sample of 500 speeches from the keyword-extracted corpus.

Per coauthor decision (WhatsApp thread 2026-05-19): random sample from the
6,531 keyword-extracted speeches, pre-classification. We do NOT stratify by
LLM stance or sexism labels -- those are precisely what the validation is
testing.

The 100 speeches from the earlier pilot annotation study are excluded to
maintain independence between the codebook-development sample and the
validation sample.

Output:
  experiments/20260520_v8_500_validation/validation_sample_500.parquet
  experiments/20260520_v8_500_validation/sampling_stats.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SEED = 42
SHUFFLE_SEED = 43  # separate seed: shuffles presentation order to avoid era-clustered annotation bias
SAMPLE_SIZE = 300
OUTPUT_DIR = Path(__file__).parent
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

ERA_BINS = [1800, 1870, 1918, 1928, 1970, 2010]
ERA_LABELS = ["1800-1869", "1870-1917", "1918-1927", "1928-1969", "1970-2005"]

V7_PATH = Path("outputs/llm_classification/v7_notrunc_results.parquet")
TXT_PATH = Path("outputs/llm_classification/suffrage_classified_with_text.parquet")
TURNS_PATH = Path("outputs/llm_classification/suffrage_debates_with_turns.parquet")
PILOT_PATH = Path("outputs/validation/validation_sample.parquet")
CONTEXT_WINDOW = 5  # number of speeches before/after to capture


def load_corpus():
    v7 = pd.read_parquet(V7_PATH)
    txt = pd.read_parquet(TXT_PATH)
    metadata_cols = [
        "speech_id", "debate_id", "speaker", "gender", "year", "word_count",
        "target_text", "context_text", "party", "chamber", "date",
    ]
    df = v7.merge(txt[metadata_cols], on="speech_id", how="left")
    df["era"] = pd.cut(df["year"], bins=ERA_BINS, labels=ERA_LABELS, right=False)
    return df


def load_pilot_ids():
    if not PILOT_PATH.exists():
        return set()
    return set(pd.read_parquet(PILOT_PATH)["speech_id"])


def attach_context_windows(sample: pd.DataFrame) -> pd.DataFrame:
    """For each sampled speech, attach `preceding_speeches` and
    `following_speeches` columns sourced from the debate-turns parquet.

    The two corpora have inconsistent speech_id numbering (off-by-one in some
    debates), so we match each sample row to its turn by (debate_id, first 80
    chars of text). Each context entry is a dict with keys
    `sequence_number`, `speaker`, `text`, `word_count`.
    """
    turns = pd.read_parquet(TURNS_PATH)
    turns_by_debate = {did: g.sort_values("sequence_number").reset_index(drop=True)
                       for did, g in turns.groupby("debate_id")}

    preceding, following = [], []
    matched_by_id = matched_by_text = 0
    for _, row in sample.iterrows():
        did = row["debate_id"]
        debate = turns_by_debate.get(did)
        if debate is None:
            preceding.append([]); following.append([]); continue

        # Primary: exact speech_id match
        hits = debate.index[debate["speech_id"] == row["speech_id"]]
        if len(hits) > 0:
            matched_by_id += 1
        else:
            # Fallback: match by first 80 chars of text
            tgt = (row["target_text"] or "")[:80]
            if tgt:
                hits = debate.index[debate["text"].str.startswith(tgt, na=False)]
                if len(hits) > 0:
                    matched_by_text += 1
        if len(hits) == 0:
            preceding.append([]); following.append([]); continue
        i = int(hits[0])
        before = debate.iloc[max(0, i - CONTEXT_WINDOW): i]
        after = debate.iloc[i + 1: i + 1 + CONTEXT_WINDOW]
        to_records = lambda d: [
            {
                "sequence_number": int(r["sequence_number"]),
                "speaker": r["speaker"],
                "text": r["text"],
                "word_count": int(r["word_count"]),
            }
            for _, r in d.iterrows()
        ]
        preceding.append(to_records(before))
        following.append(to_records(after))

    matched_total = matched_by_id + matched_by_text
    print(f"Context window match: {matched_total}/{len(sample)} "
          f"(by speech_id: {matched_by_id}, by text prefix: {matched_by_text})")
    sample = sample.copy()
    sample["preceding_speeches"] = preceding
    sample["following_speeches"] = following
    return sample


def load_already_annotated_ids() -> set[str]:
    """Speech IDs that already have at least one annotation written -- these
    are preserved in any re-sample so prior work isn't thrown away."""
    if not ANNOTATIONS_DIR.exists():
        return set()
    ids: set[str] = set()
    for jl in ANNOTATIONS_DIR.glob("*.jsonl"):
        with open(jl) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["speech_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def sample(df, exclude_ids):
    pool = df[~df["speech_id"].isin(exclude_ids)].copy()
    pre_keep_ids = load_already_annotated_ids() & set(pool["speech_id"])

    # Draw the full SAMPLE_SIZE first
    drawn = pool.sample(n=SAMPLE_SIZE, random_state=SEED)

    # Force-include any already-annotated speeches not already in the draw.
    # For each new force-include, drop one random row (deterministic via seed)
    # from the drawn set that isn't itself force-included.
    missing = pre_keep_ids - set(drawn["speech_id"])
    if missing:
        forced = pool[pool["speech_id"].isin(missing)]
        droppable = drawn[~drawn["speech_id"].isin(pre_keep_ids)]
        to_drop = droppable.sample(n=len(forced), random_state=SEED + 1)
        drawn = pd.concat([drawn.drop(to_drop.index), forced])
        print(f"  Preserved {len(forced)} already-annotated speech(es) by swapping out "
              f"{len(to_drop)} random rows from the seed-42 draw.")

    # Note: sample_idx is assigned later (after the post-sampling shuffle in
    # main) so the order shown to annotators is randomized, not chronological.
    drawn = drawn.reset_index(drop=True)
    return drawn, len(pool)


def diagnostics(s, pool_size, excluded):
    print(f"Pool: {pool_size} (after excluding {excluded} pilot speeches)")
    print(f"Drawn: {len(s)} (seed={SEED})")
    print()
    print("=== ERA DISTRIBUTION (informational; no era stratification) ===")
    era_counts = s["era"].value_counts().reindex(ERA_LABELS, fill_value=0)
    for era in ERA_LABELS:
        print(f"  {era}: {era_counts[era]}")
    print()
    print("=== EXPECTED LLM LABEL DISTRIBUTION (reference only, NOT ground truth) ===")
    print("  Stance:")
    print(s["stance"].value_counts().to_string(header=False))
    print()
    print("  Sexism binary:")
    print(s["binary"].value_counts().to_string(header=False))
    print()
    print("  AST (Axis A):")
    print(s["axis_a_label"].value_counts().to_string(header=False))
    print()
    print("=== GENDER ===")
    print(s["gender"].value_counts(dropna=False).to_string(header=False))
    print()
    print("=== WORD COUNT ===")
    print(f"  Median: {s['word_count'].median():.0f}")
    print(f"  Mean:   {s['word_count'].mean():.0f}")
    print(f"  Max:    {s['word_count'].max():.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print stats; do not write")
    parser.add_argument(
        "--include-pilot", action="store_true",
        help="Do NOT exclude the 100-speech pilot sample (default: exclude)",
    )
    args = parser.parse_args()

    df = load_corpus()
    exclude_ids = set() if args.include_pilot else load_pilot_ids()
    s, pool_size = sample(df, exclude_ids)
    diagnostics(s, pool_size, len(exclude_ids))
    s = attach_context_windows(s)

    # Shuffle presentation order so annotators don't see era-clustered runs.
    # Same shuffle for both annotators (deterministic via SHUFFLE_SEED) so
    # 'sample 42/300' refers to the same speech for both.
    s = s.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
    s["sample_idx"] = range(len(s))

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    out_parquet = OUTPUT_DIR / "validation_sample.parquet"
    cols = [
        "sample_idx", "speech_id", "debate_id", "speaker", "gender", "party",
        "chamber", "year", "era", "date", "word_count",
        "target_text", "context_text",
        "preceding_speeches", "following_speeches",
        "stance", "binary", "axis_a_label", "confidence",
    ]
    s[cols].to_parquet(out_parquet, index=False)
    print(f"\nSaved sample: {out_parquet}")

    stats = {
        "seed": SEED,
        "sample_size": len(s),
        "pool_size": pool_size,
        "excluded_pilot": not args.include_pilot,
        "excluded_count": len(exclude_ids),
        "era_distribution": {e: int(s["era"].value_counts().get(e, 0)) for e in ERA_LABELS},
        "llm_stance_counts": s["stance"].value_counts().to_dict(),
        "llm_binary_counts": s["binary"].value_counts().to_dict(),
        "llm_axis_a_counts": s["axis_a_label"].value_counts().to_dict(),
        "gender_counts": s["gender"].value_counts(dropna=False).to_dict(),
        "word_count_median": float(s["word_count"].median()),
        "word_count_mean": float(s["word_count"].mean()),
    }
    stats_path = OUTPUT_DIR / "sampling_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved stats:  {stats_path}")


if __name__ == "__main__":
    main()
