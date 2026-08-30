"""
Build corpus_with_context.parquet: 6,531 corpus speeches with proper 5-before +
5-after context windows attached, using the same matching logic as
experiments/20260520_v8_500_validation/01_create_sample.py so that LLM context
matches what human annotators saw.

The v7 pipeline's `context_text` column had "[No context available]" for ~58%
of speeches even when the underlying debate-turns parquet contained context.
This script rebuilds context for all 6,531 speeches from the turns parquet.

Output:
  experiments/may24_rewrite/corpus_with_context.parquet

Columns:
  speech_id, debate_id, target_text, preceding_speeches, following_speeches,
  context_text  (formatted string ready for LLM consumption)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V7_PATH = Path("outputs/llm_classification/v7_notrunc_results.parquet")
TXT_PATH = Path("outputs/llm_classification/suffrage_classified_with_text.parquet")
TURNS_PATH = Path("outputs/llm_classification/suffrage_debates_with_turns.parquet")
OUT_PATH = ROOT / "corpus_with_context.parquet"
CONTEXT_WINDOW = 5  # matches 01_create_sample.py


def attach_context_windows(sample: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """For each row, attach 5-before + 5-after speech records from the turns
    parquet.

    Matching strategy: prefer TEXT-PREFIX match. The speech_id numbering
    between the corpus parquet and turns parquet is off by one or two for
    ~40% of rows, so matching by speech_id silently returns the wrong row
    and produces shifted context (e.g., the target appearing as its own
    'following' speech). Text-prefix matching gives the LLM correct context.

    Fall back to speech_id only when the target text is too short or
    non-unique to match by prefix.
    """
    turns_by_debate = {did: g.sort_values("sequence_number").reset_index(drop=True)
                       for did, g in turns.groupby("debate_id")}

    preceding, following = [], []
    matched_by_text = matched_by_id_only = unmatched = 0
    for _, row in sample.iterrows():
        debate = turns_by_debate.get(row["debate_id"])
        if debate is None:
            preceding.append([]); following.append([]); unmatched += 1; continue

        # Primary: text-prefix match
        tgt = (row["target_text"] or "")[:120]
        hits = []
        if tgt and len(tgt) >= 40:
            hits = debate.index[debate["text"].str.startswith(tgt, na=False)].tolist()

        if len(hits) == 1:
            matched_by_text += 1
            i = int(hits[0])
        else:
            # Fallback: speech_id
            id_hits = debate.index[debate["speech_id"] == row["speech_id"]].tolist()
            if len(id_hits) == 1:
                matched_by_id_only += 1
                i = int(id_hits[0])
            elif len(hits) > 1:
                # Multiple text matches: prefer the one whose speech_id also matches
                refined = [h for h in hits
                           if debate.iloc[h]["speech_id"] == row["speech_id"]]
                if len(refined) == 1:
                    matched_by_text += 1
                    i = int(refined[0])
                else:
                    preceding.append([]); following.append([]); unmatched += 1; continue
            else:
                preceding.append([]); following.append([]); unmatched += 1; continue

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

    print(f"Match: by_text_prefix={matched_by_text}  by_id_fallback={matched_by_id_only}  "
          f"unmatched={unmatched}  total={len(sample)}")
    sample = sample.copy()
    sample["preceding_speeches"] = preceding
    sample["following_speeches"] = following
    return sample


def format_context(preceding: list, following: list) -> str:
    """Format the 5-before + 5-after speeches as a single string for the LLM.

    Mirrors the structure shown in the annotation app: preceding speeches in
    chronological order, then following speeches in chronological order. Each
    speech labeled with its speaker.
    """
    if not preceding and not following:
        return "[No context available]"

    parts = []
    if preceding:
        parts.append(f"PRECEDING SPEECHES ({len(preceding)} in chronological order):")
        for s in preceding:
            parts.append("")
            parts.append(f"[Speaker: {s['speaker']}]")
            parts.append(s["text"])
    if following:
        if preceding:
            parts.append("")
        parts.append(f"FOLLOWING SPEECHES ({len(following)} in chronological order):")
        for s in following:
            parts.append("")
            parts.append(f"[Speaker: {s['speaker']}]")
            parts.append(s["text"])
    return "\n".join(parts)


def main() -> None:
    print("Loading source parquets...")
    v7 = pd.read_parquet(V7_PATH)
    txt = pd.read_parquet(TXT_PATH)
    turns = pd.read_parquet(TURNS_PATH)
    print(f"  v7: {len(v7)}  txt: {len(txt)}  turns: {len(turns)}")

    # Merge in metadata + target text from the txt parquet, dropping the
    # original (buggy) context_text -- we'll rebuild it.
    base = v7[["speech_id"]].merge(
        txt[["speech_id", "debate_id", "target_text"]],
        on="speech_id", how="left",
    )
    missing = base["target_text"].isna().sum()
    if missing:
        raise SystemExit(f"FATAL: {missing} corpus speeches missing target_text")

    print("Attaching 5+5 context windows...")
    base = attach_context_windows(base, turns)

    print("Formatting context_text...")
    base["context_text"] = [
        format_context(p, f) for p, f in zip(
            base["preceding_speeches"], base["following_speeches"]
        )
    ]

    # Diagnostics: how much context now vs the old (buggy) parquet?
    new_with_ctx = (base["context_text"] != "[No context available]").sum()
    old_with_ctx = (txt["context_text"].fillna("") != "[No context available]")
    old_with_ctx = old_with_ctx.sum()
    print()
    print("=== Coverage check ===")
    print(f"  Old context_text usable (suffrage_classified_with_text):  "
          f"{old_with_ctx:>5} / {len(txt)}")
    print(f"  New context_text usable (corpus_with_context, this file): "
          f"{new_with_ctx:>5} / {len(base)}")
    usable_lens = base.loc[
        base["context_text"] != "[No context available]", "context_text"
    ].str.len()
    print()
    print(f"  Median formatted context_text length (usable): "
          f"{usable_lens.median():.0f} chars")

    cols = ["speech_id", "debate_id", "target_text", "preceding_speeches",
            "following_speeches", "context_text"]
    base[cols].to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
