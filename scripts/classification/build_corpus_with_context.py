"""
Build corpus_with_context.parquet: the 6,531 keyword-extracted speeches with
5-before + 5-after context windows attached, drawn from the full corpus so
that the LLM sees the same context human annotators saw.

Reads:  data/womens_rights/speech_classifications.parquet (speech ids,
        debate ids, and target text of the extracted speeches)
        data/corpus/speeches/speeches_*.parquet (full corpus; download the
        corpus/ folder from the Hugging Face dataset first)
Writes: data/womens_rights/corpus_with_context.parquet

Columns:
  speech_id, debate_id, target_text, preceding_speeches, following_speeches,
  context_text  (formatted string ready for LLM consumption)
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLASSIFICATIONS = REPO / "data" / "womens_rights" / "speech_classifications.parquet"
CORPUS_SPEECHES = REPO / "data" / "corpus" / "speeches"
OUT_PATH = REPO / "data" / "womens_rights" / "corpus_with_context.parquet"
CONTEXT_WINDOW = 5  # matches create_validation_sample.py


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


def load_turns(base: pd.DataFrame) -> pd.DataFrame:
    """Load every speech of every debate that contains an extracted speech,
    from the per-year corpus files."""
    debate_ids = set(base["debate_id"])
    files = sorted(CORPUS_SPEECHES.glob("speeches_*.parquet"))
    if not files:
        raise SystemExit(f"No corpus files in {CORPUS_SPEECHES}; download the "
                         "corpus/ folder from the Hugging Face dataset first")
    cols = ["speech_id", "debate_id", "sequence_number", "speaker",
            "text", "word_count"]
    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        df = df[df["debate_id"].isin(debate_ids)]
        if len(df):
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    print("Loading extracted speeches...")
    base = pd.read_parquet(
        CLASSIFICATIONS, columns=["speech_id", "debate_id", "target_text"]
    )
    missing = base["target_text"].isna().sum()
    if missing:
        raise SystemExit(f"FATAL: {missing} corpus speeches missing target_text")

    print("Loading debate turns from the full corpus...")
    turns = load_turns(base)
    print(f"  extracted: {len(base)}  turns: {len(turns)}")

    print("Attaching 5+5 context windows...")
    base = attach_context_windows(base, turns)

    print("Formatting context_text...")
    base["context_text"] = [
        format_context(p, f) for p, f in zip(
            base["preceding_speeches"], base["following_speeches"]
        )
    ]

    new_with_ctx = (base["context_text"] != "[No context available]").sum()
    print()
    print(f"  context_text usable: {new_with_ctx} / {len(base)}")
    usable_lens = base.loc[
        base["context_text"] != "[No context available]", "context_text"
    ].str.len()
    print(f"  Median formatted context_text length (usable): "
          f"{usable_lens.median():.0f} chars")

    cols = ["speech_id", "debate_id", "target_text", "preceding_speeches",
            "following_speeches", "context_text"]
    base[cols].to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
