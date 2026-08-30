"""
Compute every dataset statistic in Table 1 of the paper from the released
data. Speaker counts use canonical names (clean display names resolved
against the MP database).

Reads:  data/corpus/speeches/speeches_*.parquet   (full corpus, from HF)
        data/corpus/debates/debates_*.parquet
        data/womens_rights/speech_classifications.parquet (from HF; run
        scripts/download_data.py first)
Writes: data/results/dataset_stats.json
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPEECH_DIR = REPO / "data" / "corpus" / "speeches"
DEBATE_DIR = REPO / "data" / "corpus" / "debates"
CLASSIFICATIONS = REPO / "data" / "womens_rights" / "speech_classifications.parquet"
OUTPUT_PATH = REPO / "data" / "results" / "dataset_stats.json"


def compute_stats():
    results = {}

    # === FULL CORPUS ===
    total_speeches = 0
    total_debates = 0
    all_canonical = set()
    years = set()
    commons_speeches = 0
    lords_speeches = 0
    commons_male_speeches = 0
    commons_female_speeches = 0
    commons_male_speakers = set()
    commons_female_speakers = set()
    commons_gendered = 0

    speech_files = sorted(SPEECH_DIR.glob("speeches_*.parquet"))
    if not speech_files:
        raise SystemExit(f"No corpus files in {SPEECH_DIR}; download the "
                         "corpus/ folder from the Hugging Face dataset first")

    for f in speech_files:
        df = pd.read_parquet(
            f, columns=["speech_id", "canonical_name", "gender", "chamber", "year"]
        )
        total_speeches += len(df)
        all_canonical.update(df["canonical_name"].dropna().unique())
        years.update(df["year"].unique())

        commons = df[df["chamber"].str.contains("Commons", na=False, case=False)]
        lords = df[~df["chamber"].str.contains("Commons", na=False, case=False)]
        commons_speeches += len(commons)
        lords_speeches += len(lords)

        cm = commons[commons["gender"] == "M"]
        cf = commons[commons["gender"] == "F"]
        commons_male_speeches += len(cm)
        commons_female_speeches += len(cf)
        commons_gendered += len(cm) + len(cf)
        commons_male_speakers.update(cm["canonical_name"].dropna().unique())
        commons_female_speakers.update(cf["canonical_name"].dropna().unique())

    for f in sorted(DEBATE_DIR.glob("debates_*.parquet")):
        total_debates += len(pd.read_parquet(f, columns=["debate_id"]))

    results["full_corpus"] = {
        "total_debates": total_debates,
        "total_speeches": total_speeches,
        "total_speakers_canonical": len(all_canonical),
        "year_range": f"{min(years)}-{max(years)}",
        "unique_years": len(years),
    }

    results["commons"] = {
        "total_speeches": commons_speeches,
        "lords_speeches": lords_speeches,
        "gender_matched_speeches": commons_gendered,
        "gender_match_pct": round(commons_gendered / commons_speeches * 100, 1),
        "male_speeches": commons_male_speeches,
        "female_speeches": commons_female_speeches,
        "male_speakers": len(commons_male_speakers),
        "female_speakers": len(commons_female_speakers),
    }

    # === WOMEN'S RIGHTS SUBSET ===
    suf = pd.read_parquet(CLASSIFICATIONS)
    m_suf = suf[suf["gender"] == "M"]
    f_suf = suf[suf["gender"] == "F"]
    u_suf = suf[~suf["gender"].isin(["M", "F"])]
    n_speakers = suf["canonical_name"].nunique()

    results["womens_rights_subset"] = {
        "total_speeches": len(suf),
        "year_range": f"{int(suf['year'].min())}-{int(suf['year'].max())}",
        "unique_speakers": n_speakers,
        "male_speakers": m_suf["canonical_name"].nunique(),
        "female_speakers": f_suf["canonical_name"].nunique(),
        "unknown_speakers": u_suf["canonical_name"].nunique(),
        "male_speeches": len(m_suf),
        "female_speeches": len(f_suf),
        "unknown_speeches": len(u_suf),
        "male_speakers_pct": round(m_suf["canonical_name"].nunique() / n_speakers * 100, 1),
        "female_speakers_pct": round(f_suf["canonical_name"].nunique() / n_speakers * 100, 1),
        "unknown_speakers_pct": round(u_suf["canonical_name"].nunique() / n_speakers * 100, 1),
        "male_speeches_pct": round(len(m_suf) / len(suf) * 100, 1),
        "female_speeches_pct": round(len(f_suf) / len(suf) * 100, 1),
        "unknown_speeches_pct": round(len(u_suf) / len(suf) * 100, 1),
    }

    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}\n")

    fc = results["full_corpus"]
    c = results["commons"]
    s = results["womens_rights_subset"]
    print("=== Table 1 ===")
    print(f"Total debates: {fc['total_debates']:,}")
    print(f"Total speeches: {fc['total_speeches']:,}")
    print(f"Speakers (canonical): {fc['total_speakers_canonical']:,}")
    print(f"Year range: {fc['year_range']}")
    print(f"Commons speeches: {c['total_speeches']:,}")
    print(f"Gender-matched: {c['gender_matched_speeches']:,} ({c['gender_match_pct']}%)")
    print(f"Male MPs: {c['male_speakers']:,} ({c['male_speeches']:,} speeches)")
    print(f"Female MPs: {c['female_speakers']:,} ({c['female_speeches']:,} speeches)")
    print(f"\nWomen's rights speeches: {s['total_speeches']:,}")
    print(f"Year range: {s['year_range']}")
    print(f"Unique speakers: {s['unique_speakers']:,}")
    print(f"  Male: {s['male_speakers']} ({s['male_speakers_pct']}%), {s['male_speeches']} speeches ({s['male_speeches_pct']}%)")
    print(f"  Female: {s['female_speakers']} ({s['female_speakers_pct']}%), {s['female_speeches']} speeches ({s['female_speeches_pct']}%)")
    print(f"  Unknown: {s['unknown_speakers']} ({s['unknown_speakers_pct']}%), {s['unknown_speeches']} speeches ({s['unknown_speeches_pct']}%)")


if __name__ == "__main__":
    compute_stats()
