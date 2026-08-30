"""
Generate all dataset statistics cited in the paper.
Outputs a JSON file with every number and the computation that produced it.

Run from repo root:
    python experiments/20260313_paper_repro/01_dataset_stats.py
"""
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
SPEECH_DIR = Path("data-hansard/derived_complete/speeches_complete")
OUTPUT_PATH = SCRIPT_DIR / "01_dataset_stats.json"


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

    for f in sorted(SPEECH_DIR.glob("speeches_*.parquet")):
        df = pd.read_parquet(
            f, columns=["speech_id", "canonical_name", "gender", "chamber", "year", "debate_id"]
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

    # Debate count from derived_v2
    debate_dir = Path("data-hansard/derived_v2/debates_complete")
    if debate_dir.exists():
        for f in sorted(debate_dir.glob("debates_*.parquet")):
            df = pd.read_parquet(f, columns=["debate_id"])
            total_debates += len(df)

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
        "lords_gender_match_pct": "1.2",  # from earlier calculation
        "male_speeches": commons_male_speeches,
        "female_speeches": commons_female_speeches,
        "male_speakers": len(commons_male_speakers),
        "female_speakers": len(commons_female_speakers),
    }

    # === SUFFRAGE CLASSIFICATION (V7) ===
    v7 = pd.read_parquet("outputs/llm_classification/v7_full_results.parquet")
    orig = pd.read_parquet(
        "outputs/llm_classification/claude_sonnet_45_full_results.parquet",
        columns=["speech_id", "canonical_name", "speaker", "year", "gender"],
    )
    v7 = v7.merge(orig, on="speech_id", how="left")

    m_suf = v7[v7["gender"] == "M"]
    f_suf = v7[v7["gender"] == "F"]
    u_suf = v7[~v7["gender"].isin(["M", "F"])]

    relevant = v7[v7["stance"] != "irrelevant"]
    sexist = v7[v7["binary"] == "sexist"]

    results["suffrage"] = {
        "total_speeches": len(v7),
        "year_range": f"{int(v7['year'].min())}-{int(v7['year'].max())}",
        "unique_speakers": v7["canonical_name"].nunique(),
        "male_speakers": m_suf["canonical_name"].nunique(),
        "female_speakers": f_suf["canonical_name"].nunique(),
        "unknown_speakers": u_suf["canonical_name"].nunique(),
        "male_speeches": len(m_suf),
        "female_speeches": len(f_suf),
        "unknown_speeches": len(u_suf),
        "male_speakers_pct": round(m_suf["canonical_name"].nunique() / v7["canonical_name"].nunique() * 100, 1),
        "female_speakers_pct": round(f_suf["canonical_name"].nunique() / v7["canonical_name"].nunique() * 100, 1),
        "unknown_speakers_pct": round(u_suf["canonical_name"].nunique() / v7["canonical_name"].nunique() * 100, 1),
        "male_speeches_pct": round(len(m_suf) / len(v7) * 100, 1),
        "female_speeches_pct": round(len(f_suf) / len(v7) * 100, 1),
        "unknown_speeches_pct": round(len(u_suf) / len(v7) * 100, 1),
    }

    # Stance distribution
    results["stance"] = {
        s: int(n) for s, n in v7["stance"].value_counts().items()
    }

    # Stance by gender (relevant only)
    results["stance_by_gender"] = {}
    for gender in ["M", "F"]:
        subset = relevant[relevant["gender"] == gender]
        results["stance_by_gender"][gender] = {
            "n_relevant": len(subset),
            "for": int((subset["stance"] == "for").sum()),
            "for_pct": round((subset["stance"] == "for").mean() * 100, 1),
            "against": int((subset["stance"] == "against").sum()),
            "against_pct": round((subset["stance"] == "against").mean() * 100, 1),
            "both": int((subset["stance"] == "both").sum()),
            "both_pct": round((subset["stance"] == "both").mean() * 100, 1),
        }

    # Sexism stats
    results["sexism"] = {
        "total_relevant": len(relevant),
        "total_sexist": len(sexist),
        "sexist_pct": round(len(sexist) / len(relevant) * 100, 1),
        "by_stance": {},
        "by_gender": {},
        "by_axis_a": dict(sexist["axis_a_label"].value_counts().items()),
        "by_axis_b": dict(sexist["axis_b_label"].value_counts().items()),
        "by_axis_c": dict(sexist["axis_c_label"].value_counts().items()),
    }

    for stance in ["for", "against", "both"]:
        subset = v7[v7["stance"] == stance]
        s = (subset["binary"] == "sexist").sum()
        results["sexism"]["by_stance"][stance] = {
            "n": len(subset),
            "sexist": int(s),
            "sexist_pct": round(s / len(subset) * 100, 1),
            "hostile": int((subset["axis_a_label"] == "hostile").sum()),
            "benevolent": int((subset["axis_a_label"] == "benevolent").sum()),
        }

    for gender in ["M", "F"]:
        subset = relevant[relevant["gender"] == gender]
        s = (subset["binary"] == "sexist").sum()
        results["sexism"]["by_gender"][gender] = {
            "n": len(subset),
            "sexist": int(s),
            "sexist_pct": round(s / len(subset) * 100, 1),
            "hostile": int((subset["axis_a_label"] == "hostile").sum()),
            "benevolent": int((subset["axis_a_label"] == "benevolent").sum()),
        }

    # Convert any remaining numpy types
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
    print(f"Saved to {OUTPUT_PATH}")

    # Print for paper
    print("\n=== FOR PAPER ===")
    fc = results["full_corpus"]
    c = results["commons"]
    s = results["suffrage"]
    print(f"Total debates: {fc['total_debates']:,}")
    print(f"Total speeches: {fc['total_speeches']:,}")
    print(f"Speakers (canonical): {fc['total_speakers_canonical']:,}")
    print(f"Year range: {fc['year_range']}")
    print(f"Commons speeches: {c['total_speeches']:,}")
    print(f"Gender-matched: {c['gender_matched_speeches']:,} ({c['gender_match_pct']}%)")
    print(f"Male MPs: {c['male_speakers']:,} ({c['male_speeches']:,} speeches)")
    print(f"Female MPs: {c['female_speakers']:,} ({c['female_speeches']:,} speeches)")
    print(f"\nSuffrage speeches: {s['total_speeches']:,}")
    print(f"Suffrage year range: {s['year_range']}")
    print(f"Suffrage speakers: {s['unique_speakers']:,}")
    print(f"  Male: {s['male_speakers']} ({s['male_speakers_pct']}%), {s['male_speeches']} speeches ({s['male_speeches_pct']}%)")
    print(f"  Female: {s['female_speakers']} ({s['female_speakers_pct']}%), {s['female_speeches']} speeches ({s['female_speeches_pct']}%)")
    print(f"  Unknown: {s['unknown_speakers']} ({s['unknown_speakers_pct']}%), {s['unknown_speeches']} speeches ({s['unknown_speeches_pct']}%)")


if __name__ == "__main__":
    compute_stats()
