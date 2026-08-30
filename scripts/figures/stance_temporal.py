"""
Generate Figure 1 (figure_stance_temporal.pdf) from the stance labels.

Left panel: absolute counts by decade (stacked).
Right panel: percentage of relevant speeches by decade.
Both panels mark the 1918 and 1928 Representation of the People Acts.

Reads:  data/classifications/stance_llm.jsonl
        data/womens_rights/speech_classifications.parquet (run
        scripts/download_data.py first)
Writes: data/results/figure_stance_temporal.{pdf,png}
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_PDF = REPO / "data" / "results" / "figure_stance_temporal.pdf"
OUT_PNG = REPO / "data" / "results" / "figure_stance_temporal.png"

# Project palette (CLAUDE.md)
COLORS = {
    "for":        "#10B981",  # green
    "against":    "#3B82C4",  # blue (project male color but works here for "against")
    "both":       "#F59E0B",  # amber
    "irrelevant": "#9CA3AF",  # muted grey
    "text":       "#1F2937",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.labelcolor": COLORS["text"],
    "axes.titlecolor": COLORS["text"],
})


def load_stance():
    stance = {}
    for line in open(REPO / "data" / "classifications" / "stance_llm.jsonl"):
        r = json.loads(line)
        if r.get("parsed"):
            stance[r["speech_id"]] = r["parsed"]["stance"]
    return stance


def main():
    stance = load_stance()
    print(f"Stance rows: {len(stance)}")

    # Get year per speech_id from the corpus metadata
    txt = pd.read_parquet(
        REPO / "data" / "womens_rights" / "speech_classifications.parquet"
    )[["speech_id", "year"]]
    df = pd.DataFrame([{"speech_id": s, "stance": st}
                        for s, st in stance.items()])
    df = df.merge(txt, on="speech_id", how="left")
    df["decade"] = (df["year"] // 10) * 10
    print(f"Year coverage: {df['year'].min()}-{df['year'].max()}")
    print(f"Total decades: {df['decade'].nunique()}")

    # Pivot: decade x stance counts
    counts = (df.groupby(["decade", "stance"]).size()
                .unstack(fill_value=0)
                .reindex(columns=["for", "against", "both", "irrelevant"], fill_value=0))
    print(counts.head())

    # Relevant-only (drop irrelevant) for right panel
    relevant = counts[["for", "against", "both"]]
    relevant_pct = relevant.div(relevant.sum(axis=1), axis=0) * 100

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 4.2), constrained_layout=True
    )

    # --- Left: absolute counts, stacked ---
    decades = counts.index.tolist()
    bottom = np.zeros(len(decades))
    for stance_label in ["for", "against", "both", "irrelevant"]:
        ax_left.bar(decades, counts[stance_label], width=8, bottom=bottom,
                     color=COLORS[stance_label],
                     label=stance_label.capitalize(),
                     edgecolor="white", linewidth=0.4)
        bottom += counts[stance_label].values
    ax_left.set_xlabel("Decade")
    ax_left.set_ylabel("Speeches")
    ax_left.set_title("Absolute counts (all extracted speeches)")
    ax_left.axvline(1918, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_left.axvline(1928, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_left.text(1918, ax_left.get_ylim()[1] * 0.95, " 1918", fontsize=8,
                  color=COLORS["text"], va="top")
    ax_left.text(1928, ax_left.get_ylim()[1] * 0.88, " 1928", fontsize=8,
                  color=COLORS["text"], va="top")
    ax_left.legend(loc="upper left", frameon=False, fontsize=9)

    # --- Right: percentage of relevant (excluding irrelevant) ---
    bottom = np.zeros(len(decades))
    for stance_label in ["for", "against", "both"]:
        ax_right.bar(decades, relevant_pct[stance_label].fillna(0),
                      width=8, bottom=bottom,
                      color=COLORS[stance_label],
                      label=stance_label.capitalize(),
                      edgecolor="white", linewidth=0.4)
        bottom += relevant_pct[stance_label].fillna(0).values
    ax_right.set_xlabel("Decade")
    ax_right.set_ylabel("% of relevant speeches")
    ax_right.set_title("Stance composition (relevant only)")
    ax_right.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_right.set_ylim(0, 100)
    ax_right.axvline(1918, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_right.axvline(1928, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_right.legend(loc="lower right", frameon=False, fontsize=9)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
