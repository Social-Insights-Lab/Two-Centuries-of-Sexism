"""
Rebuttal check 1 (Reviewer 9ZNL): are the 55% irrelevant-classified speeches
distributed uniformly across time periods and debate contexts, or could
systematic filtering skew the composition of the final corpus?

Breakdowns of the irrelevant rate by decade, era, keyword tier, chamber, and
speaker gender, plus chi-square tests of independence and a comparison of the
era composition of the corpus before vs after the LLM relevance filter.

Reads:  experiments/may24_rewrite/v8_corpus_classifications.csv (via 00_shared)
Writes: results_irrelevant_by_time.json
        results_irrelevant_by_time.md
        fig_irrelevant_by_decade.png
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("shared", ROOT / "00_shared.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)

JSON_OUT = ROOT / "results_irrelevant_by_time.json"
MD_OUT = ROOT / "results_irrelevant_by_time.md"
FIG_OUT = ROOT / "fig_irrelevant_by_decade.png"

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

COLORS = {"bar": "#3B82C4", "overall": "#F59E0B", "text": "#1F2937"}


def rate_table(df: pd.DataFrame, by: str, order: list | None = None) -> pd.DataFrame:
    g = df.groupby(by, dropna=False).agg(
        n=("is_irrelevant", "size"),
        n_irrelevant=("is_irrelevant", "sum"),
    )
    g["irrelevant_pct"] = (100 * g["n_irrelevant"] / g["n"]).round(1)
    if order is not None:
        g = g.reindex(order)
    return g.reset_index()


def chi2_report(df: pd.DataFrame, by: str) -> dict:
    ct = pd.crosstab(df[by], df["is_irrelevant"])
    chi2, p, dof, expected = chi2_contingency(ct)
    return {
        "chi2": round(float(chi2), 2),
        "p": float(p),
        "dof": int(dof),
        "min_expected_cell": round(float(expected.min()), 1),
    }


def md_table(tab: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(str(c) for c in tab.columns) + " |"]
    lines.append("|" + "---|" * len(tab.columns))
    for _, row in tab.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def main():
    df = shared.load_corpus()
    eras = shared.era_order(df)

    overall_pct = round(100 * df["is_irrelevant"].mean(), 1)
    n_no_match = int((df["tier"] == "no_match").sum())

    by_decade = rate_table(df, "decade").sort_values("decade")
    by_decade = by_decade.astype({"decade": int, "n": int, "n_irrelevant": int})
    by_era = rate_table(df, "era", order=eras)
    by_tier = rate_table(df, "tier")
    by_chamber = rate_table(df, "chamber")
    by_gender = rate_table(df, "gender")

    tier_era = (
        df.groupby(["era", "tier"])
        .agg(n=("is_irrelevant", "size"), n_irrelevant=("is_irrelevant", "sum"))
        .reset_index()
    )
    tier_era["irrelevant_pct"] = (
        100 * tier_era["n_irrelevant"] / tier_era["n"]
    ).round(1)
    tier_era["era"] = pd.Categorical(tier_era["era"], categories=eras, ordered=True)
    tier_era = tier_era.sort_values(["era", "tier"])

    # Does the filter change what the final corpus looks like over time?
    era_share_pre = (100 * df["era"].value_counts(normalize=True)).round(1)
    era_share_post = (
        100 * df.loc[~df["is_irrelevant"], "era"].value_counts(normalize=True)
    ).round(1)
    composition = pd.DataFrame(
        {
            "era": eras,
            "share_pre_filter_pct": [float(era_share_pre.get(e, 0)) for e in eras],
            "share_post_filter_pct": [float(era_share_post.get(e, 0)) for e in eras],
        }
    )
    composition["shift_pp"] = (
        composition["share_post_filter_pct"] - composition["share_pre_filter_pct"]
    ).round(1)

    tests = {
        "decade": chi2_report(df, "decade"),
        "era": chi2_report(df, "era"),
        "tier": chi2_report(df, "tier"),
    }

    wc = (
        df.groupby("is_irrelevant")["word_count"]
        .median()
        .rename(index={False: "relevant", True: "irrelevant"})
        .to_dict()
    )

    results = {
        "source": str(shared.CSV_PATH),
        "n_total": int(len(df)),
        "n_irrelevant": int(df["is_irrelevant"].sum()),
        "overall_irrelevant_pct": overall_pct,
        "tier_no_match_count": n_no_match,
        "by_decade": by_decade.to_dict(orient="records"),
        "by_era": by_era.to_dict(orient="records"),
        "by_tier": by_tier.to_dict(orient="records"),
        "by_tier_x_era": tier_era.astype({"era": str}).to_dict(orient="records"),
        "by_chamber": by_chamber.to_dict(orient="records"),
        "by_gender": by_gender.to_dict(orient="records"),
        "era_composition_pre_vs_post_filter": composition.to_dict(orient="records"),
        "chi2_independence_vs_irrelevant": tests,
        "median_word_count": {k: float(v) for k, v in wc.items()},
    }
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n")

    md = [
        "# Irrelevant-rate distribution (rebuttal check for Reviewer 9ZNL)",
        "",
        f"Source: `{shared.CSV_PATH.name}` (n={len(df):,}; "
        f"{int(df['is_irrelevant'].sum()):,} irrelevant = {overall_pct}%).",
        f"Tier re-derived from target_text with the canonical extractor logic; "
        f"{n_no_match} speeches matched neither tier.",
        "",
        "## By decade",
        md_table(by_decade),
        "",
        "## By era",
        md_table(by_era),
        "",
        "## By keyword tier",
        md_table(by_tier),
        "",
        "## Tier x era",
        md_table(tier_era),
        "",
        "## By chamber",
        md_table(by_chamber),
        "",
        "## By speaker gender",
        md_table(by_gender),
        "",
        "## Era composition of corpus, before vs after the relevance filter",
        md_table(composition),
        "",
        "## Chi-square tests (grouping variable vs irrelevant flag)",
        "",
    ]
    for k, t in tests.items():
        md.append(
            f"- {k}: chi2 = {t['chi2']}, dof = {t['dof']}, p = {t['p']:.3g}, "
            f"min expected cell = {t['min_expected_cell']}"
        )
    md += [
        "",
        "## Median word count",
        f"- relevant: {wc['relevant']:.0f}",
        f"- irrelevant: {wc['irrelevant']:.0f}",
        "",
    ]
    MD_OUT.write_text("\n".join(md))

    plot = by_decade.dropna(subset=["decade"]).copy()
    plot["decade"] = plot["decade"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        plot["decade"], plot["irrelevant_pct"], width=8,
        color=COLORS["bar"], edgecolor="white",
    )
    ax.axhline(
        overall_pct, color=COLORS["overall"], linestyle="--", linewidth=1.5,
        label=f"Overall ({overall_pct}%)",
    )
    for _, row in plot.iterrows():
        ax.annotate(
            f"{int(row['n'])}", (row["decade"], row["irrelevant_pct"]),
            textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=7, color=COLORS["text"],
        )
    ax.set_xlabel("Decade")
    ax.set_ylabel("Speeches classified irrelevant (%)")
    ax.set_title(
        f"LLM relevance filter by decade (n above bars; N={len(df):,} speeches)"
    )
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_OUT, facecolor="white")

    print(f"Overall irrelevant: {overall_pct}%  (no-tier-match: {n_no_match})")
    print(by_era.to_string(index=False))
    print(by_tier.to_string(index=False))
    print(f"\nWrote {JSON_OUT.name}, {MD_OUT.name}, {FIG_OUT.name}")


if __name__ == "__main__":
    main()
