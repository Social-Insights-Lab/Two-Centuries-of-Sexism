"""Reproduce the paper's headline statistics from the released data files.

Covers:
  - Stance distribution overall, by era, and by speaker gender (Table 6)
  - Gender-stance chi-squared test (Section 6.2)
  - Logistic regression of supportive stance on gender and decade (Appendix F)
  - Sexism categories by stance, LLM labels (Table 7)
  - Hostile share of sexist speeches over time (Section 6.3)
  - Sexism categories by stance, human labels on the 300-speech
    validation set, with Fisher's exact test (Appendix J)

Inputs:
  - v8_corpus_classifications.parquet (from the Hugging Face dataset;
    see README for download instructions)
  - data/annotations/gold.jsonl (in this repository)

Usage:
  python scripts/analysis/paper_stats.py [--corpus PATH] [--gold PATH]

Requires: pandas, scipy, statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO / "data" / "womens_rights" / "v8_corpus_classifications.parquet"
DEFAULT_GOLD = REPO / "data" / "annotations" / "gold.jsonl"

RELEVANT = ["for", "against", "both"]


def stance_table(df, label):
    n = len(df)
    if n == 0:
        return f"{label}: n=0"
    pcts = {s: 100 * (df["llm_stance"] == s).sum() / n for s in RELEVANT}
    return (f"{label:<12} n={n:<6} for={pcts['for']:.0f}%  "
            f"against={pcts['against']:.0f}%  both={pcts['both']:.0f}%")


def sexism_by_stance(df, hostile_col, benevolent_col, stance_col):
    rows = {}
    for s in RELEVANT:
        sub = df[df[stance_col] == s]
        n = len(sub)
        if n == 0:
            continue
        h = sub[hostile_col].astype(bool)
        b = sub[benevolent_col].astype(bool)
        rows[s] = {
            "n": n,
            "no_sexism": 100 * (~h & ~b).sum() / n,
            "hostile_only": 100 * (h & ~b).sum() / n,
            "benevolent_only": 100 * (~h & b).sum() / n,
            "both": 100 * (h & b).sum() / n,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus)
    assert len(df) == 6531, f"expected 6531 speeches, got {len(df)}"
    # 14 relevant speeches have a null benevolent flag (the model returned
    # schema-incomplete output); the paper counts them as not sexist.
    for col in ("llm_hostile", "llm_benevolent"):
        df[col] = df[col].map({True: True, "True": True}).fillna(False).astype(bool)

    rel = df[df["llm_stance"].isin(RELEVANT)].copy()
    print(f"Total speeches: {len(df)}")
    print(f"Irrelevant: {len(df) - len(rel)} "
          f"({100 * (len(df) - len(rel)) / len(df):.0f}%)")
    print(f"Relevant: {len(rel)}\n")

    print("=== Stance distribution (Table 6) ===")
    era_order = list(rel.groupby("era")["year"].min().sort_values().index)
    for era in era_order:
        print(stance_table(rel[rel["era"] == era], era))
    print(stance_table(rel[rel["gender"] == "F"], "Female MPs"))
    print(stance_table(rel[rel["gender"] == "M"], "Male MPs"))
    print(stance_table(rel, "Overall"))

    print("\n=== Gender-stance test (Section 6.2) ===")
    g = rel[rel["gender"].isin(["M", "F"])]
    male, female = g[g["gender"] == "M"], g[g["gender"] == "F"]
    m_for = 100 * (male["llm_stance"] == "for").sum() / len(male)
    f_for = 100 * (female["llm_stance"] == "for").sum() / len(female)
    table = [
        [(male["llm_stance"] == "for").sum(), (male["llm_stance"] != "for").sum()],
        [(female["llm_stance"] == "for").sum(), (female["llm_stance"] != "for").sum()],
    ]
    chi2, p, _, _ = stats.chi2_contingency(table)
    print(f"Male for: {m_for:.1f}%  Female for: {f_for:.1f}%")
    print(f"Chi-squared: {chi2:.2f}, p = {p:.2e}  (n = {len(g)} gendered relevant)")

    print("\n=== Logistic regression (Appendix F) ===")
    import statsmodels.api as sm
    X = pd.DataFrame({
        "female": (g["gender"] == "F").astype(int),
        "decade": g["year"] // 10,
    })
    X = sm.add_constant(X)
    y = (g["llm_stance"] == "for").astype(int)
    fit = sm.Logit(y, X).fit(disp=0)
    ors = np.exp(fit.params)
    print(f"Female (vs male): OR = {ors['female']:.2f}, p = {fit.pvalues['female']:.3f}")
    print(f"Decade:           OR = {ors['decade']:.2f}, p = {fit.pvalues['decade']:.3g}")

    print("\n=== Sexism by stance, LLM labels (Table 7) ===")
    n_sexist = (rel["llm_hostile"].astype(bool) | rel["llm_benevolent"].astype(bool)).sum()
    print(f"Sexist speeches: {n_sexist} of {len(rel)} relevant "
          f"({100 * n_sexist / len(rel):.0f}%); "
          f"hostile {rel['llm_hostile'].astype(bool).sum()}, "
          f"benevolent {rel['llm_benevolent'].astype(bool).sum()}, "
          f"both {(rel['llm_hostile'].astype(bool) & rel['llm_benevolent'].astype(bool)).sum()}")
    for s, r in sexism_by_stance(rel, "llm_hostile", "llm_benevolent", "llm_stance").items():
        print(f"{s:<8} n={r['n']:<5} none={r['no_sexism']:.0f}%  "
              f"hostile-only={r['hostile_only']:.0f}%  "
              f"benevolent-only={r['benevolent_only']:.0f}%  both={r['both']:.0f}%")

    print("\n=== Hostile share of sexist speeches by era (Section 6.3) ===")
    sexist = rel[rel["llm_hostile"].astype(bool) | rel["llm_benevolent"].astype(bool)]
    for era in era_order:
        sub = sexist[sexist["era"] == era]
        if len(sub) == 0:
            continue
        print(f"{era:<12} n={len(sub):<5} "
              f"hostile={100 * sub['llm_hostile'].astype(bool).sum() / len(sub):.0f}%  "
              f"benevolent={100 * sub['llm_benevolent'].astype(bool).sum() / len(sub):.0f}%")

    print("\n=== Human labels on validation set (Appendix J) ===")
    gold = pd.DataFrame(
        json.loads(line) for line in open(args.gold)
    )
    grel = gold[gold["stance"].isin(RELEVANT)].copy()
    for s, r in sexism_by_stance(grel, "hostile", "benevolent", "stance").items():
        print(f"{s:<8} n={r['n']:<5} none={r['no_sexism']:.1f}%  "
              f"hostile-only={r['hostile_only']:.1f}%  "
              f"benevolent-only={r['benevolent_only']:.1f}%  both={r['both']:.1f}%")
    gfor = grel[grel["stance"] == "for"]
    gagainst = grel[grel["stance"] == "against"]
    def n_sex(d):
        return (d["hostile"].astype(bool) | d["benevolent"].astype(bool)).sum()
    ftab = [
        [n_sex(gagainst), len(gagainst) - n_sex(gagainst)],
        [n_sex(gfor), len(gfor) - n_sex(gfor)],
    ]
    _, fp = stats.fisher_exact(ftab)
    print(f"Sexism in Against: {n_sex(gagainst)}/{len(gagainst)} "
          f"({100 * n_sex(gagainst) / len(gagainst):.1f}%)  "
          f"For: {n_sex(gfor)}/{len(gfor)} "
          f"({100 * n_sex(gfor) / len(gfor):.1f}%)")
    print(f"Fisher's exact test: p = {fp:.2e}")


if __name__ == "__main__":
    main()
