"""
Rebuttal check 2 (Reviewer xe7f): are the headline per-speech results driven
by prolific speakers? Recompute every headline proportion with one vote per
speaker and compare against the paper's per-speech numbers.

Step 0 verifies the per-speech baseline against the exact numbers in the
resubmitted paper (tex_snippets.md); the script aborts if any of them fail
to reproduce, so the speaker-level numbers can only ever be read alongside
a confirmed baseline.

Speaker-level aggregation:
  - unit = (speaker_key, gender); speeches with no parseable speaker are
    excluded from speaker-level tables (count reported).
  - "speaker-mean" = compute the proportion within each speaker, then average
    across speakers unweighted (one speaker, one vote).
  - "speaker-majority" = classify each speaker by their majority stance, then
    tabulate speakers.

Reads:  experiments/may24_rewrite/v8_corpus_classifications.csv (via 00_shared)
Writes: results_speaker_aggregation.json
        results_speaker_aggregation.md
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("shared", ROOT / "00_shared.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)

JSON_OUT = ROOT / "results_speaker_aggregation.json"
MD_OUT = ROOT / "results_speaker_aggregation.md"

# Every constant below is copied from experiments/may24_rewrite/tex_snippets.md
# (the source of the numbers in the resubmitted paper).
PAPER = {
    "n_for": 2167,
    "n_against": 570,
    "n_both": 205,
    "n_sexist": 886,
    "n_hostile": 392,
    "n_benevolent": 706,
    "n_hostile_and_benevolent": 212,
    "female_support_pct": 93,
    "male_support_pct": 70,
    "any_sexism_pct": {"for": 21, "against": 54, "both": 57},
    "sexist_mix_pct": {
        "for": {"hostile_only": 11, "benevolent_only": 81, "both": 8},
        "against": {"hostile_only": 37, "benevolent_only": 19, "both": 44},
        "both": {"hostile_only": 14, "benevolent_only": 52, "both": 34},
    },
}

STANCES = ["for", "against", "both"]


def verify_baseline(rel: pd.DataFrame) -> list[str]:
    """Reproduce the paper's per-speech numbers; raise on any mismatch."""
    checks = []

    def check(name, got, want):
        ok = got == want
        checks.append(f"{'OK ' if ok else 'FAIL'} {name}: paper={want} computed={got}")
        if not ok:
            raise AssertionError(
                f"Baseline mismatch on {name}: paper={want}, computed={got}. "
                "Refusing to produce speaker-level numbers on top of an "
                "unverified baseline."
            )

    counts = rel["llm_stance"].value_counts()
    check("n_for", int(counts.get("for", 0)), PAPER["n_for"])
    check("n_against", int(counts.get("against", 0)), PAPER["n_against"])
    check("n_both", int(counts.get("both", 0)), PAPER["n_both"])

    check("n_hostile", int(rel["llm_hostile"].sum()), PAPER["n_hostile"])
    check("n_benevolent", int(rel["llm_benevolent"].sum()), PAPER["n_benevolent"])
    check("n_sexist", int(rel["any_sexism"].sum()), PAPER["n_sexist"])
    check(
        "n_hostile_and_benevolent",
        int((rel["llm_hostile"] & rel["llm_benevolent"]).sum()),
        PAPER["n_hostile_and_benevolent"],
    )

    for g, want_key in (("F", "female_support_pct"), ("M", "male_support_pct")):
        sub = rel[rel["gender"] == g]
        got = round(100 * (sub["llm_stance"] == "for").mean())
        check(f"support_pct_gender_{g}", got, PAPER[want_key])

    for s in STANCES:
        sub = rel[rel["llm_stance"] == s]
        got = round(100 * sub["any_sexism"].mean())
        check(f"any_sexism_pct_{s}", got, PAPER["any_sexism_pct"][s])
        sx = sub[sub["any_sexism"]]
        mix = {
            "hostile_only": round(100 * (sx["llm_hostile"] & ~sx["llm_benevolent"]).mean()),
            "benevolent_only": round(100 * (~sx["llm_hostile"] & sx["llm_benevolent"]).mean()),
            "both": round(100 * (sx["llm_hostile"] & sx["llm_benevolent"]).mean()),
        }
        for k, v in mix.items():
            check(f"sexist_mix_{s}_{k}", v, PAPER["sexist_mix_pct"][s][k])

    return checks


def speaker_mean(sub: pd.DataFrame, flag_col: str) -> tuple[float, int]:
    """One-vote-per-speaker proportion: per-speaker mean of flag, averaged
    unweighted across speakers. Returns (pct, n_speakers)."""
    per = sub.groupby(["speaker_key", "gender"], dropna=False)[flag_col].mean()
    return round(100 * float(per.mean()), 1), int(len(per))


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    df = shared.load_corpus()
    eras = shared.era_order(df)

    rel = df[~df["is_irrelevant"]].copy()
    rel["any_sexism"] = rel["llm_hostile"] | rel["llm_benevolent"]
    rel["is_for"] = rel["llm_stance"] == "for"

    baseline_checks = verify_baseline(rel)
    print("\n".join(baseline_checks))

    n_no_speaker = int(rel["speaker_key"].isna().sum())
    spk = rel.dropna(subset=["speaker_key"]).copy()

    # --- Speaker concentration: how real is the prolific-speaker concern? ---
    per_speaker_n = spk.groupby(["speaker_key", "gender"], dropna=False).size()
    top10_share = round(
        100 * per_speaker_n.sort_values(ascending=False).head(10).sum() / len(spk), 1
    )
    concentration = {
        "n_relevant_speeches": int(len(rel)),
        "n_speeches_without_speaker": n_no_speaker,
        "n_unique_speakers": int(len(per_speaker_n)),
        "max_speeches_one_speaker": int(per_speaker_n.max()),
        "median_speeches_per_speaker": float(per_speaker_n.median()),
        "top10_speakers_share_pct": top10_share,
        "speakers_with_one_speech": int((per_speaker_n == 1).sum()),
    }

    # --- Stance by gender: per-speech vs one-vote-per-speaker ---
    stance_gender_rows = []
    for g in ("F", "M"):
        sub = spk[spk["gender"] == g]
        per_speech = round(100 * sub["is_for"].mean(), 1)
        sp_mean, n_sp = speaker_mean(sub, "is_for")
        stance_gender_rows.append(
            {
                "gender": g,
                "n_speeches": int(len(sub)),
                "n_speakers": n_sp,
                "for_pct_per_speech": per_speech,
                "for_pct_speaker_mean": sp_mean,
            }
        )

    # Speaker-majority stance and chi-square at the speaker level
    grp = spk.groupby(["speaker_key", "gender"], dropna=False)["is_for"]
    maj = pd.DataFrame({"n": grp.size(), "n_for": grp.sum()}).reset_index()
    maj["majority"] = "tie"
    maj.loc[maj["n_for"] * 2 > maj["n"], "majority"] = "for"
    maj.loc[maj["n_for"] * 2 < maj["n"], "majority"] = "not_for"
    maj_g = maj[maj["gender"].isin(["F", "M"]) & (maj["majority"] != "tie")]
    ct = pd.crosstab(maj_g["gender"], maj_g["majority"])
    chi2, p, dof, _ = chi2_contingency(ct)
    speaker_chi2 = {
        "contingency": {g: ct.loc[g].to_dict() for g in ct.index},
        "n_tie_speakers_excluded": int((maj["majority"] == "tie").sum()),
        "chi2": round(float(chi2), 2),
        "p": float(p),
        "dof": int(dof),
    }
    majority_stance_by_gender = [
        {
            "gender": g,
            "n_speakers": int((maj_g["gender"] == g).sum()),
            "majority_for_pct": round(
                100
                * (maj_g.loc[maj_g["gender"] == g, "majority"] == "for").mean(),
                1,
            ),
        }
        for g in ("F", "M")
    ]

    # Speaker-majority sexism, for completeness. Note: the median speaker has
    # one speech, so majority == mean for most speakers; and among multi-speech
    # speakers sexism is minority behavior, so "majority sexist" measures a
    # stricter quantity than the per-speech rate. Reported alongside
    # "ever sexist" (which instead grows with speech count) to bracket the
    # mean-based estimate.
    majority_sexism_rows = []
    for s in STANCES:
        per = (
            spk[spk["llm_stance"] == s]
            .groupby(["speaker_key", "gender"], dropna=False)["any_sexism"]
            .mean()
        )
        majority_sexism_rows.append(
            {
                "stance": s,
                "n_speakers": int(len(per)),
                "majority_sexist_pct": round(100 * float((per > 0.5).mean()), 1),
                "exact_tie_pct": round(100 * float((per == 0.5).mean()), 1),
                "ever_sexist_pct": round(100 * float((per > 0).mean()), 1),
            }
        )

    # --- Support by era x gender (the convergence claim), speaker-mean ---
    era_gender_rows = []
    for era in eras:
        row = {"era": era}
        for g in ("F", "M"):
            sub = spk[(spk["era"] == era) & (spk["gender"] == g)]
            if len(sub) == 0:
                row[f"{g}_per_speech"] = None
                row[f"{g}_speaker_mean"] = None
                row[f"{g}_n_speakers"] = 0
                continue
            row[f"{g}_per_speech"] = round(100 * sub["is_for"].mean(), 1)
            row[f"{g}_speaker_mean"], row[f"{g}_n_speakers"] = speaker_mean(
                sub, "is_for"
            )
        era_gender_rows.append(row)

    # --- Sexism by stance: per-speech vs one-vote-per-speaker ---
    sexism_rows = []
    for s in STANCES:
        sub = spk[spk["llm_stance"] == s]
        row = {"stance": s, "n_speeches": int(len(sub))}
        for col, label in (
            ("any_sexism", "any"),
            ("llm_hostile", "hostile"),
            ("llm_benevolent", "benevolent"),
        ):
            row[f"{label}_per_speech"] = round(100 * sub[col].mean(), 1)
            row[f"{label}_speaker_mean"], row["n_speakers"] = speaker_mean(sub, col)
        sexism_rows.append(row)

    # --- Mix within sexist speeches (hostile-only / benevolent-only / both) ---
    mix_rows = []
    for s in STANCES:
        sx = spk[(spk["llm_stance"] == s) & spk["any_sexism"]].copy()
        sx["hostile_only"] = sx["llm_hostile"] & ~sx["llm_benevolent"]
        sx["benevolent_only"] = ~sx["llm_hostile"] & sx["llm_benevolent"]
        sx["both_hb"] = sx["llm_hostile"] & sx["llm_benevolent"]
        row = {"stance": s, "n_sexist_speeches": int(len(sx))}
        for col in ("hostile_only", "benevolent_only", "both_hb"):
            row[f"{col}_per_speech"] = round(100 * sx[col].mean(), 1)
            row[f"{col}_speaker_mean"], row["n_speakers"] = speaker_mean(sx, col)
        mix_rows.append(row)

    # --- Hostile share of sexist speeches over time (declining-hostility claim) ---
    hostility_rows = []
    for era in eras:
        sx = spk[(spk["era"] == era) & spk["any_sexism"]]
        if len(sx) == 0:
            continue
        sp_mean, n_sp = speaker_mean(sx, "llm_hostile")
        hostility_rows.append(
            {
                "era": era,
                "n_sexist_speeches": int(len(sx)),
                "n_speakers": n_sp,
                "hostile_share_per_speech": round(100 * sx["llm_hostile"].mean(), 1),
                "hostile_share_speaker_mean": sp_mean,
            }
        )

    results = {
        "source": str(shared.CSV_PATH),
        "baseline_verification": baseline_checks,
        "speaker_concentration": concentration,
        "stance_by_gender": stance_gender_rows,
        "speaker_majority_chi2": speaker_chi2,
        "majority_stance_by_gender": majority_stance_by_gender,
        "majority_sexism_by_stance": majority_sexism_rows,
        "support_by_era_gender": era_gender_rows,
        "sexism_by_stance": sexism_rows,
        "sexist_mix_by_stance": mix_rows,
        "hostile_share_by_era": hostility_rows,
    }
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n")

    md = [
        "# Speaker-level aggregation (rebuttal check for Reviewer xe7f)",
        "",
        f"Source: `{shared.CSV_PATH.name}`. Baseline per-speech numbers verified "
        "against the paper before aggregating (see JSON for the check list).",
        f"Speaker unit = normalized (name, gender); {n_no_speaker} relevant "
        "speeches without a parseable speaker excluded from speaker-level rows.",
        "",
        "## Speaker concentration",
        "",
    ]
    md += [f"- {k}: {v}" for k, v in concentration.items()]
    md += [
        "",
        "## Support (stance = for) by gender: per-speech vs one-vote-per-speaker",
        md_table(stance_gender_rows),
        "",
        "## Speaker-majority stance by gender (chi-square at speaker level)",
        f"- contingency: {speaker_chi2['contingency']}",
        f"- chi2 = {speaker_chi2['chi2']}, dof = {speaker_chi2['dof']}, "
        f"p = {speaker_chi2['p']:.3g} "
        f"({speaker_chi2['n_tie_speakers_excluded']} tied speakers excluded)",
        md_table(majority_stance_by_gender),
        "",
        "## Speaker-majority sexism by stance (see script comment on interpretation)",
        md_table(majority_sexism_rows),
        "",
        "## Support by era and gender",
        md_table(era_gender_rows),
        "",
        "## Sexism rates by stance",
        md_table(sexism_rows),
        "",
        "## Mix within sexist speeches",
        md_table(mix_rows),
        "",
        "## Hostile share of sexist speeches by era",
        md_table(hostility_rows),
        "",
    ]
    MD_OUT.write_text("\n".join(md))

    print()
    print(pd.DataFrame(stance_gender_rows).to_string(index=False))
    print()
    print(pd.DataFrame(sexism_rows).to_string(index=False))
    print(f"\nWrote {JSON_OUT.name}, {MD_OUT.name}")


if __name__ == "__main__":
    main()
