"""Noise induction robustness check (Appendix H).

In each of 1,000 iterations, randomly flip 3% of speaker gender labels and
recompute the gender-stance analysis, to test robustness of the gender gap
to potential misgendering in the speaker-matching pipeline.

Reads:  data/classifications/stance_llm.jsonl
        data/womens_rights/speech_classifications.parquet (run
        scripts/download_data.py first)
Writes: data/results/noise_induction_results.json
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_ITERATIONS = 1000
NOISE_RATE = 0.03
REPO = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO / "data" / "results" / "noise_induction_results.json"

random.seed(SEED)
np.random.seed(SEED)

# Load stance labels
stance = {}
for line in open(REPO / "data" / "classifications" / "stance_llm.jsonl"):
    r = json.loads(line)
    if r.get("parsed"):
        stance[r["speech_id"]] = r["parsed"]["stance"]

# Merge with gender from corpus metadata
txt = pd.read_parquet(
    REPO / "data" / "womens_rights" / "speech_classifications.parquet"
)[["speech_id", "gender"]]
df = pd.DataFrame([{"speech_id": s, "stance": st}
                     for s, st in stance.items()]).merge(txt, on="speech_id", how="left")

relevant = df[df["stance"].isin(["for", "against", "both"])].copy()
gendered = relevant[relevant["gender"].isin(["M", "F"])].copy()

print(f"Gendered relevant speeches: {len(gendered)}")
print(f"Noise rate: {NOISE_RATE} ({int(len(gendered) * NOISE_RATE)} flipped per iteration)")
print(f"Iterations: {N_ITERATIONS}\n")

# Baseline
male = gendered[gendered["gender"] == "M"]
female = gendered[gendered["gender"] == "F"]
baseline_male_for = (male["stance"] == "for").sum() / len(male) * 100
baseline_female_for = (female["stance"] == "for").sum() / len(female) * 100
table = pd.DataFrame({
    "for": [(male["stance"] == "for").sum(), (female["stance"] == "for").sum()],
    "not_for": [(male["stance"] != "for").sum(), (female["stance"] != "for").sum()],
}, index=["male", "female"])
baseline_chi2, baseline_p, _, _ = stats.chi2_contingency(table)

print(f"Baseline (no noise):")
print(f"  Male for: {baseline_male_for:.1f}%")
print(f"  Female for: {baseline_female_for:.1f}%")
print(f"  Chi-squared: {baseline_chi2:.2f}, p={baseline_p:.2e}\n")

# Iterations
results = {"male_for_pcts": [], "female_for_pcts": [], "chi2_values": [],
           "p_values": [], "gap_pcts": [], "significant": []}

for i in range(N_ITERATIONS):
    noisy = gendered.copy()
    n_flip = int(len(noisy) * NOISE_RATE)
    flip_indices = np.random.choice(noisy.index, size=n_flip, replace=False)
    for idx in flip_indices:
        if noisy.loc[idx, "gender"] == "M":
            noisy.loc[idx, "gender"] = "F"
        else:
            noisy.loc[idx, "gender"] = "M"

    m = noisy[noisy["gender"] == "M"]
    f = noisy[noisy["gender"] == "F"]
    m_for = (m["stance"] == "for").sum() / len(m) * 100 if len(m) > 0 else 0
    f_for = (f["stance"] == "for").sum() / len(f) * 100 if len(f) > 0 else 0
    t = pd.DataFrame({
        "for": [(m["stance"] == "for").sum(), (f["stance"] == "for").sum()],
        "not_for": [(m["stance"] != "for").sum(), (f["stance"] != "for").sum()],
    }, index=["male", "female"])
    chi2, p, _, _ = stats.chi2_contingency(t)
    results["male_for_pcts"].append(round(float(m_for), 2))
    results["female_for_pcts"].append(round(float(f_for), 2))
    results["chi2_values"].append(round(float(chi2), 2))
    results["p_values"].append(float(p))
    results["gap_pcts"].append(round(float(f_for - m_for), 2))
    results["significant"].append(p < 0.05)

sig_count = sum(results["significant"])
male_for_mean = np.mean(results["male_for_pcts"])
female_for_mean = np.mean(results["female_for_pcts"])
gap_mean = np.mean(results["gap_pcts"])
chi2_mean = np.mean(results["chi2_values"])

print(f"Results across {N_ITERATIONS} noisy iterations:")
print(f"  Male for: {male_for_mean:.1f}% (range: {min(results['male_for_pcts']):.1f}-{max(results['male_for_pcts']):.1f}%)")
print(f"  Female for: {female_for_mean:.1f}% (range: {min(results['female_for_pcts']):.1f}-{max(results['female_for_pcts']):.1f}%)")
print(f"  Gap (F-M): {gap_mean:.1f}pp (range: {min(results['gap_pcts']):.1f}-{max(results['gap_pcts']):.1f}pp)")
print(f"  Chi-squared: {chi2_mean:.2f} (range: {min(results['chi2_values']):.2f}-{max(results['chi2_values']):.2f})")
print(f"  Significant (p<0.05): {sig_count}/{N_ITERATIONS}\n")

output = {
    "seed": SEED, "n_iterations": N_ITERATIONS, "noise_rate": NOISE_RATE,
    "n_flipped_per_iteration": int(len(gendered) * NOISE_RATE),
    "n_gendered_speeches": len(gendered),
    "baseline": {
        "male_for_pct": round(float(baseline_male_for), 1),
        "female_for_pct": round(float(baseline_female_for), 1),
        "chi2": round(float(baseline_chi2), 2),
        "p": float(baseline_p),
    },
    "noisy": {
        "male_for_mean": round(float(male_for_mean), 1),
        "female_for_mean": round(float(female_for_mean), 1),
        "gap_mean": round(float(gap_mean), 1),
        "chi2_mean": round(float(chi2_mean), 2),
        "pct_significant": round(float(sig_count / N_ITERATIONS * 100), 1),
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
