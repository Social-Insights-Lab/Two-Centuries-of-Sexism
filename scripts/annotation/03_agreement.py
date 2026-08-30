"""
Inter-annotator agreement for the V8 500-speech validation round.

Reports:
- Stance: percent agreement + Cohen's kappa (4-class)
- Hostile: percent agreement + Cohen's kappa (binary)
- Benevolent: percent agreement + Cohen's kappa (binary)
- Derived sexist (hostile OR benevolent): percent agreement + kappa
- Per-subcategory agreement (multi-label Jaccard + binary kappa per sub)
- Disagreement breakdown for the resolution pass

All metrics computed over the intersection of speech_ids that both
annotators labelled. Outputs JSON for reproducibility and a human-readable
TXT summary.

Usage: python 03_agreement.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).parent
ANN_DIR = ROOT / "annotations"
OMAR_PATH = ANN_DIR / "omar.jsonl"
MANDIRA_PATH = ANN_DIR / "mandira.jsonl"
OUT_JSON = ROOT / "agreement.json"
OUT_TXT = ROOT / "agreement.txt"

HOSTILE_SUBS = [
    "dominative_paternalism",
    "competitive_gender_differentiation",
    "heterosexual_hostility",
]
BENEVOLENT_SUBS = [
    "protective_paternalism",
    "complementary_gender_differentiation",
    "heterosexual_intimacy",
]
STANCE_OPTIONS = ["for", "against", "both", "irrelevant"]


def load(path):
    with open(path) as f:
        return {json.loads(line)["speech_id"]: json.loads(line)
                for line in f if line.strip()}


def agreement(values_o, values_m, label_name, labels=None):
    n = len(values_o)
    matches = sum(1 for a, b in zip(values_o, values_m) if a == b)
    pct = matches / n
    kappa = cohen_kappa_score(values_o, values_m, labels=labels)
    return {
        "n": n,
        "agree": matches,
        "percent": round(pct, 3),
        "kappa": round(float(kappa), 3),
    }


def cm_to_dict(cm, labels):
    return {lo: {lm: int(cm[i, j]) for j, lm in enumerate(labels)}
            for i, lo in enumerate(labels)}


def main():
    omar = load(OMAR_PATH)
    mandira = load(MANDIRA_PATH)
    common = sorted(set(omar) & set(mandira))
    n = len(common)

    # Aligned label vectors
    o = [omar[s] for s in common]
    m = [mandira[s] for s in common]

    out = {"n": n, "speech_ids_count": n}

    # Stance (4-class)
    o_stance = [r["stance"] for r in o]
    m_stance = [r["stance"] for r in m]
    out["stance"] = agreement(o_stance, m_stance, "stance", labels=STANCE_OPTIONS)
    out["stance"]["confusion"] = cm_to_dict(
        confusion_matrix(o_stance, m_stance, labels=STANCE_OPTIONS), STANCE_OPTIONS
    )

    # Hostile (binary)
    out["hostile_binary"] = agreement(
        [r["hostile"] for r in o], [r["hostile"] for r in m],
        "hostile", labels=[False, True],
    )

    # Benevolent (binary)
    out["benevolent_binary"] = agreement(
        [r["benevolent"] for r in o], [r["benevolent"] for r in m],
        "benevolent", labels=[False, True],
    )

    # Derived sexist
    out["sexist_binary"] = agreement(
        [r["sexist"] for r in o], [r["sexist"] for r in m],
        "sexist", labels=[False, True],
    )

    # Per-subcategory agreement (treat each sub as its own binary flag)
    sub_results = {}
    for sub in HOSTILE_SUBS + BENEVOLENT_SUBS:
        col = "hostile_subcategories" if sub in HOSTILE_SUBS else "benevolent_subcategories"
        o_sub = [sub in r[col] for r in o]
        m_sub = [sub in r[col] for r in m]
        # If neither annotator ever used the sub, kappa is undefined; report None
        if not (any(o_sub) or any(m_sub)):
            sub_results[sub] = {"n": n, "agree": n, "percent": 1.0, "kappa": None,
                                 "note": "never used by either annotator"}
        else:
            sub_results[sub] = agreement(o_sub, m_sub, sub, labels=[False, True])
            sub_results[sub]["omar_count"] = sum(o_sub)
            sub_results[sub]["mandira_count"] = sum(m_sub)
    out["subcategories"] = sub_results

    # Disagreement breakdown
    disagreements = []
    for sid in common:
        ro, rm = omar[sid], mandira[sid]
        if (ro["stance"] != rm["stance"]
            or ro["hostile"] != rm["hostile"]
            or ro["benevolent"] != rm["benevolent"]
            or set(ro["hostile_subcategories"]) != set(rm["hostile_subcategories"])
            or set(ro["benevolent_subcategories"]) != set(rm["benevolent_subcategories"])):
            disagreements.append({
                "speech_id": sid,
                "stance": [ro["stance"], rm["stance"]],
                "hostile": [ro["hostile"], rm["hostile"]],
                "benevolent": [ro["benevolent"], rm["benevolent"]],
                "hostile_subs": [sorted(ro["hostile_subcategories"]),
                                  sorted(rm["hostile_subcategories"])],
                "benevolent_subs": [sorted(ro["benevolent_subcategories"]),
                                     sorted(rm["benevolent_subcategories"])],
                "stance_differs": ro["stance"] != rm["stance"],
                "sexism_differs": (ro["hostile"] != rm["hostile"]
                                    or ro["benevolent"] != rm["benevolent"]
                                    or set(ro["hostile_subcategories"])
                                       != set(rm["hostile_subcategories"])
                                    or set(ro["benevolent_subcategories"])
                                       != set(rm["benevolent_subcategories"])),
            })
    out["disagreements"] = {
        "total": len(disagreements),
        "stance_only": sum(1 for d in disagreements if d["stance_differs"]
                                                       and not d["sexism_differs"]),
        "sexism_only": sum(1 for d in disagreements if d["sexism_differs"]
                                                       and not d["stance_differs"]),
        "both": sum(1 for d in disagreements if d["stance_differs"]
                                                  and d["sexism_differs"]),
        "items": disagreements,
    }

    # Label distributions for transparency
    out["distributions"] = {
        "omar": {
            "stance": dict(Counter(r["stance"] for r in o)),
            "hostile": sum(1 for r in o if r["hostile"]),
            "benevolent": sum(1 for r in o if r["benevolent"]),
            "sexist": sum(1 for r in o if r["sexist"]),
        },
        "mandira": {
            "stance": dict(Counter(r["stance"] for r in m)),
            "hostile": sum(1 for r in m if r["hostile"]),
            "benevolent": sum(1 for r in m if r["benevolent"]),
            "sexist": sum(1 for r in m if r["sexist"]),
        },
    }

    # Write JSON
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # Human-readable summary
    lines = []
    lines.append(f"V8 Inter-Annotator Agreement (n={n})")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Top-level kappa (Omar vs Mandira):")
    for key, label in [("stance", "Stance (4-class)"),
                        ("hostile_binary", "Hostile (binary)"),
                        ("benevolent_binary", "Benevolent (binary)"),
                        ("sexist_binary", "Sexist (derived)")]:
        r = out[key]
        lines.append(f"  {label:25s}  agree={r['agree']}/{n} ({r['percent']:.1%})  "
                     f"kappa={r['kappa']:.3f}")
    lines.append("")
    lines.append("Stance confusion (rows=Omar, cols=Mandira):")
    cm = out["stance"]["confusion"]
    lines.append("           " + "  ".join(f"{s[:6]:>10s}" for s in STANCE_OPTIONS))
    for s_o in STANCE_OPTIONS:
        lines.append(f"  {s_o[:8]:8s}  " +
                     "  ".join(f"{cm[s_o][s_m]:>10d}" for s_m in STANCE_OPTIONS))
    lines.append("")
    lines.append("Per-subcategory agreement:")
    for sub, r in out["subcategories"].items():
        if r["kappa"] is None:
            lines.append(f"  {sub:45s}  unused by both annotators")
        else:
            lines.append(f"  {sub:45s}  "
                         f"O={r['omar_count']:3d}  M={r['mandira_count']:3d}  "
                         f"kappa={r['kappa']:.3f}")
    lines.append("")
    d = out["disagreements"]
    lines.append(f"Disagreements: {d['total']}/{n}  "
                 f"(stance-only: {d['stance_only']}, sexism-only: {d['sexism_only']}, "
                 f"both: {d['both']})")
    lines.append("")
    lines.append("Distributions:")
    for who in ("omar", "mandira"):
        dist = out["distributions"][who]
        lines.append(f"  {who}: stance={dist['stance']}  "
                     f"hostile={dist['hostile']}  benevolent={dist['benevolent']}  "
                     f"sexist={dist['sexist']}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
