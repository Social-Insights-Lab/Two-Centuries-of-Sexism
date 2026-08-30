"""
Compare V7 LLM labels (from outputs/llm_classification/v7_notrunc_results.parquet)
against the V8 gold labels (gold.jsonl) on the 300-speech validation sample.

Projects V7 single-label AST onto V8 multi-label scheme:
  hostile  := (v7.axis_a_label == "hostile")
  benevolent := (v7.axis_a_label == "benevolent")
  sexist := (v7.binary == "sexist") OR hostile OR benevolent

Reports:
  - Stance: kappa + per-class P/R/F1 + confusion matrix (4-class)
  - Hostile, Benevolent, Sexist: kappa + P/R/F1 (binary)
  - Per-era kappa (Us7C temporal concern)
  - Per-instance disagreement summary
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "gold.jsonl"
V7 = Path("outputs/llm_classification/v7_notrunc_results.parquet")
TXT = Path("outputs/llm_classification/suffrage_classified_with_text.parquet")
OUT_JSON = ROOT / "llm_vs_gold_v7.json"
OUT_TXT = ROOT / "llm_vs_gold_v7.txt"

STANCE_OPTIONS = ["for", "against", "both", "irrelevant"]
ERA_BINS = [1800, 1870, 1918, 1928, 1970, 2010]
ERA_LABELS = ["1800-1869", "1870-1917", "1918-1927", "1928-1969", "1970-2005"]


def load_gold():
    return [json.loads(l) for l in open(GOLD) if l.strip()]


def project_v7(v7_row):
    """Project V7's single-label AST onto V8's multi-label scheme."""
    a = v7_row["axis_a_label"]
    hostile = (a == "hostile")
    benevolent = (a == "benevolent")
    binary_sexist = (v7_row["binary"] == "sexist")
    return {
        "stance": v7_row["stance"],
        "hostile": hostile,
        "benevolent": benevolent,
        "sexist": binary_sexist or hostile or benevolent,
    }


def binary_metrics(y_true, y_pred, label):
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=[False, True]))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[True], average="binary", zero_division=0
    )
    n_pos_true = sum(y_true)
    n_pos_pred = sum(y_pred)
    return {
        "label": label,
        "n": len(y_true),
        "n_pos_gold": int(n_pos_true),
        "n_pos_llm": int(n_pos_pred),
        "agree": int(sum(a == b for a, b in zip(y_true, y_pred))),
        "agree_pct": round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 3),
        "kappa": round(kappa, 3),
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
    }


def stance_metrics(y_true, y_pred):
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=STANCE_OPTIONS))
    cm = confusion_matrix(y_true, y_pred, labels=STANCE_OPTIONS).tolist()
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE_OPTIONS, average=None, zero_division=0
    )
    per_class = {
        STANCE_OPTIONS[i]: {
            "precision": round(float(p[i]), 3),
            "recall": round(float(r[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        } for i in range(len(STANCE_OPTIONS))
    }
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE_OPTIONS, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE_OPTIONS, average="weighted", zero_division=0
    )
    return {
        "n": len(y_true),
        "agree": sum(a == b for a, b in zip(y_true, y_pred)),
        "agree_pct": round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 3),
        "kappa": round(kappa, 3),
        "per_class": per_class,
        "macro_f1": round(float(macro_f1), 3),
        "weighted_f1": round(float(weighted_f1), 3),
        "confusion_rows_gold_cols_llm": cm,
    }


def per_era_kappa(rows):
    """Compute stance + sexism kappa within each era."""
    by_era = defaultdict(list)
    for r in rows:
        by_era[r["era"]].append(r)
    out = {}
    for era in ERA_LABELS:
        items = by_era.get(era, [])
        if len(items) < 5:
            out[era] = {"n": len(items), "kappa_stance": None, "kappa_sexist": None}
            continue
        s_t = [r["gold_stance"] for r in items]
        s_p = [r["llm_stance"] for r in items]
        x_t = [r["gold_sexist"] for r in items]
        x_p = [r["llm_sexist"] for r in items]
        try:
            ks = float(cohen_kappa_score(s_t, s_p, labels=STANCE_OPTIONS))
        except Exception:
            ks = None
        try:
            kx = float(cohen_kappa_score(x_t, x_p, labels=[False, True]))
        except Exception:
            kx = None
        out[era] = {
            "n": len(items),
            "kappa_stance": round(ks, 3) if ks is not None else None,
            "kappa_sexist": round(kx, 3) if kx is not None else None,
        }
    return out


def main():
    gold = load_gold()
    v7 = pd.read_parquet(V7).set_index("speech_id")
    txt = pd.read_parquet(TXT)[["speech_id", "year"]]
    txt["era"] = pd.cut(txt["year"], bins=ERA_BINS, labels=ERA_LABELS, right=False)
    txt = txt.set_index("speech_id")

    rows = []
    for g in gold:
        sid = g["speech_id"]
        if sid not in v7.index:
            print(f"  WARNING: no V7 label for {sid}")
            continue
        v = project_v7(v7.loc[sid])
        rows.append({
            "speech_id": sid,
            "gold_stance": g["stance"],
            "llm_stance": v["stance"],
            "gold_hostile": g["hostile"],
            "llm_hostile": v["hostile"],
            "gold_benevolent": g["benevolent"],
            "llm_benevolent": v["benevolent"],
            "gold_sexist": g["sexist"],
            "llm_sexist": v["sexist"],
            "era": str(txt.loc[sid, "era"]),
            "provenance": g["provenance"],
        })

    out = {}
    out["n"] = len(rows)
    out["projection_note"] = ("V7 LLM was single-label AST; "
                               "hostile := (axis_a==hostile), "
                               "benevolent := (axis_a==benevolent). "
                               "V7 cannot assign BOTH simultaneously.")

    out["stance"] = stance_metrics(
        [r["gold_stance"] for r in rows], [r["llm_stance"] for r in rows]
    )
    out["hostile"] = binary_metrics(
        [r["gold_hostile"] for r in rows], [r["llm_hostile"] for r in rows],
        "hostile"
    )
    out["benevolent"] = binary_metrics(
        [r["gold_benevolent"] for r in rows], [r["llm_benevolent"] for r in rows],
        "benevolent"
    )
    out["sexist"] = binary_metrics(
        [r["gold_sexist"] for r in rows], [r["llm_sexist"] for r in rows],
        "sexist"
    )

    # Per-era
    out["per_era"] = per_era_kappa(rows)

    # Per-instance disagreement summary (Us7C ask)
    disagreements = []
    for r in rows:
        if (r["gold_stance"] != r["llm_stance"]
                or r["gold_hostile"] != r["llm_hostile"]
                or r["gold_benevolent"] != r["llm_benevolent"]):
            disagreements.append({
                "speech_id": r["speech_id"],
                "stance": [r["gold_stance"], r["llm_stance"]],
                "hostile": [r["gold_hostile"], r["llm_hostile"]],
                "benevolent": [r["gold_benevolent"], r["llm_benevolent"]],
                "era": r["era"],
                "from_consensus": r["provenance"] == "consensus",
            })
    out["disagreements"] = {
        "total": len(disagreements),
        "from_consensus_subset": sum(1 for d in disagreements if d["from_consensus"]),
        "from_agreement_subset": sum(1 for d in disagreements if not d["from_consensus"]),
        "stance_mismatch": sum(1 for d in disagreements
                                if d["stance"][0] != d["stance"][1]),
        "hostile_mismatch": sum(1 for d in disagreements
                                 if d["hostile"][0] != d["hostile"][1]),
        "benevolent_mismatch": sum(1 for d in disagreements
                                    if d["benevolent"][0] != d["benevolent"][1]),
    }

    out["distributions"] = {
        "gold": {
            "stance": dict(Counter(r["gold_stance"] for r in rows)),
            "hostile": sum(1 for r in rows if r["gold_hostile"]),
            "benevolent": sum(1 for r in rows if r["gold_benevolent"]),
            "sexist": sum(1 for r in rows if r["gold_sexist"]),
        },
        "llm_v7": {
            "stance": dict(Counter(r["llm_stance"] for r in rows)),
            "hostile": sum(1 for r in rows if r["llm_hostile"]),
            "benevolent": sum(1 for r in rows if r["llm_benevolent"]),
            "sexist": sum(1 for r in rows if r["llm_sexist"]),
        },
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))

    # Human-readable summary
    lines = []
    lines.append(f"V7 LLM vs V8 gold on n={out['n']} validation speeches")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Stance (4-class)")
    s = out["stance"]
    lines.append(f"  agreement: {s['agree']}/{s['n']} ({s['agree_pct']:.1%})  kappa={s['kappa']:.3f}")
    lines.append(f"  macro F1: {s['macro_f1']:.3f}   weighted F1: {s['weighted_f1']:.3f}")
    for cls, m in s["per_class"].items():
        lines.append(f"    {cls:11s}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                     f"F1={m['f1']:.3f}  n={m['support']}")
    lines.append("")
    lines.append("Confusion (rows = gold, cols = LLM):")
    cm = s["confusion_rows_gold_cols_llm"]
    lines.append("            " + "  ".join(f"{c[:8]:>10s}" for c in STANCE_OPTIONS))
    for i, c in enumerate(STANCE_OPTIONS):
        lines.append(f"  {c[:8]:8s}  " +
                     "  ".join(f"{cm[i][j]:>10d}" for j in range(len(STANCE_OPTIONS))))
    lines.append("")
    for k in ("hostile", "benevolent", "sexist"):
        b = out[k]
        lines.append(f"{k.capitalize()} (binary)")
        lines.append(f"  gold +{b['n_pos_gold']:<3d} llm +{b['n_pos_llm']:<3d}  "
                     f"agree {b['agree']}/{b['n']} ({b['agree_pct']:.1%})  "
                     f"kappa={b['kappa']:.3f}  "
                     f"P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")
    lines.append("")
    lines.append("Per-era kappa (addresses Us7C temporal validity concern):")
    for era, m in out["per_era"].items():
        ks = m["kappa_stance"]
        kx = m["kappa_sexist"]
        ks_s = f"{ks:.3f}" if ks is not None else "  n/a"
        kx_s = f"{kx:.3f}" if kx is not None else "  n/a"
        lines.append(f"  {era}  n={m['n']:>3d}   stance={ks_s}   sexist={kx_s}")
    lines.append("")
    d = out["disagreements"]
    lines.append(f"Per-instance: {d['total']}/{out['n']} disagreements "
                 f"(stance: {d['stance_mismatch']}, hostile: {d['hostile_mismatch']}, "
                 f"benevolent: {d['benevolent_mismatch']})")
    lines.append(f"  Of disagreements, {d['from_consensus_subset']} are on the "
                 f"106 disagreement-resolved subset, {d['from_agreement_subset']} on "
                 f"the 194 unanimous-agreement subset.")
    lines.append("")
    lines.append("Distributions:")
    lines.append(f"  gold:   {out['distributions']['gold']}")
    lines.append(f"  llm v7: {out['distributions']['llm_v7']}")

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
