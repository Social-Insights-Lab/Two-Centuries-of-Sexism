"""Validation metrics for the LLM classifiers against the gold labels.

Covers:
  - Claude per-class stance P/R/F1 and kappa on the 300-speech validation
    set (Table 3, and the LLM row of Table 4)
  - Pairwise stance agreement between human consensus and all four models
    (Table 5)
  - Per-model stance / hostile / benevolent kappa vs human labels (Table 8)
  - Claude sexism flag kappa, precision, and recall vs human labels
    (sexism validation table)

Human-human agreement (kappa 0.644) is computed by
scripts/annotation/agreement.py from the two annotators' independent labels.

Reads:  data/annotations/gold.jsonl
        data/classifications/stance_llm.jsonl, sexism_llm.jsonl
        data/results/cross_llm/<model>_{stance,sexism}.jsonl
Writes: data/results/validation_metrics.json
"""
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data" / "annotations" / "gold.jsonl"
CLS = REPO / "data" / "classifications"
CROSS = REPO / "data" / "results" / "cross_llm"
OUT = REPO / "data" / "results" / "validation_metrics.json"

STANCES = ["for", "against", "both", "irrelevant"]
MODELS = {
    "claude": (CLS / "stance_llm.jsonl", CLS / "sexism_llm.jsonl"),
    "gpt5": (CROSS / "gpt5_stance.jsonl", CROSS / "gpt5_sexism.jsonl"),
    "gemini": (CROSS / "gemini_stance.jsonl", CROSS / "gemini_sexism.jsonl"),
    "deepseek": (CROSS / "deepseek_stance.jsonl", CROSS / "deepseek_sexism.jsonl"),
}


def load_parsed(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("parsed"):
            out[r["speech_id"]] = r["parsed"]
    return out


def stance_report(y_true, y_pred):
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=STANCES))
    agree = sum(a == b for a, b in zip(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCES, average=None, zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCES, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCES, average="weighted", zero_division=0)
    return {
        "n": len(y_true),
        "agree_pct": round(agree / len(y_true), 3),
        "kappa": round(kappa, 3),
        "per_class": {
            STANCES[i]: {"P": round(float(p[i]), 2), "R": round(float(r[i]), 2),
                         "F1": round(float(f1[i]), 2), "n": int(support[i])}
            for i in range(len(STANCES))
        },
        "macro": {"P": round(float(mp), 2), "R": round(float(mr), 2),
                  "F1": round(float(mf1), 2)},
        "weighted": {"P": round(float(wp), 2), "R": round(float(wr), 2),
                     "F1": round(float(wf1), 2)},
    }


def binary_report(y_true, y_pred):
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=[False, True]))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[True], average="binary", zero_division=0)
    return {"kappa": round(kappa, 3), "precision": round(float(p), 2),
            "recall": round(float(r), 2), "f1": round(float(f1), 2),
            "n_pos_gold": int(sum(y_true)), "n_pos_pred": int(sum(y_pred))}


def main():
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    gold_ids = [g["speech_id"] for g in gold]
    gold_stance = {g["speech_id"]: g["stance"] for g in gold}
    gold_hostile = {g["speech_id"]: bool(g["hostile"]) for g in gold}
    gold_benevolent = {g["speech_id"]: bool(g["benevolent"]) for g in gold}

    results = {"n_validation": len(gold)}
    model_stances = {}

    for name, (stance_path, sexism_path) in MODELS.items():
        stance = load_parsed(stance_path)
        sexism = load_parsed(sexism_path)
        missing = [s for s in gold_ids if s not in stance]
        if missing:
            print(f"WARNING: {name} missing stance for {len(missing)} speeches")
        ids = [s for s in gold_ids if s in stance]
        y_true = [gold_stance[s] for s in ids]
        y_pred = [stance[s]["stance"] for s in ids]
        model_stances[name] = {s: stance[s]["stance"] for s in ids}

        # Speeches the model deemed irrelevant were not run through the
        # sexism pass; they carry no sexism flags.
        h_pred = [bool(sexism.get(s, {}).get("hostile")) for s in ids]
        b_pred = [bool(sexism.get(s, {}).get("benevolent")) for s in ids]
        h_true = [gold_hostile[s] for s in ids]
        b_true = [gold_benevolent[s] for s in ids]
        sx_true = [h or b for h, b in zip(h_true, b_true)]
        sx_pred = [h or b for h, b in zip(h_pred, b_pred)]

        results[name] = {
            "stance": stance_report(y_true, y_pred),
            "hostile": binary_report(h_true, h_pred),
            "benevolent": binary_report(b_true, b_pred),
            "any_sexism": binary_report(sx_true, sx_pred),
        }

    # Pairwise stance kappa: human consensus + all models (Table 5)
    parties = {"human": gold_stance, **model_stances}
    pairwise = {}
    names = list(parties)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = [s for s in gold_ids if s in parties[a] and s in parties[b]]
            k = cohen_kappa_score([parties[a][s] for s in common],
                                  [parties[b][s] for s in common],
                                  labels=STANCES)
            pairwise[f"{a}-{b}"] = {"n": len(common), "kappa": round(float(k), 3)}
    results["pairwise_stance_kappa"] = pairwise

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved to {OUT}\n")

    print("=== Stance vs human labels ===")
    for name in MODELS:
        s = results[name]["stance"]
        print(f"{name:<9} agree={s['agree_pct']:.1%}  kappa={s['kappa']:.3f}")
    c = results["claude"]["stance"]["per_class"]
    print("\n=== Claude per-class stance (Table 3) ===")
    for cls, m in c.items():
        print(f"  {cls:<11} P={m['P']:.2f}  R={m['R']:.2f}  F1={m['F1']:.2f}  n={m['n']}")

    print("\n=== Sexism vs human labels (Table 8 columns 2-3) ===")
    for name in MODELS:
        h, b = results[name]["hostile"], results[name]["benevolent"]
        print(f"{name:<9} hostile kappa={h['kappa']:.3f}  benevolent kappa={b['kappa']:.3f}")

    cl = results["claude"]
    print("\n=== Claude sexism detail ===")
    for k in ("hostile", "benevolent", "any_sexism"):
        m = cl[k]
        print(f"  {k:<11} kappa={m['kappa']:.3f}  P={m['precision']:.2f}  "
              f"R={m['recall']:.2f}  gold+={m['n_pos_gold']}  pred+={m['n_pos_pred']}")

    print("\n=== Pairwise stance kappa (Table 5) ===")
    for pair, m in results["pairwise_stance_kappa"].items():
        print(f"  {pair:<18} kappa={m['kappa']:.3f}  (n={m['n']})")


if __name__ == "__main__":
    main()
