"""
Stance baselines on the 300-speech validation set. Produces the non-LLM rows
of the baseline comparison table (Table 4).

Three baselines:
  - Majority class
  - TF-IDF + Logistic Regression (5-fold stratified CV)
  - DeBERTa-v3 zero-shot NLI

Reads:  data/annotations/gold.jsonl
        data/womens_rights/corpus_with_context.parquet (run
        scripts/download_data.py first)
Writes: data/results/baselines_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (cohen_kappa_score, classification_report,
                              precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "data" / "annotations" / "gold.jsonl"
CORPUS = REPO / "data" / "womens_rights" / "corpus_with_context.parquet"
OUT = REPO / "data" / "results" / "baselines_results.json"

STANCE = ["for", "against", "both", "irrelevant"]


def load_gold_with_text():
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    corpus = pd.read_parquet(CORPUS)[["speech_id", "target_text"]].set_index("speech_id")
    rows = []
    for g in gold:
        sid = g["speech_id"]
        if sid not in corpus.index:
            continue
        rows.append({
            "speech_id": sid,
            "text": corpus.loc[sid, "target_text"],
            "stance": g["stance"],
        })
    return pd.DataFrame(rows)


def report(name, y_true, y_pred):
    n = len(y_true)
    agree = sum(int(a == b) for a, b in zip(y_true, y_pred))
    kappa = cohen_kappa_score(y_true, y_pred, labels=STANCE)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=STANCE, average="macro", zero_division=0
    )
    out = {
        "name": name,
        "n": n,
        "agree": agree,
        "agree_pct": round(agree / n, 4),
        "kappa": round(float(kappa), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {STANCE[i]: {"P": round(float(p[i]), 4),
                                    "R": round(float(r[i]), 4),
                                    "F1": round(float(f1[i]), 4),
                                    "n_gold": int(sup[i])}
                       for i in range(len(STANCE))},
    }
    print(f"\n--- {name} ---")
    print(f"  n={n}  agree={agree}/{n} ({agree/n:.1%})  kappa={kappa:.3f}  macro_f1={macro_f1:.3f}")
    for i, c in enumerate(STANCE):
        print(f"    {c:11s}  P={p[i]:.3f}  R={r[i]:.3f}  F1={f1[i]:.3f}  n_gold={sup[i]}")
    return out


def majority_baseline(df):
    majority = df["stance"].value_counts().idxmax()
    y_pred = [majority] * len(df)
    return report("Majority class", df["stance"].tolist(), y_pred)


def tfidf_baseline(df, n_splits=5, seed=42):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                    sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=seed)),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(pipe, df["text"].tolist(), df["stance"].tolist(),
                                 cv=skf)
    return report(f"TF-IDF + LogReg ({n_splits}-fold CV)",
                  df["stance"].tolist(), y_pred.tolist())


def deberta_zero_shot(df):
    """DeBERTa-v3-large zero-shot NLI on each speech."""
    from transformers import pipeline
    print("\nLoading DeBERTa-v3-large-mnli...")
    clf = pipeline("zero-shot-classification",
                    model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                    device=-1)  # CPU
    hypotheses = {
        "for": "This speech supports women's political rights or representation.",
        "against": "This speech opposes women's political rights or representation.",
        "both": "This speech contains both support and opposition for women's political rights.",
        "irrelevant": "This speech is not about women's political rights or representation.",
    }
    labels = list(hypotheses.values())
    label_to_stance = {v: k for k, v in hypotheses.items()}
    y_pred = []
    for i, txt in enumerate(df["text"].tolist()):
        # Truncate text since DeBERTa context limit is ~512 tokens
        snippet = (txt or "")[:4000]
        result = clf(snippet, candidate_labels=labels, multi_label=False)
        top_label = result["labels"][0]
        y_pred.append(label_to_stance[top_label])
        if (i + 1) % 50 == 0:
            print(f"  [deberta] {i+1}/{len(df)}")
    return report("DeBERTa-v3 zero-shot (NLI)", df["stance"].tolist(), y_pred)


def main():
    df = load_gold_with_text()
    print(f"Loaded {len(df)} validation speeches")
    print(f"Stance distribution: {df['stance'].value_counts().to_dict()}")

    results = {}
    results["majority"] = majority_baseline(df)
    results["tfidf_logreg"] = tfidf_baseline(df)
    results["deberta_zero_shot"] = deberta_zero_shot(df)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
