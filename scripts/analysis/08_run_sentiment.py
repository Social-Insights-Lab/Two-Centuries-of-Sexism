"""
Sentiment confound analysis on V8 sexist speeches. Uses DistilBERT-SST-2 to
match the original paper's methodology, reports % negative-sentiment broken
down by stance and sexism type. Generates the replacement for the appendix
sentiment table (sec:sentiment).

Output:
  experiments/may24_rewrite/sentiment_results.json
"""
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parent
STANCE_JSONL = ROOT / "v8_stance.jsonl"
SEXISM_JSONL = ROOT / "v8_sexism.jsonl"
CORPUS = ROOT / "corpus_with_context.parquet"
OUT = ROOT / "sentiment_results.json"

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_CHARS = 4000  # SST-2 model has a small context window; truncate


def load_v8():
    stance = {}
    for line in open(STANCE_JSONL):
        r = json.loads(line)
        if r.get("parsed"):
            stance[r["speech_id"]] = r["parsed"]["stance"]
    sexism = {}
    for line in open(SEXISM_JSONL):
        r = json.loads(line)
        if r.get("parsed"):
            sexism[r["speech_id"]] = r["parsed"]
    return stance, sexism


def main():
    stance, sexism = load_v8()
    corpus = pd.read_parquet(CORPUS)[["speech_id", "target_text"]].set_index("speech_id")

    sexist_ids = [sid for sid, s in sexism.items()
                  if s.get("hostile") or s.get("benevolent")]
    print(f"Total sexist speeches: {len(sexist_ids)}")

    from transformers import pipeline
    print(f"Loading {MODEL}...")
    clf = pipeline("sentiment-analysis", model=MODEL, device=-1)

    rows = []
    for i, sid in enumerate(sexist_ids):
        if sid not in corpus.index:
            continue
        text = (corpus.loc[sid, "target_text"] or "")[:MAX_CHARS]
        if not text.strip():
            continue
        # DistilBERT-SST-2 has a 512-token limit; truncate via tokenizer
        result = clf(text, truncation=True, max_length=512)[0]
        rows.append({
            "speech_id": sid,
            "stance": stance.get(sid),
            "hostile": bool(sexism[sid].get("hostile")),
            "benevolent": bool(sexism[sid].get("benevolent")),
            "sentiment_label": result["label"],
            "sentiment_score": result["score"],
        })
        if (i + 1) % 100 == 0:
            print(f"  [sentiment] {i+1}/{len(sexist_ids)}")

    print(f"Scored {len(rows)} sexist speeches")

    # Compute table: % NEGATIVE by stance x sexism_type
    by_stance_type = defaultdict(lambda: {"total": 0, "negative": 0})
    for r in rows:
        st = r["stance"]
        is_neg = r["sentiment_label"] == "NEGATIVE"
        if r["hostile"]:
            key = (st, "hostile")
            by_stance_type[key]["total"] += 1
            by_stance_type[key]["negative"] += int(is_neg)
        if r["benevolent"]:
            key = (st, "benevolent")
            by_stance_type[key]["total"] += 1
            by_stance_type[key]["negative"] += int(is_neg)

    print("\n=== %% NEGATIVE sentiment by stance x sexism type ===")
    print(f"{'Stance':<12}{'Type':<14}{'n':>6}{'% neg':>10}")
    print("-" * 45)
    table = {}
    for st in ["for", "against", "both"]:
        for ty in ["hostile", "benevolent"]:
            d = by_stance_type.get((st, ty))
            if not d or d["total"] == 0:
                continue
            pct = d["negative"] / d["total"]
            print(f"{st:<12}{ty:<14}{d['total']:>6}{pct*100:>9.0f}%")
            table.setdefault(st, {})[ty] = {
                "n": d["total"],
                "negative": d["negative"],
                "pct_negative": round(pct, 4),
            }

    # Overall ("All")
    print("-" * 45)
    for ty in ["hostile", "benevolent"]:
        tot = sum(d["total"] for (st, t), d in by_stance_type.items() if t == ty)
        neg = sum(d["negative"] for (st, t), d in by_stance_type.items() if t == ty)
        if tot == 0:
            continue
        pct = neg / tot
        print(f"{'All':<12}{ty:<14}{tot:>6}{pct*100:>9.0f}%")
        table.setdefault("All", {})[ty] = {
            "n": tot, "negative": neg, "pct_negative": round(pct, 4),
        }

    out = {
        "model": MODEL,
        "n_sexist_total": len(rows),
        "table": {st: {ty: v for ty, v in d.items()} for st, d in table.items()},
        "per_speech": rows,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
