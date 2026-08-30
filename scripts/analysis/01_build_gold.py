"""
Build the gold label set for the 300-speech V8 validation sample.

For each speech in the validation_sample.parquet (n=300):
- If Omar and Mandira agreed on every field (stance + hostile subs +
  benevolent subs), the gold label IS that agreement.
- If they disagreed on anything, the gold label is the consensus from
  consensus.jsonl produced by the resolution app.

Output: experiments/may24_rewrite/gold.jsonl, one record per speech with
the same schema as omar.jsonl / mandira.jsonl, plus a `provenance` field
("agreement" or "consensus") for transparency.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V8_ROOT = ROOT.parent / "20260520_v8_500_validation"
ANN = V8_ROOT / "annotations"
OUT = ROOT / "gold.jsonl"
SAMPLE = V8_ROOT / "validation_sample.parquet"

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


def load_jsonl(path):
    return {json.loads(l)["speech_id"]: json.loads(l)
            for l in open(path) if l.strip()}


def fields_match(o, m):
    return (o["stance"] == m["stance"]
            and set(o["hostile_subcategories"]) == set(m["hostile_subcategories"])
            and set(o["benevolent_subcategories"]) == set(m["benevolent_subcategories"]))


def derive(rec):
    rec["hostile"] = bool(rec.get("hostile_subcategories"))
    rec["benevolent"] = bool(rec.get("benevolent_subcategories"))
    rec["sexist"] = rec["hostile"] or rec["benevolent"]
    return rec


def main():
    import pandas as pd
    sample = pd.read_parquet(SAMPLE)
    omar = load_jsonl(ANN / "omar.jsonl")
    mandira = load_jsonl(ANN / "mandira.jsonl")
    consensus = load_jsonl(ANN / "consensus.jsonl")

    n_total = len(sample)
    n_agree = n_consensus = n_missing = 0
    gold = []

    for _, row in sample.iterrows():
        sid = row["speech_id"]
        o = omar.get(sid)
        m = mandira.get(sid)
        if o is None or m is None:
            n_missing += 1
            continue
        if fields_match(o, m):
            rec = {
                "speech_id": sid,
                "sample_idx": int(row["sample_idx"]),
                "stance": o["stance"],
                "hostile_subcategories": sorted(o["hostile_subcategories"]),
                "benevolent_subcategories": sorted(o["benevolent_subcategories"]),
                "provenance": "agreement",
            }
            n_agree += 1
        else:
            c = consensus.get(sid)
            if c is None:
                n_missing += 1
                print(f"  WARNING: disagreement with no consensus for {sid}")
                continue
            rec = {
                "speech_id": sid,
                "sample_idx": int(row["sample_idx"]),
                "stance": c["stance"],
                "hostile_subcategories": sorted(c["hostile_subcategories"]),
                "benevolent_subcategories": sorted(c["benevolent_subcategories"]),
                "provenance": "consensus",
            }
            n_consensus += 1
        derive(rec)
        gold.append(rec)

    gold.sort(key=lambda r: r["sample_idx"])
    with open(OUT, "w") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total sample: {n_total}")
    print(f"  Agreement (both annotators identical):  {n_agree}")
    print(f"  Consensus (resolved disagreement):      {n_consensus}")
    print(f"  Missing (no annotation or no consensus): {n_missing}")
    print(f"  Wrote {len(gold)} gold records -> {OUT}")
    print()
    print("Gold label distributions:")
    print(f"  Stance:    {dict(Counter(r['stance'] for r in gold))}")
    print(f"  Hostile:   {sum(1 for r in gold if r['hostile'])}/{len(gold)}")
    print(f"  Benevolent:{sum(1 for r in gold if r['benevolent'])}/{len(gold)}")
    print(f"  Sexist:    {sum(1 for r in gold if r['sexist'])}/{len(gold)}")
    print()
    print("Hostile subs:")
    for s in HOSTILE_SUBS:
        print(f"  {s:45s} {sum(1 for r in gold if s in r['hostile_subcategories'])}")
    print("Benevolent subs:")
    for s in BENEVOLENT_SUBS:
        print(f"  {s:45s} {sum(1 for r in gold if s in r['benevolent_subcategories'])}")


if __name__ == "__main__":
    main()
