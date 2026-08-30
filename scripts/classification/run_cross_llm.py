"""
Run the stance + sexism prompts through GPT-5, Gemini 2.5 Flash, DeepSeek V3
on the 300-speech validation sample. Uses OpenRouter so one key reaches all
three model families.

Output JSONL files mirror the Anthropic run format. This generates the data
for the cross-model agreement tables (Tables 5 and 8).

Reads:  prompts/stance.md, prompts/sexism.md
        data/validation/validation_sample.parquet
        data/womens_rights/corpus_with_context.parquet (run
        scripts/download_data.py first)
Writes: data/results/cross_llm/<model_slug>_{stance,sexism}.jsonl

Usage:
  python scripts/classification/run_cross_llm.py --model gpt5
  python scripts/classification/run_cross_llm.py --model all --limit 5

Environment:
  OPENROUTER_HANSARD_API_KEY
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "prompts"
CORPUS = REPO / "data" / "womens_rights" / "corpus_with_context.parquet"
VAL = REPO / "data" / "validation" / "validation_sample.parquet"
OUT_DIR = REPO / "data" / "results" / "cross_llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENROUTER_HANSARD_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# Match the four models the paper references (Sonnet is the primary, already done)
MODELS = {
    "gpt5":     {"slug": "openai/gpt-5",                 "label": "GPT-5"},
    "gemini":   {"slug": "google/gemini-2.5-flash",      "label": "Gemini 2.5 Flash"},
    "deepseek": {"slug": "deepseek/deepseek-chat-v3.1",  "label": "DeepSeek V3"},
}

STANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_stance",
        "description": "Record the stance classification.",
        "parameters": {
            "type": "object",
            "properties": {
                "stance": {"type": "string",
                           "enum": ["for", "against", "both", "irrelevant"]},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["stance", "rationale", "confidence"],
        },
    },
}

HOSTILE_SUBS = ["dominative_paternalism", "competitive_gender_differentiation",
                "heterosexual_hostility"]
BENEVOLENT_SUBS = ["protective_paternalism", "complementary_gender_differentiation",
                   "heterosexual_intimacy"]

SEXISM_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_sexism",
        "description": (
            "Record the multi-label AST sexism classification. If hostile is "
            "true, hostile_subcategories must be non-empty; same for benevolent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hostile": {"type": "boolean"},
                "hostile_subcategories": {
                    "type": "array",
                    "items": {"type": "string", "enum": HOSTILE_SUBS},
                },
                "hostile_quote": {"type": "string"},
                "benevolent": {"type": "boolean"},
                "benevolent_subcategories": {
                    "type": "array",
                    "items": {"type": "string", "enum": BENEVOLENT_SUBS},
                },
                "benevolent_quote": {"type": "string"},
            },
            "required": ["hostile", "hostile_subcategories", "hostile_quote",
                         "benevolent", "benevolent_subcategories", "benevolent_quote"],
        },
    },
}


def extract_prompt_body(md_path: Path) -> str:
    m = re.search(r"```\s*\n(.*?)\n```", md_path.read_text(), re.DOTALL)
    if not m:
        raise SystemExit(f"No fenced code block in {md_path}")
    return m.group(1)


def split_system_user(body: str) -> tuple[str, str]:
    parts = re.split(r"^USER\s*$", body, maxsplit=1, flags=re.MULTILINE)
    system = re.sub(r"^SYSTEM\s*\n", "", parts[0]).strip()
    user_tpl = parts[1].strip()
    return system, user_tpl


def load_validation_speeches() -> list:
    val = pd.read_parquet(VAL)[["speech_id"]]
    corpus = pd.read_parquet(CORPUS)[["speech_id", "target_text", "context_text"]]
    df = val.merge(corpus, on="speech_id", how="left")
    missing = df["target_text"].isna().sum()
    if missing:
        raise SystemExit(f"FATAL: {missing} validation speeches missing text")
    return df.to_dict("records")


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        try:
            done.add(json.loads(line)["speech_id"])
        except Exception:
            continue
    return done


def append_line(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def call_with_tool(client, model_slug, system, user_text, tool,
                          max_tokens=1024, max_retries=4):
    # Reasoning models (GPT-5, o3, etc) need much more headroom because
    # reasoning tokens count against max_tokens. Bump silently for those.
    if "gpt-5" in model_slug or "o1" in model_slug or "o3" in model_slug:
        max_tokens = max(max_tokens, 16000)
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_slug,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                tools=[tool],
                tool_choice={"type": "function",
                              "function": {"name": tool["function"]["name"]}},
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                # Some models (Gemini) sometimes return content instead. Try parse.
                if msg.content:
                    try:
                        return {"ok": True,
                                "parsed": json.loads(msg.content),
                                "usage": dict(resp.usage) if resp.usage else {}}
                    except Exception:
                        pass
                return {"ok": False, "error": "no_tool_call_or_json",
                        "content": (msg.content or "")[:300]}
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments)
            return {"ok": True, "parsed": args,
                    "usage": {"prompt_tokens": resp.usage.prompt_tokens,
                              "completion_tokens": resp.usage.completion_tokens}
                              if resp.usage else {}}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(2 ** attempt)
    return {"ok": False, "error": "max_retries"}


async def run_one_model(client, model_key, speeches, concurrency=4,
                         skip_stance=False, skip_sexism=False, dry_run=False):
    info = MODELS[model_key]
    slug = info["slug"]
    print(f"\n=== {info['label']} ({slug}) ===")

    stance_out = OUT_DIR / f"{model_key}_stance.jsonl"
    sexism_out = OUT_DIR / f"{model_key}_sexism.jsonl"

    # Pass 1 -- stance
    if not skip_stance:
        sys_p, user_tpl = split_system_user(
            extract_prompt_body(PROMPTS / "stance.md")
        )
        done = load_done(stance_out)
        todo = [s for s in speeches if s["speech_id"] not in done]
        print(f"[stance] total={len(speeches)} done={len(done)} remaining={len(todo)}")
        if not dry_run and todo:
            sem = asyncio.Semaphore(concurrency)
            async def one_stance(s):
                async with sem:
                    user_text = (user_tpl
                                 .replace("{target_text}", s["target_text"] or "")
                                 .replace("{context_text}", s["context_text"] or "[No context available]"))
                    result = await call_with_tool(client, slug, sys_p, user_text, STANCE_TOOL)
                    rec = {"speech_id": s["speech_id"], "model": model_key, "pass": "stance"}
                    rec.update(result)
                    append_line(stance_out, rec)
            await asyncio.gather(*(one_stance(s) for s in todo))
        print(f"[stance] done -> {stance_out}")

    # Pass 2 -- sexism (only relevant)
    if not skip_sexism:
        stance_by_id = {}
        for line in open(stance_out):
            r = json.loads(line)
            if r.get("ok"):
                stance_by_id[r["speech_id"]] = (r.get("parsed") or {}).get("stance")
        relevant = [{**s, "_stance": stance_by_id.get(s["speech_id"])}
                    for s in speeches
                    if stance_by_id.get(s["speech_id"]) in ("for", "against", "both")]

        sys_p, user_tpl = split_system_user(
            extract_prompt_body(PROMPTS / "sexism.md")
        )
        done = load_done(sexism_out)
        todo = [s for s in relevant if s["speech_id"] not in done]
        print(f"[sexism] relevant={len(relevant)} done={len(done)} remaining={len(todo)}")
        if not dry_run and todo:
            sem = asyncio.Semaphore(concurrency)
            async def one_sex(s):
                async with sem:
                    user_text = (user_tpl
                                 .replace("{stance}", s["_stance"])
                                 .replace("{target_text}", s["target_text"] or "")
                                 .replace("{context_text}", s["context_text"] or "[No context available]"))
                    result = await call_with_tool(client, slug, sys_p, user_text, SEXISM_TOOL)
                    rec = {"speech_id": s["speech_id"], "model": model_key,
                           "pass": "sexism", "stance": s["_stance"]}
                    rec.update(result)
                    append_line(sexism_out, rec)
            await asyncio.gather(*(one_sex(s) for s in todo))
        print(f"[sexism] done -> {sexism_out}")


async def main_async(args):
    if not API_KEY:
        print("ERROR: OPENROUTER_HANSARD_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    speeches = load_validation_speeches()
    if args.limit:
        speeches = speeches[: args.limit]
    print(f"Loaded {len(speeches)} validation speeches")

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for m in targets:
        await run_one_model(client, m, speeches, concurrency=args.concurrency,
                             skip_stance=args.skip_stance,
                             skip_sexism=args.skip_sexism, dry_run=args.dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--skip-stance", action="store_true")
    ap.add_argument("--skip-sexism", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
