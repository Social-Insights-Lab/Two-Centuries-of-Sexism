"""
Two-pass Claude classification over the keyword-extracted corpus.

Pass 1 (stance):  for / against / both / irrelevant  -- prompts/stance.md
Pass 2 (sexism):  multi-label hostile/benevolent with subcategories
                  -- prompts/sexism.md
                  only run on speeches Pass 1 marked as for / against / both.

Uses the Anthropic SDK with:
  - Structured outputs via strict tool_use (no JSON parse failures)
  - Prompt caching on the (frozen) system prompt + tool definition
  - AsyncAnthropic for concurrent requests
  - API key from the HANSARD_ANTHROPIC_API_KEY environment variable

Reads:  prompts/stance.md, prompts/sexism.md
        data/validation/validation_sample.parquet (--scope validation)
        data/womens_rights/corpus_with_context.parquet (--scope corpus;
        run scripts/download_data.py first)
Writes (checkpointed JSONL, resumable; re-runs skip rows already present):
  data/classifications/stance_llm.jsonl
  data/classifications/sexism_llm.jsonl

Usage:
  # 300-speech validation set only (default):
  python scripts/classification/run_llm_classification.py --scope validation

  # Full corpus (~6,531 speeches):
  python scripts/classification/run_llm_classification.py --scope corpus

  # Smoke test (first 5 speeches):
  python scripts/classification/run_llm_classification.py --scope validation --limit 5
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "prompts"
VALIDATION = REPO / "data" / "validation" / "validation_sample.parquet"
CORPUS_WITH_CONTEXT = REPO / "data" / "womens_rights" / "corpus_with_context.parquet"

STANCE_OUT = REPO / "data" / "classifications" / "stance_llm.jsonl"
SEXISM_OUT = REPO / "data" / "classifications" / "sexism_llm.jsonl"

MODEL = "claude-sonnet-4-6"  # same model the paper reports
API_KEY_ENV = "HANSARD_ANTHROPIC_API_KEY"
API_KEY = os.environ.get(API_KEY_ENV)
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 5

# --------------------------------------------------------------------------- #
# Structured output schemas (forced via tool_choice -> guaranteed valid JSON)
# --------------------------------------------------------------------------- #

STANCE_TOOL = {
    "name": "classify_stance",
    "description": (
        "Record the stance classification for the parliamentary speech under "
        "Pass 1 (stance classification). Apply this tool exactly once per call."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "stance": {
                "type": "string",
                "enum": ["for", "against", "both", "irrelevant"],
                "description": (
                    "Speaker's stance on women's political rights. Use the "
                    "definitions and edge-case rules in the system prompt."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "1-2 sentence justification grounded in the speech."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the call, 0.0 to 1.0.",
            },
        },
        "required": ["stance", "rationale", "confidence"],
        "additionalProperties": False,
    },
}

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

SEXISM_TOOL = {
    "name": "classify_sexism",
    "description": (
        "Record the multi-label AST sexism classification for the speech "
        "under Pass 2. Hostile and benevolent are independent binary flags. "
        "If a binary flag is true, the corresponding subcategory list must "
        "contain at least one entry; if no subcategory fits, set the binary "
        "flag to false. Apply this tool exactly once per call."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "hostile": {
                "type": "boolean",
                "description": (
                    "Whether the speech contains hostile sexism (degrades, "
                    "blames, controls; see system prompt). Must be true iff "
                    "hostile_subcategories is non-empty."
                ),
            },
            "hostile_subcategories": {
                "type": "array",
                "items": {"type": "string", "enum": HOSTILE_SUBS},
                "description": (
                    "Subcategories of hostile sexism that apply. Must be "
                    "non-empty when hostile is true; must be empty when "
                    "hostile is false."
                ),
            },
            "hostile_quote": {
                "type": "string",
                "description": (
                    "Verbatim quote from the speech supporting the hostile "
                    "call. Empty string if hostile is false."
                ),
            },
            "benevolent": {
                "type": "boolean",
                "description": (
                    "Whether the speech contains benevolent sexism "
                    "(essentializes women's traits to restrict roles). "
                    "Must be true iff benevolent_subcategories is non-empty."
                ),
            },
            "benevolent_subcategories": {
                "type": "array",
                "items": {"type": "string", "enum": BENEVOLENT_SUBS},
                "description": (
                    "Subcategories of benevolent sexism that apply. Must be "
                    "non-empty when benevolent is true; must be empty when "
                    "benevolent is false."
                ),
            },
            "benevolent_quote": {
                "type": "string",
                "description": (
                    "Verbatim quote from the speech supporting the benevolent "
                    "call. Empty string if benevolent is false."
                ),
            },
        },
        "required": [
            "hostile", "hostile_subcategories", "hostile_quote",
            "benevolent", "benevolent_subcategories", "benevolent_quote",
        ],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------- #
# Prompt extraction
# --------------------------------------------------------------------------- #

def extract_prompt_body(md_path: Path) -> str:
    """Pull the prompt content out of the first fenced code block."""
    m = re.search(r"```\s*\n(.*?)\n```", md_path.read_text(), re.DOTALL)
    if not m:
        raise SystemExit(f"No fenced code block found in {md_path}")
    return m.group(1)


def split_system_user(prompt_body: str) -> tuple[str, str]:
    """Split a prompt body into (system, user_template) on the 'USER' line.

    Strips the JSON-schema OUTPUT section from the system prompt -- the
    tool definition replaces it.
    """
    parts = re.split(r"^USER\s*$", prompt_body, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        raise SystemExit("Expected SYSTEM ... USER ... structure in prompt")
    system = re.sub(r"^SYSTEM\s*\n", "", parts[0]).strip()
    user_template = parts[1].strip()
    # Strip the "Return ONLY this JSON" + schema block (now handled by tool_use)
    system = re.sub(
        r"\n+Return ONLY this JSON[^{]*\{[^}]*\}[^=]*$",
        "",
        system,
        flags=re.DOTALL,
    )
    system = re.sub(
        r"={3,}\s*\nOUTPUT.*$",
        "",
        system,
        flags=re.DOTALL,
    ).strip()
    return system, user_template


# --------------------------------------------------------------------------- #
# Storage layer (checkpointed JSONL)
# --------------------------------------------------------------------------- #

def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["speech_id"])
            except Exception:
                continue
    return done


def append_line(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #

async def classify(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    system: str,
    user_text: str,
    tool: dict,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: int = 1024,
) -> dict:
    """Issue one classification call, forcing the tool schema via tool_choice."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                msg = await client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    # Cache the (frozen) system prompt + tool definition.
                    # cache_control on the last system block caches tools+system
                    # together since render order is tools -> system -> messages.
                    system=[{
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": user_text}],
                )
                # Find the tool_use block (forced by tool_choice).
                for block in msg.content:
                    if block.type == "tool_use":
                        return {
                            "ok": True,
                            "parsed": block.input,
                            "stop_reason": msg.stop_reason,
                            "input_tokens": msg.usage.input_tokens,
                            "output_tokens": msg.usage.output_tokens,
                            "cache_read_input_tokens":
                                getattr(msg.usage, "cache_read_input_tokens", 0),
                            "cache_creation_input_tokens":
                                getattr(msg.usage, "cache_creation_input_tokens", 0),
                        }
                return {"ok": False, "error": "no tool_use block in response"}
            except anthropic.RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except anthropic.OverloadedError:
                await asyncio.sleep(5 * (attempt + 1))
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"ok": False, "error": f"{type(e).__name__}: {e}"}
            except Exception as e:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": False, "error": "max_retries_exceeded"}


# --------------------------------------------------------------------------- #
# Pass 1: stance
# --------------------------------------------------------------------------- #

async def pass1_stance(speeches, concurrency, dry_run=False) -> None:
    system, user_tpl = split_system_user(
        extract_prompt_body(PROMPTS / "stance.md")
    )
    done = load_done(STANCE_OUT)
    remaining = [s for s in speeches if s["speech_id"] not in done]
    print(f"[stance] total={len(speeches)}  done={len(done)}  "
          f"remaining={len(remaining)}")
    if dry_run:
        print("[stance] dry run: not calling API"); return

    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    semaphore = asyncio.Semaphore(concurrency)
    written = 0
    cache_reads = cache_writes = uncached_in = output_out = 0

    async def one(s):
        nonlocal written, cache_reads, cache_writes, uncached_in, output_out
        user_text = (user_tpl
                     .replace("{target_text}", s["target_text"] or "")
                     .replace("{context_text}", s["context_text"] or "[No context available]"))
        result = await classify(client, semaphore, system, user_text, STANCE_TOOL)
        rec = {"speech_id": s["speech_id"], "pass": "stance"}
        if result["ok"]:
            rec.update({
                "parsed": result["parsed"],
                "stop_reason": result["stop_reason"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cache_read_input_tokens": result["cache_read_input_tokens"],
                "cache_creation_input_tokens": result["cache_creation_input_tokens"],
            })
            cache_reads += result["cache_read_input_tokens"]
            cache_writes += result["cache_creation_input_tokens"]
            uncached_in += result["input_tokens"]
            output_out += result["output_tokens"]
        else:
            rec["error"] = result["error"]
        append_line(STANCE_OUT, rec)
        written += 1
        if written % 50 == 0:
            print(f"  [stance] {written}/{len(remaining)} written  "
                  f"(cache reads: {cache_reads:,}  writes: {cache_writes:,})")

    await asyncio.gather(*(one(s) for s in remaining))
    print(f"[stance] done. wrote {written} new rows -> {STANCE_OUT}")
    print(f"[stance] tokens: cache_read={cache_reads:,}  "
          f"cache_creation={cache_writes:,}  uncached_input={uncached_in:,}  "
          f"output={output_out:,}")


# --------------------------------------------------------------------------- #
# Pass 2: sexism (only relevant speeches from Pass 1)
# --------------------------------------------------------------------------- #

def relevant_speeches_from_stance(stance_jsonl: Path, all_speeches) -> list:
    """Subset of speeches whose Pass 1 stance is non-irrelevant. Preserves the
    full speech record (including context_text) for Pass 2."""
    stance_by_id = {}
    with open(stance_jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            stance_by_id[rec["speech_id"]] = (
                (rec.get("parsed") or {}).get("stance")
            )
    out = []
    for s in all_speeches:
        st = stance_by_id.get(s["speech_id"])
        if st in ("for", "against", "both"):
            out.append({**s, "_stance": st})
    return out


async def pass2_sexism(speeches, concurrency, dry_run=False) -> None:
    system, user_tpl = split_system_user(
        extract_prompt_body(PROMPTS / "sexism.md")
    )
    done = load_done(SEXISM_OUT)
    remaining = [s for s in speeches if s["speech_id"] not in done]
    print(f"[sexism] total={len(speeches)}  done={len(done)}  "
          f"remaining={len(remaining)}")
    if dry_run:
        print("[sexism] dry run: not calling API"); return

    client = anthropic.AsyncAnthropic(api_key=API_KEY)
    semaphore = asyncio.Semaphore(concurrency)
    written = 0
    cache_reads = cache_writes = uncached_in = output_out = 0

    async def one(s):
        nonlocal written, cache_reads, cache_writes, uncached_in, output_out
        user_text = (user_tpl
                     .replace("{stance}", s["_stance"])
                     .replace("{target_text}", s["target_text"] or "")
                     .replace("{context_text}", s["context_text"] or "[No context available]"))
        result = await classify(
            client, semaphore, system, user_text, SEXISM_TOOL, max_tokens=1024,
        )
        rec = {"speech_id": s["speech_id"], "pass": "sexism", "stance": s["_stance"]}
        if result["ok"]:
            rec.update({
                "parsed": result["parsed"],
                "stop_reason": result["stop_reason"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cache_read_input_tokens": result["cache_read_input_tokens"],
                "cache_creation_input_tokens": result["cache_creation_input_tokens"],
            })
            cache_reads += result["cache_read_input_tokens"]
            cache_writes += result["cache_creation_input_tokens"]
            uncached_in += result["input_tokens"]
            output_out += result["output_tokens"]
        else:
            rec["error"] = result["error"]
        append_line(SEXISM_OUT, rec)
        written += 1
        if written % 50 == 0:
            print(f"  [sexism] {written}/{len(remaining)} written  "
                  f"(cache reads: {cache_reads:,}  writes: {cache_writes:,})")

    await asyncio.gather(*(one(s) for s in remaining))
    print(f"[sexism] done. wrote {written} new rows -> {SEXISM_OUT}")
    print(f"[sexism] tokens: cache_read={cache_reads:,}  "
          f"cache_creation={cache_writes:,}  uncached_input={uncached_in:,}  "
          f"output={output_out:,}")


# --------------------------------------------------------------------------- #
# Speech loading
# --------------------------------------------------------------------------- #

def load_validation() -> list:
    """Load 300-speech validation set with proper 5+5 context built from the
    turns parquet (overrides the validation parquet's stored context_text,
    which inherits the speech_id-misalignment bug)."""
    val = pd.read_parquet(VALIDATION)[["speech_id"]]
    corpus = pd.read_parquet(CORPUS_WITH_CONTEXT)[
        ["speech_id", "target_text", "context_text"]
    ]
    df = val.merge(corpus, on="speech_id", how="left")
    missing = df["target_text"].isna().sum()
    if missing:
        raise SystemExit(f"FATAL: {missing} validation speeches not in corpus_with_context")
    return df.to_dict("records")


def load_corpus() -> list:
    """Load 6,531-speech corpus with proper 5+5 context built from the turns
    parquet (replaces suffrage_classified_with_text.parquet's context_text,
    which had '[No context available]' for ~58% of speeches)."""
    df = pd.read_parquet(CORPUS_WITH_CONTEXT)[
        ["speech_id", "target_text", "context_text"]
    ]
    return df.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--scope", choices=["validation", "corpus"],
                    default="validation",
                    help="validation = 300 speeches; corpus = full 6,531")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of speeches (for smoke testing)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--skip-stance", action="store_true",
                    help="Skip Pass 1 (use existing stance_llm.jsonl)")
    ap.add_argument("--skip-sexism", action="store_true", help="Skip Pass 2")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show counts; do not call the API")
    args = ap.parse_args()

    if not API_KEY:
        print(
            f"ERROR: {API_KEY_ENV} is not set in the environment.\n"
            f"  Load from 1Password, e.g.:\n"
            f"    export {API_KEY_ENV}=\"$(op read 'op://Cluster Secrets/anthropic-hansard/credential')\"",
            file=sys.stderr,
        )
        sys.exit(1)

    speeches = load_validation() if args.scope == "validation" else load_corpus()
    if args.limit:
        speeches = speeches[: args.limit]
    print(f"Loaded {len(speeches)} speeches for scope={args.scope}")

    if not args.skip_stance:
        asyncio.run(pass1_stance(speeches, args.concurrency, args.dry_run))

    if not args.skip_sexism:
        relevant = relevant_speeches_from_stance(STANCE_OUT, speeches)
        print(f"[sexism] relevant after Pass 1: {len(relevant)}")
        asyncio.run(pass2_sexism(relevant, args.concurrency, args.dry_run))


if __name__ == "__main__":
    main()
