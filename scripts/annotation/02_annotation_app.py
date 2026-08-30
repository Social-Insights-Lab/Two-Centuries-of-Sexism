"""
V8 annotation app: stance + multi-label AST sexism with subcategories.

Schema (per coauthor decision 2026-05-19):
  - Stance: exactly one of for / against / both / irrelevant
  - Hostile sexism: multi-label across 3 subcategories (or none)
  - Benevolent sexism: multi-label across 3 subcategories (or none)
  - Hostile and benevolent are independent; both/either may be selected

Design:
  - Two annotators (Omar, Mandira) each label 500 speeches independently
  - LLM labels are loaded but never displayed; annotator works blind
  - Auto-save on every change (atomic write); navigation is always free
  - Long speeches (>3000 words) default to keyword-excerpt view
  - Resume = jump to first unannotated speech on login

Usage:
    streamlit run experiments/20260520_v8_500_validation/02_annotation_app.py
"""
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "validation_sample.parquet"
ANNOTATIONS_DIR = ROOT / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

STANCE_OPTIONS = ["for", "against", "both", "irrelevant"]
STANCE_HELP = {
    "for": "Supports women's political rights or representation.",
    "against": "Opposes women's political rights or representation.",
    "both": "Explicit support AND opposition in the same speech.",
    "irrelevant": "Not about women's political rights at all.",
}

HOSTILE_SUBS = [
    ("dominative_paternalism",
     "Dominative paternalism",
     "Women are incompetent; men must rule, decide, control."),
    ("competitive_gender_differentiation",
     "Competitive gender differentiation",
     "Men have traits women lack (rationality, judgment, courage)."),
    ("heterosexual_hostility",
     "Heterosexual hostility",
     "Women's sexuality framed as manipulative, dangerous, threatening."),
]

BENEVOLENT_SUBS = [
    ("protective_paternalism",
     "Protective paternalism",
     "Women are fragile; men should protect, provide, shield."),
    ("complementary_gender_differentiation",
     "Complementary gender differentiation",
     "Women have special purity / nurturance / moral nature."),
    ("heterosexual_intimacy",
     "Heterosexual intimacy",
     "Men are incomplete without women; women complete men."),
]

# Keyword matchers -- mirror scripts/analysis/extract_suffrage_reliable.py
#
# Tier 1 (HIGH) qualification uses the extractor's exact regex (with greedy
# .* pivots). We use that just for the yes/no qualification check. Visual
# highlighting uses the individual keyword tokens and phrases below, so that
# patterns like `women.*suffrage` don't blob into a single 500-char span.

TIER1_QUALIFY_REGEX = re.compile(
    r"women.*suffrage|female suffrage|suffrage.*women|"
    r"votes for women|suffragette|suffragist|"
    r"enfranchise.*women|women.*enfranchise|"
    r"equal franchise|"
    r"representation of the people.*women|"
    r"sex disqualification|"
    r"women.*social.*political.*union",
    re.IGNORECASE,
)

# For each Tier 1 alternative, declare the pivot regexes whose hits should
# be highlighted within any greedy match of that alternative. This makes the
# scattered-pivot false positives visible (e.g. women.*social.*political.*union
# firing on four words scattered across a 3000-char speech).
def _r(p):
    return re.compile(p, re.IGNORECASE)

TIER1_ALT_PIVOTS = [
    # (alternative regex, [pivot regexes to highlight inside its match span])
    (_r(r"women.*suffrage"),                          [_r(r"\bwomen\b"),     _r(r"\bsuffrage\w*\b")]),
    (_r(r"female\s+suffrage"),                        [_r(r"\bfemale\s+suffrage\b")]),
    (_r(r"suffrage.*women"),                          [_r(r"\bsuffrage\w*\b"), _r(r"\bwomen\b")]),
    (_r(r"votes?\s+for\s+women"),                     [_r(r"\bvotes?\s+for\s+women\b")]),
    (_r(r"suffragettes?"),                            [_r(r"\bsuffragettes?\b")]),
    (_r(r"suffragists?"),                             [_r(r"\bsuffragists?\b")]),
    (_r(r"enfranchise.*women"),                       [_r(r"\benfranchis\w*\b"), _r(r"\bwomen\b")]),
    (_r(r"women.*enfranchise"),                       [_r(r"\bwomen\b"), _r(r"\benfranchis\w*\b")]),
    (_r(r"equal\s+franchise"),                        [_r(r"\bequal\s+franchise\b")]),
    (_r(r"representation\s+of\s+the\s+people.*women"),[_r(r"\brepresentation\s+of\s+the\s+people\b"), _r(r"\bwomen\b")]),
    (_r(r"sex\s+disqualification"),                   [_r(r"\bsex\s+disqualification\b")]),
    (_r(r"women.*social.*political.*union"),          [_r(r"\bwomen\b"), _r(r"\bsocial\b"),
                                                       _r(r"\bpolitical\b"), _r(r"\bunion\b")]),
]

# Tier 2 (MEDIUM): women/female within 25 words of any voting term.
# Match the extractor's exact behaviour, which uses bare substring searches
# (e.g. r'vote' will fire on 'devoted', 'voters', 'devotedly'). Word-bounded
# regex would diverge from what actually qualified the speech for extraction.
TIER2_VOTING_SUBSTRINGS = (
    "vote", "voting", "voter", "voters",
    "electoral", "electorate",
    "franchise", "enfranchise",
    "representation",
)
TIER2_WINDOW_WORDS = 25  # +/- 25 words from each women/female occurrence


# --------------------------------------------------------------------------- #
# Storage layer (atomic JSONL per annotator)
# --------------------------------------------------------------------------- #

def annotation_path(annotator: str) -> Path:
    safe = re.sub(r"[^a-z0-9_-]", "_", annotator.strip().lower())
    return ANNOTATIONS_DIR / f"{safe}.jsonl"


def load_annotations(annotator: str) -> dict:
    """Return {speech_id: annotation_record}."""
    path = annotation_path(annotator)
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["speech_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def save_annotations(annotator: str, records: dict) -> None:
    """Atomic write of all records to per-annotator JSONL."""
    path = annotation_path(annotator)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    )
    try:
        for speech_id in sorted(records.keys()):
            tmp.write(json.dumps(records[speech_id], ensure_ascii=False) + "\n")
        tmp.flush()
        Path(tmp.name).replace(path)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- #
# Keyword detection + rendering
# --------------------------------------------------------------------------- #

def _word_positions(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, word_lower) for each whitespace-delimited token,
    matching the extraction script's `text_lower.split()` behaviour."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        out.append((i, j, text[i:j].lower()))
        i = j
    return out


def find_tier1_spans(text: str) -> list[tuple[int, int]]:
    """Tier 1 highlights: the literal pivot-word occurrences of whichever
    Tier 1 alternative(s) the speech actually fires under, restricted to
    inside each greedy match span. This way single-word patterns (suffragette)
    light up wherever they appear, and scatter-pivot patterns (e.g.
    women.*social.*political.*union) light up each of women/social/political/
    union inside the match span -- annotator can see whether the four words
    are clustered (real WSPU mention) or scattered (extractor false positive)."""
    spans: list[tuple[int, int]] = []
    for alt_regex, pivot_regexes in TIER1_ALT_PIVOTS:
        for m in alt_regex.finditer(text):
            window = text[m.start():m.end()]
            for pr in pivot_regexes:
                for pm in pr.finditer(window):
                    spans.append((m.start() + pm.start(), m.start() + pm.end()))
    return spans


def _word_contains_voting(lower_word: str) -> bool:
    """Substring check matching the extractor's `re.search(pat, context)`
    behaviour: a word containing 'vote' (e.g. 'devoted', 'voters') counts."""
    return any(sub in lower_word for sub in TIER2_VOTING_SUBSTRINGS)


def find_tier2_spans(text: str) -> list[tuple[int, int]]:
    """Tier 2: women/female occurrences plus the voting term(s) within +/- 25
    words. Mirrors `_is_medium_confidence` in the extractor: substring
    matching, so qualifying voting tokens include things like 'devoted'
    (contains 'vote') if they happen to sit within the window."""
    words = _word_positions(text)
    if not words:
        return []

    gender_idx = [i for i, (_, _, w) in enumerate(words)
                  if "women" in w or "female" in w]

    out: list[tuple[int, int]] = []
    seen_voting_token_idx: set[int] = set()

    for gi in gender_idx:
        start_word = max(0, gi - TIER2_WINDOW_WORDS)
        end_word = min(len(words), gi + TIER2_WINDOW_WORDS)
        window_idx = list(range(start_word, end_word))
        voting_hits = [wi for wi in window_idx
                       if wi != gi and _word_contains_voting(words[wi][2])]
        if not voting_hits:
            continue
        s, e, _ = words[gi]
        out.append((s, e))
        for wi in voting_hits:
            if wi in seen_voting_token_idx:
                continue
            seen_voting_token_idx.add(wi)
            s, e, _ = words[wi]
            out.append((s, e))
    return out


def find_keyword_spans(text: str) -> tuple[list[tuple[int, int, str]], dict]:
    """Return (spans, info). Spans are non-overlapping (start, end, kind)
    with kind in {tier1, tier2}. info reports which tier qualified the speech."""
    qualifies_tier1 = bool(TIER1_QUALIFY_REGEX.search(text))
    tier1_spans = find_tier1_spans(text)
    tier2_spans = find_tier2_spans(text)

    if qualifies_tier1:
        qualifying_tier = "Tier 1 (HIGH)"
        # Show Tier 1 highlights; also show Tier 2 pairs for context (women+voting)
        raw = [(s, e, "tier1") for s, e in tier1_spans] + \
              [(s, e, "tier2") for s, e in tier2_spans]
    elif tier2_spans:
        qualifying_tier = "Tier 2 (MEDIUM)"
        raw = [(s, e, "tier2") for s, e in tier2_spans]
    else:
        qualifying_tier = "(no extraction match)"
        raw = []

    # Merge overlaps; tier1 wins
    raw.sort(key=lambda s: (s[0], 0 if s[2] == "tier1" else 1))
    merged: list[tuple[int, int, str]] = []
    for s, e, k in raw:
        if merged and s < merged[-1][1]:
            # Overlap: expand previous if same kind, or skip if previous is higher precedence
            ps, pe, pk = merged[-1]
            if pk == k:
                merged[-1] = (ps, max(pe, e), pk)
            # else previous is tier1 (higher precedence); drop this tier2 span
            continue
        merged.append((s, e, k))

    info = {
        "qualifying_tier": qualifying_tier,
        "tier1_token_spans": len(tier1_spans),
        "tier2_pair_spans": len(tier2_spans),
    }
    return merged, info


def render_highlighted_html(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Wrap spans in <mark> tags using theme-aware kw-tier1/kw-tier2 classes."""
    pieces = []
    cursor = 0
    for start, end, kind in spans:
        if start < cursor:
            continue
        pieces.append(_escape(text[cursor:start]))
        cls = "kw-tier1" if kind == "tier1" else "kw-tier2"
        pieces.append(f'<mark class="{cls}">{_escape(text[start:end])}</mark>')
        cursor = end
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces)


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z—‘“])")


def extract_keyword_sentences(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Return HTML showing just the sentences that contain at least one
    keyword span, with the highlights preserved. Dedupes overlapping sentences
    and joins separate hits with a `...` marker so the annotator can see
    quickly what the extraction caught.
    """
    if not text or not spans:
        return ""

    # Compute sentence boundaries (char offsets) once
    starts = [0]
    for m in _SENTENCE_BREAK.finditer(text):
        starts.append(m.end())
    starts.append(len(text))
    sent_ranges = list(zip(starts[:-1], starts[1:]))

    # Find which sentences contain any keyword
    hit_sentences: list[tuple[int, int]] = []
    for sa, sb in sent_ranges:
        if any(sa <= s < sb for s, _, _ in spans):
            hit_sentences.append((sa, sb))
    if not hit_sentences:
        return ""

    # Merge adjacent sentences (no gap marker between them)
    merged: list[tuple[int, int]] = []
    for sa, sb in hit_sentences:
        if merged and sa == merged[-1][1]:
            merged[-1] = (merged[-1][0], sb)
        else:
            merged.append((sa, sb))

    # Render each merged block with its highlights
    chunks = []
    for i, (sa, sb) in enumerate(merged):
        sub_text = text[sa:sb].strip()
        sub_spans = [(s - sa, e - sa, k) for s, e, k in spans if sa <= s < sb]
        chunks.append(render_highlighted_html(sub_text, sub_spans))
    return ' <span style="opacity:0.5">...</span> '.join(chunks)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PATH)
    # Strip LLM label columns so we cannot accidentally render them
    return df.drop(columns=["stance", "binary", "axis_a_label", "confidence"],
                   errors="ignore")


def _context_list(value) -> list[dict]:
    """Normalize a preceding/following column entry to a plain list of dicts."""
    if value is None:
        return []
    try:
        # numpy array / list / pandas array
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "items"):
                out.append({k: item[k] for k in item})
            else:
                out.append({})
        return out
    except TypeError:
        return []


def render_context_section(title: str, speeches: list[dict], default_open: int = 0):
    """Render a Preceding / Following section. First `default_open` speeches
    are expanded by default; the rest are collapsed click-to-expand."""
    if not speeches:
        st.caption(f"_{title}: none in debate_")
        return
    st.markdown(f"**{title}** ({len(speeches)})")
    for i, sp in enumerate(speeches):
        speaker = sp.get("speaker", "?")
        seq = sp.get("sequence_number", "?")
        wc = sp.get("word_count", 0)
        text = sp.get("text", "") or ""
        header = (f"seq {seq}  -  {speaker}  -  {wc} words")
        with st.expander(header, expanded=(i < default_open)):
            st.markdown(
                f"<div class='speech-body ctx-body'>{_escape(text)}</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Annotation record helpers
# --------------------------------------------------------------------------- #

def empty_record(row) -> dict:
    return {
        "speech_id": row["speech_id"],
        "sample_idx": int(row["sample_idx"]),
        "stance": None,
        "hostile_subcategories": [],
        "benevolent_subcategories": [],
        "hostile": False,
        "benevolent": False,
        "sexist": False,
        "notes": "",
        "flagged": False,
        "annotator": None,
        "annotated_at": None,
    }


def is_complete(rec: dict) -> bool:
    """A record counts as annotated once stance is set."""
    return rec is not None and rec.get("stance") in STANCE_OPTIONS


def derive(rec: dict) -> dict:
    rec["hostile"] = bool(rec.get("hostile_subcategories"))
    rec["benevolent"] = bool(rec.get("benevolent_subcategories"))
    rec["sexist"] = rec["hostile"] or rec["benevolent"]
    return rec


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="V8 Annotation", layout="wide")

st.markdown(
    """
    <style>
      .speech-body { font-size: 1.02rem; line-height: 1.55; color: inherit; }
      .speech-body p { margin: 0 0 0.6em 0; }
      .meta-pill { display:inline-block; padding:2px 8px; border-radius:10px;
                   background: rgba(128,128,128,0.18); color: inherit;
                   font-size:0.78rem; margin-right:6px; }
      .meta-pill-warn { background: rgba(239,68,68,0.18); color: #ef4444; }
      .meta-pill-speaker { background: rgba(99,102,241,0.18); color: inherit;
                            font-weight: 600; }
      .ctx-body { font-size: 0.95rem; color: inherit; opacity: 0.92; }
      mark.kw-tier1 { background: rgba(252,211,77,0.55); color: inherit;
                       padding:0 2px; border-radius:2px; }
      mark.kw-tier2 { background: rgba(96,165,250,0.45); color: inherit;
                       padding:0 2px; border-radius:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session bootstrap
if "annotator" not in st.session_state:
    st.session_state.annotator = ""
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "all"

sample = load_sample()
N = len(sample)


# --------- Sidebar: login, progress, navigation --------- #

with st.sidebar:
    st.markdown("### Annotator")
    name = st.text_input("Your name (e.g. omar)", value=st.session_state.annotator)
    if name != st.session_state.annotator:
        st.session_state.annotator = name
        st.session_state.annotations = load_annotations(name) if name else {}
        # Resume at first unannotated
        for i in range(N):
            sid = sample.iloc[i]["speech_id"]
            if not is_complete(st.session_state.annotations.get(sid)):
                st.session_state.idx = i
                break
        else:
            st.session_state.idx = 0
        st.rerun()

    if not st.session_state.annotator:
        st.info("Enter your name to start.")
        st.stop()

    annotations = st.session_state.annotations
    completed = sum(1 for r in annotations.values() if is_complete(r))
    flagged = sum(1 for r in annotations.values() if r.get("flagged"))
    st.metric("Completed", f"{completed}/{N}")
    st.progress(completed / N if N else 0.0)
    if flagged:
        st.caption(f"{flagged} flagged for discussion")

    st.markdown("### Navigation")
    st.session_state.filter_mode = st.radio(
        "Show", ["all", "unannotated", "flagged"],
        index=["all", "unannotated", "flagged"].index(st.session_state.filter_mode),
        horizontal=True,
    )

    def filtered_indices() -> list[int]:
        mode = st.session_state.filter_mode
        out = []
        for i in range(N):
            sid = sample.iloc[i]["speech_id"]
            rec = annotations.get(sid)
            if mode == "unannotated" and is_complete(rec):
                continue
            if mode == "flagged" and not (rec and rec.get("flagged")):
                continue
            out.append(i)
        return out

    visible = filtered_indices()
    if not visible:
        st.success("Nothing to show with current filter.")
        st.stop()

    if st.session_state.idx not in visible:
        st.session_state.idx = visible[0]

    pos_in_visible = visible.index(st.session_state.idx)
    cols = st.columns([1, 1, 1, 1])
    if cols[0].button("First", help="First in filter", use_container_width=True,
                       disabled=pos_in_visible == 0):
        st.session_state.idx = visible[0]; st.rerun()
    if cols[1].button("Prev", help="Previous in filter", use_container_width=True,
                       disabled=pos_in_visible == 0):
        st.session_state.idx = visible[pos_in_visible - 1]; st.rerun()
    if cols[2].button("Next", help="Next in filter", use_container_width=True,
                       disabled=pos_in_visible == len(visible) - 1):
        st.session_state.idx = visible[pos_in_visible + 1]; st.rerun()
    if cols[3].button("Last", help="Last in filter", use_container_width=True,
                       disabled=pos_in_visible == len(visible) - 1):
        st.session_state.idx = visible[-1]; st.rerun()

    jump = st.number_input("Jump to sample_idx (0-indexed)", min_value=0, max_value=N - 1,
                            value=st.session_state.idx, step=1)
    if jump != st.session_state.idx:
        st.session_state.idx = int(jump); st.rerun()

    with st.expander("Codebook quick reference"):
        st.markdown(
            """
**Stance**
- **for**: supports women's political rights/representation
- **against**: opposes
- **both**: explicit support AND opposition
- **irrelevant**: not about women's political rights

**Hostile sexism** -- degrades, blames, controls; treats women as inferior/threatening

**Benevolent sexism** -- essentializes women (attributes inherent traits to women as a class), idealises while restricting. The boundary: praising specific women's accomplishments is NOT sexist; claiming women as a class have a special moral/nurturing nature IS.

**Hostile and benevolent are independent** -- both can be present.

**Quoting sexism to condemn it is NOT sexist** -- only mark sexist if the speaker endorses the framing.
            """
        )


# --------- Main: speech + annotation form --------- #

row = sample.iloc[st.session_state.idx]
sid = row["speech_id"]
rec = annotations.get(sid) or empty_record(row)

# Header
left, right = st.columns([3, 1])
with left:
    st.markdown(f"### {row['speaker']} ({int(row['year'])})")
    pills = [
        f"sample {int(row['sample_idx']) + 1}/{N}",
        f"era {row['era']}",
        f"{int(row['word_count'])} words",
    ]
    if pd.notna(row.get("gender")):
        pills.append(f"gender: {row['gender']}")
    if pd.notna(row.get("party")):
        pills.append(f"party: {row['party']}")
    if pd.notna(row.get("chamber")):
        pills.append(f"{row['chamber']}")
    st.markdown(" ".join(f'<span class="meta-pill">{p}</span>' for p in pills),
                unsafe_allow_html=True)
    st.caption(f"speech_id: `{sid}`  •  debate_id: `{row.get('debate_id', '')}`")

with right:
    status = "[OK] annotated" if is_complete(rec) else "unannotated"
    flag = "[FLAG] " if rec.get("flagged") else ""
    st.markdown(f"**{flag}{status}**")
    if rec.get("annotated_at"):
        st.caption(f"saved {rec['annotated_at']}")

st.divider()

# Speech body
text = row["target_text"] or ""
spans, span_info = find_keyword_spans(text)

preceding = _context_list(row.get("preceding_speeches"))
following = _context_list(row.get("following_speeches"))
ctx_present = bool(preceding) or bool(following)

pills = (
    f"<span class='meta-pill'>{span_info['qualifying_tier']}</span>"
    f"<span class='meta-pill'>{len(spans)} keyword highlights</span>"
)
if ctx_present:
    pills += (f"<span class='meta-pill'>"
              f"context: {len(preceding)} before / {len(following)} after"
              f"</span>")
else:
    pills += "<span class='meta-pill meta-pill-warn'>no context</span>"
st.markdown(f"#### Speech {pills}", unsafe_allow_html=True)

quickview_html = extract_keyword_sentences(text, spans)
if quickview_html:
    st.markdown(
        f"<div style='background:rgba(128,128,128,0.08);border-left:3px solid "
        f"rgba(252,211,77,0.7);padding:10px 14px;border-radius:4px;"
        f"margin-bottom:14px'>"
        f"<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;"
        f"opacity:0.7;margin-bottom:6px'>Keyword sentences (skim first)</div>"
        f"<div class='speech-body' style='font-size:0.96rem'>{quickview_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

html = render_highlighted_html(text, spans)
st.markdown(f'<div class="speech-body" id="speech-body-anchor">{html}</div>',
            unsafe_allow_html=True)

if ctx_present:
    with st.expander(
        f"Debate context  -  {len(preceding)} before / {len(following)} after",
        expanded=False,
    ):
        render_context_section("Preceding speeches", preceding)
        st.markdown("---")
        render_context_section("Following speeches", following)

st.divider()

# --------- Annotation form --------- #

st.markdown("#### Your annotation")

def save_now():
    rec["annotator"] = st.session_state.annotator
    rec["annotated_at"] = datetime.now().isoformat(timespec="seconds")
    derive(rec)
    annotations[sid] = rec
    save_annotations(st.session_state.annotator, annotations)

# Stance
stance_idx = STANCE_OPTIONS.index(rec["stance"]) if rec.get("stance") in STANCE_OPTIONS else None
new_stance = st.radio(
    "Stance  (F=for, A=against, B=both, I=irrelevant)",
    STANCE_OPTIONS,
    index=stance_idx,
    horizontal=True,
    format_func=lambda s: f"{s}",
    help=" | ".join(f"**{s}**: {STANCE_HELP[s]}" for s in STANCE_OPTIONS),
    key=f"stance_{sid}",
)
if new_stance != rec.get("stance"):
    prev = rec.get("stance")
    rec["stance"] = new_stance
    save_now()
    # Auto-advance when irrelevant is freshly selected -- one-keystroke flow
    if new_stance == "irrelevant" and prev != "irrelevant":
        if pos_in_visible < len(visible) - 1:
            st.session_state.idx = visible[pos_in_visible + 1]
        st.rerun()

# Hostile and benevolent in two columns
hostile_col, benevolent_col = st.columns(2)

with hostile_col:
    st.markdown("**Hostile sexism** -- degrade / blame / control")
    current_h = set(rec.get("hostile_subcategories", []))
    new_h = set()
    for key, label, desc in HOSTILE_SUBS:
        checked = st.checkbox(label, value=key in current_h, help=desc,
                              key=f"h_{key}_{sid}")
        if checked:
            new_h.add(key)
    if new_h != current_h:
        rec["hostile_subcategories"] = sorted(new_h)
        save_now()

with benevolent_col:
    st.markdown("**Benevolent sexism** -- idealise / essentialise / restrict")
    current_b = set(rec.get("benevolent_subcategories", []))
    new_b = set()
    for key, label, desc in BENEVOLENT_SUBS:
        checked = st.checkbox(label, value=key in current_b, help=desc,
                              key=f"b_{key}_{sid}")
        if checked:
            new_b.add(key)
    if new_b != current_b:
        rec["benevolent_subcategories"] = sorted(new_b)
        save_now()

# Notes + flag
note_col, flag_col = st.columns([4, 1])
with note_col:
    new_notes = st.text_area(
        "Notes (optional)", value=rec.get("notes", ""), height=70,
        placeholder="Anything to discuss in resolution -- ambiguity, hard cases, etc.",
        key=f"notes_{sid}",
    )
    if new_notes != rec.get("notes", ""):
        rec["notes"] = new_notes
        save_now()
with flag_col:
    st.write("")
    new_flag = st.checkbox("Flag", value=rec.get("flagged", False),
                            help="Mark for discussion with co-annotator",
                            key=f"flag_{sid}")
    if new_flag != rec.get("flagged", False):
        rec["flagged"] = new_flag
        save_now()

# Save indicator + advance
status_col, advance_col = st.columns([3, 1])
with status_col:
    if rec.get("annotated_at"):
        derived = []
        if rec["hostile"]: derived.append("hostile")
        if rec["benevolent"]: derived.append("benevolent")
        if not derived: derived.append("not sexist")
        st.caption(f"Saved {rec['annotated_at']} -- stance: {rec.get('stance', '-') or '-'}; "
                   f"sexism: {', '.join(derived)}")
    else:
        st.caption("Set stance to save this annotation.")
with advance_col:
    if st.button("Save & Next  (Enter)", type="primary", use_container_width=True,
                 disabled=not is_complete(rec)):
        # Find next visible
        if pos_in_visible < len(visible) - 1:
            st.session_state.idx = visible[pos_in_visible + 1]
        st.rerun()


# Keyboard shortcuts: F/A/B/I = stance, 1-3 = hostile subs, 4-6 = benevolent
# subs, Enter = Save & Next. Listener attaches to the parent document once;
# component re-renders are idempotent.
components.html(
    """
<script>
(function() {
  const doc = window.parent.document;

  const STANCE = {f: 'for', a: 'against', b: 'both', i: 'irrelevant'};
  const SUBS = {
    '1': 'Dominative paternalism',
    '2': 'Competitive gender differentiation',
    '3': 'Heterosexual hostility',
    '4': 'Protective paternalism',
    '5': 'Complementary gender differentiation',
    '6': 'Heterosexual intimacy',
  };

  function clickLabelled(selector, target) {
    const els = doc.querySelectorAll(selector);
    for (const el of els) {
      const lbl = (el.closest('label')?.innerText || '').trim();
      if (lbl === target) { el.click(); return true; }
    }
    return false;
  }

  function handler(e) {
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;

    const key = e.key.toLowerCase();

    if (STANCE[key]) {
      if (clickLabelled('input[type=radio]', STANCE[key])) e.preventDefault();
      return;
    }
    if (SUBS[key]) {
      if (clickLabelled('input[type=checkbox]', SUBS[key])) e.preventDefault();
      return;
    }
    if (e.key === 'Enter') {
      const buttons = doc.querySelectorAll('button');
      for (const b of buttons) {
        const txt = (b.innerText || '').trim();
        if (txt.startsWith('Save & Next') && !b.disabled) {
          e.preventDefault();
          b.click();
          return;
        }
      }
    }
  }

  // Streamlit destroys this iframe on every rerun, which orphans any
  // listener that closed over the iframe's V8 scope. So we remove any
  // previous handler and attach a fresh one each time the script runs.
  if (doc._v8_handler) {
    doc.removeEventListener('keydown', doc._v8_handler);
  }
  doc._v8_handler = handler;
  doc.addEventListener('keydown', handler);
})();
</script>
""",
    height=0,
)
