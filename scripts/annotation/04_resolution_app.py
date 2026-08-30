"""
V8 resolution app: walk through the 106 disagreements between Omar's and
Mandira's annotations and commit consensus labels.

Workflow (designed for both annotators sitting together):
- Sidebar: progress (resolved / total), nav, filter (unresolved / all)
- Main: speech text with keyword highlights, then side-by-side comparison
  of Omar's vs Mandira's labels, then a consensus form pre-filled from one
  side. Quick buttons: Use Omar's / Use Mandira's / Not sexist.
- Each disagreement's consensus is auto-saved to annotations/consensus.jsonl
  on change (atomic write).

Usage:
    streamlit run experiments/20260520_v8_500_validation/04_resolution_app.py
"""
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "validation_sample.parquet"
ANNOTATIONS_DIR = ROOT / "annotations"
OMAR_PATH = ANNOTATIONS_DIR / "omar.jsonl"
MANDIRA_PATH = ANNOTATIONS_DIR / "mandira.jsonl"
CONSENSUS_PATH = ANNOTATIONS_DIR / "consensus.jsonl"

STANCE_OPTIONS = ["for", "against", "both", "irrelevant"]
HOSTILE_SUBS = [
    ("dominative_paternalism", "Dominative paternalism",
     "Women are incompetent; men must rule, decide, control."),
    ("competitive_gender_differentiation", "Competitive gender differentiation",
     "Men have traits women lack (rationality, judgment, courage)."),
    ("heterosexual_hostility", "Heterosexual hostility",
     "Women's sexuality framed as manipulative, dangerous, threatening."),
]
BENEVOLENT_SUBS = [
    ("protective_paternalism", "Protective paternalism",
     "Women are fragile; men should protect, provide, shield."),
    ("complementary_gender_differentiation", "Complementary gender differentiation",
     "Women have special purity / nurturance / moral nature."),
    ("heterosexual_intimacy", "Heterosexual intimacy",
     "Men are incomplete without women; women complete men."),
]

# --- Keyword highlighting (duplicated from 02_annotation_app.py) ---
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


def _r(p):
    return re.compile(p, re.IGNORECASE)


TIER1_ALT_PIVOTS = [
    (_r(r"women.*suffrage"),                          [_r(r"\bwomen\b"), _r(r"\bsuffrage\w*\b")]),
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
TIER2_VOTING_SUBSTRINGS = ("vote", "voting", "voter", "voters", "electoral", "electorate",
                            "franchise", "enfranchise", "representation")
TIER2_WINDOW_WORDS = 25


# --------------------------------------------------------------------------- #
# Pure helpers (highlight + render)
# --------------------------------------------------------------------------- #

def _word_positions(text):
    out, i, n = [], 0, len(text)
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


def find_tier1_spans(text):
    spans = []
    for alt_regex, pivot_regexes in TIER1_ALT_PIVOTS:
        for m in alt_regex.finditer(text):
            window = text[m.start():m.end()]
            for pr in pivot_regexes:
                for pm in pr.finditer(window):
                    spans.append((m.start() + pm.start(), m.start() + pm.end()))
    return spans


def find_tier2_spans(text):
    words = _word_positions(text)
    if not words:
        return []
    gender_idx = [i for i, (_, _, w) in enumerate(words)
                  if "women" in w or "female" in w]
    out, seen = [], set()
    for gi in gender_idx:
        sw = max(0, gi - TIER2_WINDOW_WORDS)
        ew = min(len(words), gi + TIER2_WINDOW_WORDS)
        voting_hits = [wi for wi in range(sw, ew)
                       if wi != gi and any(sub in words[wi][2]
                                            for sub in TIER2_VOTING_SUBSTRINGS)]
        if not voting_hits:
            continue
        s, e, _ = words[gi]
        out.append((s, e))
        for wi in voting_hits:
            if wi in seen:
                continue
            seen.add(wi)
            s, e, _ = words[wi]
            out.append((s, e))
    return out


def find_keyword_spans(text):
    qualifies_tier1 = bool(TIER1_QUALIFY_REGEX.search(text))
    t1 = find_tier1_spans(text)
    t2 = find_tier2_spans(text)
    if qualifies_tier1:
        raw = [(s, e, "tier1") for s, e in t1] + [(s, e, "tier2") for s, e in t2]
    elif t2:
        raw = [(s, e, "tier2") for s, e in t2]
    else:
        raw = []
    raw.sort(key=lambda s: (s[0], 0 if s[2] == "tier1" else 1))
    merged = []
    for s, e, k in raw:
        if merged and s < merged[-1][1]:
            ps, pe, pk = merged[-1]
            if pk == k:
                merged[-1] = (ps, max(pe, e), pk)
            continue
        merged.append((s, e, k))
    return merged


def _escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("\n", "<br>"))


def render_highlighted_html(text, spans):
    pieces, cursor = [], 0
    for start, end, kind in spans:
        if start < cursor:
            continue
        pieces.append(_escape(text[cursor:start]))
        cls = "kw-tier1" if kind == "tier1" else "kw-tier2"
        pieces.append(f'<mark class="{cls}">{_escape(text[start:end])}</mark>')
        cursor = end
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces)


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z—‘“])")


def extract_keyword_sentences(text, spans):
    if not text or not spans:
        return ""
    starts = [0] + [m.end() for m in _SENTENCE_BREAK.finditer(text)] + [len(text)]
    sent_ranges = list(zip(starts[:-1], starts[1:]))
    hits = [(sa, sb) for sa, sb in sent_ranges
            if any(sa <= s < sb for s, _, _ in spans)]
    if not hits:
        return ""
    merged = []
    for sa, sb in hits:
        if merged and sa == merged[-1][1]:
            merged[-1] = (merged[-1][0], sb)
        else:
            merged.append((sa, sb))
    chunks = []
    for sa, sb in merged:
        sub_text = text[sa:sb].strip()
        sub_spans = [(s - sa, e - sa, k) for s, e, k in spans if sa <= s < sb]
        chunks.append(render_highlighted_html(sub_text, sub_spans))
    return ' <span style="opacity:0.5">...</span> '.join(chunks)


def _context_list(value):
    if value is None:
        return []
    try:
        out = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "items"):
                out.append({k: item[k] for k in item})
        return out
    except TypeError:
        return []


def derive(rec):
    rec["hostile"] = bool(rec.get("hostile_subcategories"))
    rec["benevolent"] = bool(rec.get("benevolent_subcategories"))
    rec["sexist"] = rec["hostile"] or rec["benevolent"]
    return rec


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def load_jsonl(path):
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


def save_jsonl(path, records):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    )
    try:
        for sid in sorted(records):
            tmp.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
        tmp.flush()
        Path(tmp.name).replace(path)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise


@st.cache_data(show_spinner=False)
def load_sample():
    df = pd.read_parquet(SAMPLE_PATH)
    return df.drop(columns=["stance", "binary", "axis_a_label", "confidence"],
                   errors="ignore")


# --------------------------------------------------------------------------- #
# Disagreement detection
# --------------------------------------------------------------------------- #

def is_disagreement(o, m):
    return (o["stance"] != m["stance"]
            or o["hostile"] != m["hostile"]
            or o["benevolent"] != m["benevolent"]
            or set(o["hostile_subcategories"]) != set(m["hostile_subcategories"])
            or set(o["benevolent_subcategories"]) != set(m["benevolent_subcategories"]))


def disagreement_summary(o, m):
    parts = []
    if o["stance"] != m["stance"]:
        parts.append(f"stance: O={o['stance']} | M={m['stance']}")
    if o["hostile"] != m["hostile"] or set(o["hostile_subcategories"]) != set(m["hostile_subcategories"]):
        parts.append(f"hostile: O={o['hostile_subcategories'] or 'no'} | M={m['hostile_subcategories'] or 'no'}")
    if o["benevolent"] != m["benevolent"] or set(o["benevolent_subcategories"]) != set(m["benevolent_subcategories"]):
        parts.append(f"benevolent: O={o['benevolent_subcategories'] or 'no'} | M={m['benevolent_subcategories'] or 'no'}")
    return parts


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="V8 Resolution", layout="wide")
st.markdown(
    """
    <style>
      .speech-body { font-size: 1.02rem; line-height: 1.55; color: inherit; }
      .speech-body p { margin: 0 0 0.6em 0; }
      .meta-pill { display:inline-block; padding:2px 8px; border-radius:10px;
                   background: rgba(128,128,128,0.18); color: inherit;
                   font-size:0.78rem; margin-right:6px; }
      .ctx-body { font-size: 0.95rem; color: inherit; opacity: 0.92; }
      mark.kw-tier1 { background: rgba(252,211,77,0.55); color: inherit;
                       padding:0 2px; border-radius:2px; }
      mark.kw-tier2 { background: rgba(96,165,250,0.45); color: inherit;
                       padding:0 2px; border-radius:2px; }
      .ann-card { border-radius: 6px; padding: 8px 12px; margin-bottom: 4px;
                   background: rgba(128,128,128,0.08); }
      .ann-card.diff { background: rgba(239,68,68,0.10); border-left: 3px solid #ef4444; }
      .ann-card h5 { margin: 0 0 4px 0; font-size: 0.85rem; opacity: 0.7;
                      text-transform: uppercase; letter-spacing: 0.5px; }
      .ann-card p { margin: 0; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Session bootstrap
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "unresolved"

omar = load_jsonl(OMAR_PATH)
mandira = load_jsonl(MANDIRA_PATH)
consensus = load_jsonl(CONSENSUS_PATH)
sample = load_sample()

# Compute ordered list of disagreement speech_ids (in sample_idx order)
common = set(omar) & set(mandira)
disagree_ids = []
sample_by_id = {row["speech_id"]: row for _, row in sample.iterrows()}
for _, row in sample.iterrows():
    sid = row["speech_id"]
    if sid in common and is_disagreement(omar[sid], mandira[sid]):
        disagree_ids.append(sid)
N = len(disagree_ids)


# --------- Sidebar --------- #

with st.sidebar:
    st.markdown("### Resolution")
    n_resolved = sum(1 for sid in disagree_ids if sid in consensus)
    st.metric("Resolved", f"{n_resolved}/{N}")
    st.progress(n_resolved / N if N else 1.0)

    st.session_state.filter_mode = st.radio(
        "Show", ["unresolved", "all"],
        index=["unresolved", "all"].index(st.session_state.filter_mode),
        horizontal=True,
    )
    visible = [
        i for i, sid in enumerate(disagree_ids)
        if st.session_state.filter_mode == "all" or sid not in consensus
    ]
    if not visible:
        st.success("All disagreements resolved!")
        st.stop()
    if st.session_state.idx not in visible:
        st.session_state.idx = visible[0]
    pos = visible.index(st.session_state.idx)

    nav = st.columns([1, 1, 1, 1])
    if nav[0].button("First", use_container_width=True, disabled=pos == 0):
        st.session_state.idx = visible[0]; st.rerun()
    if nav[1].button("Prev", use_container_width=True, disabled=pos == 0):
        st.session_state.idx = visible[pos - 1]; st.rerun()
    if nav[2].button("Next", use_container_width=True, disabled=pos == len(visible) - 1):
        st.session_state.idx = visible[pos + 1]; st.rerun()
    if nav[3].button("Last", use_container_width=True, disabled=pos == len(visible) - 1):
        st.session_state.idx = visible[-1]; st.rerun()

    st.caption(f"Disagreement {pos + 1} of {len(visible)} (in current filter); "
               f"overall idx {st.session_state.idx + 1}/{N}")

    with st.expander("Notes from annotators"):
        sid = disagree_ids[st.session_state.idx]
        for who, rec in (("Omar", omar[sid]), ("Mandira", mandira[sid])):
            note = rec.get("notes", "").strip()
            flag = " [FLAGGED]" if rec.get("flagged") else ""
            if note or flag:
                st.markdown(f"**{who}{flag}**: {note or '(no note)'}")
            else:
                st.caption(f"{who}: -")


# --------- Main panel --------- #

sid = disagree_ids[st.session_state.idx]
o_rec = omar[sid]
m_rec = mandira[sid]
c_rec = consensus.get(sid)
row = sample_by_id[sid]
text = row["target_text"] or ""

# Header
st.markdown(
    f"### {row['speaker']} ({int(row['year'])})  "
    f"<span class='meta-pill'>sample {int(row['sample_idx']) + 1}/{len(sample)}</span>"
    f"<span class='meta-pill'>{int(row['word_count'])} words</span>"
    + (f"<span class='meta-pill' style='background:rgba(34,197,94,0.18);color:#22c55e'>"
       f"resolved</span>" if c_rec else
       "<span class='meta-pill' style='background:rgba(239,68,68,0.18);color:#ef4444'>"
       "unresolved</span>"),
    unsafe_allow_html=True,
)
st.caption(f"speech_id: `{sid}`")

# Disagreement summary
diffs = disagreement_summary(o_rec, m_rec)
if diffs:
    st.markdown("**Disagreement:**  " + "  -  ".join(diffs))

st.divider()

# Speech body with quick-view + full text
spans = find_keyword_spans(text)
qv_html = extract_keyword_sentences(text, spans)
if qv_html:
    st.markdown(
        f"<div style='background:rgba(128,128,128,0.08);border-left:3px solid "
        f"rgba(252,211,77,0.7);padding:10px 14px;border-radius:4px;margin-bottom:14px'>"
        f"<div style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;"
        f"opacity:0.7;margin-bottom:6px'>Keyword sentences (skim first)</div>"
        f"<div class='speech-body' style='font-size:0.96rem'>{qv_html}</div></div>",
        unsafe_allow_html=True,
    )
with st.expander(f"Full speech text ({int(row['word_count'])} words)", expanded=False):
    html = render_highlighted_html(text, spans)
    st.markdown(f'<div class="speech-body">{html}</div>', unsafe_allow_html=True)

# Context
preceding = _context_list(row.get("preceding_speeches"))
following = _context_list(row.get("following_speeches"))
if preceding or following:
    with st.expander(f"Debate context  -  {len(preceding)} before / {len(following)} after",
                      expanded=False):
        if preceding:
            st.markdown(f"**Preceding** ({len(preceding)})")
            for sp in preceding:
                with st.expander(f"seq {sp.get('sequence_number')}  -  {sp.get('speaker')}  -  "
                                  f"{sp.get('word_count')} words", expanded=False):
                    st.markdown(f"<div class='speech-body ctx-body'>"
                                 f"{_escape(sp.get('text', '') or '')}</div>",
                                 unsafe_allow_html=True)
        st.markdown("---")
        if following:
            st.markdown(f"**Following** ({len(following)})")
            for sp in following:
                with st.expander(f"seq {sp.get('sequence_number')}  -  {sp.get('speaker')}  -  "
                                  f"{sp.get('word_count')} words", expanded=False):
                    st.markdown(f"<div class='speech-body ctx-body'>"
                                 f"{_escape(sp.get('text', '') or '')}</div>",
                                 unsafe_allow_html=True)

st.divider()


# --------- Side-by-side comparison --------- #

def fmt_subs(subs):
    return ", ".join(subs) if subs else "_none_"


def card(title, rec, diff_keys, key=None):
    """Render an annotator's labels in a card. `diff_keys` controls which
    fields get the 'diff' background tint."""
    stance_diff = "stance" in diff_keys
    hostile_diff = "hostile" in diff_keys
    benev_diff = "benevolent" in diff_keys
    bg = "rgba(128,128,128,0.08)"
    st.markdown(f"<div class='ann-card'><h5>{title}</h5>"
                 f"<p><b>Stance:</b> "
                 f"<span style='{'background:rgba(239,68,68,0.2);padding:1px 4px' if stance_diff else ''}'>"
                 f"{rec['stance']}</span></p>"
                 f"<p><b>Hostile:</b> "
                 f"<span style='{'background:rgba(239,68,68,0.2);padding:1px 4px' if hostile_diff else ''}'>"
                 f"{fmt_subs(rec['hostile_subcategories'])}</span></p>"
                 f"<p><b>Benevolent:</b> "
                 f"<span style='{'background:rgba(239,68,68,0.2);padding:1px 4px' if benev_diff else ''}'>"
                 f"{fmt_subs(rec['benevolent_subcategories'])}</span></p>"
                 f"</div>",
                 unsafe_allow_html=True)


diff_keys = set()
if o_rec["stance"] != m_rec["stance"]:
    diff_keys.add("stance")
if (o_rec["hostile"] != m_rec["hostile"]
    or set(o_rec["hostile_subcategories"]) != set(m_rec["hostile_subcategories"])):
    diff_keys.add("hostile")
if (o_rec["benevolent"] != m_rec["benevolent"]
    or set(o_rec["benevolent_subcategories"]) != set(m_rec["benevolent_subcategories"])):
    diff_keys.add("benevolent")

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    card("Omar", o_rec, diff_keys)
with c2:
    card("Mandira", m_rec, diff_keys)
with c3:
    st.markdown("<div class='ann-card' style='background:rgba(34,197,94,0.08);"
                 "border-left:3px solid #22c55e'><h5>Consensus</h5></div>",
                 unsafe_allow_html=True)


# --------- Consensus form --------- #

# Initialise consensus from existing record, or default to Omar's labels
if c_rec is None:
    initial = {
        "stance": o_rec["stance"],
        "hostile_subcategories": list(o_rec["hostile_subcategories"]),
        "benevolent_subcategories": list(o_rec["benevolent_subcategories"]),
    }
else:
    initial = c_rec

# Quick-fill buttons
fill_cols = st.columns(4)
def _set_consensus(stance, hostile_subs, benevolent_subs):
    rec = {
        "speech_id": sid,
        "sample_idx": int(row["sample_idx"]),
        "stance": stance,
        "hostile_subcategories": sorted(hostile_subs),
        "benevolent_subcategories": sorted(benevolent_subs),
        "notes": (c_rec or {}).get("notes", ""),
        "flagged": False,
        "annotator": "consensus",
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
    }
    derive(rec)
    consensus[sid] = rec
    save_jsonl(CONSENSUS_PATH, consensus)
    # Bump widget keys so the form re-renders with the new values
    st.session_state["form_nonce_" + sid] = st.session_state.get("form_nonce_" + sid, 0) + 1


if fill_cols[0].button("Use Omar's", use_container_width=True):
    _set_consensus(o_rec["stance"], o_rec["hostile_subcategories"],
                   o_rec["benevolent_subcategories"])
    st.rerun()
if fill_cols[1].button("Use Mandira's", use_container_width=True):
    _set_consensus(m_rec["stance"], m_rec["hostile_subcategories"],
                   m_rec["benevolent_subcategories"])
    st.rerun()
if fill_cols[2].button("Mark irrelevant", use_container_width=True):
    _set_consensus("irrelevant", [], [])
    st.rerun()
if fill_cols[3].button("Mark not sexist (keep stance)",
                        use_container_width=True,
                        disabled=initial["stance"] is None):
    _set_consensus(initial["stance"], [], [])
    st.rerun()

st.markdown("**Or pick consensus manually:**")

nonce = st.session_state.get("form_nonce_" + sid, 0)
stance_idx = (STANCE_OPTIONS.index(initial["stance"])
              if initial["stance"] in STANCE_OPTIONS else None)
new_stance = st.radio(
    "Stance", STANCE_OPTIONS, index=stance_idx, horizontal=True,
    key=f"stance_{sid}_{nonce}",
)

hc, bc = st.columns(2)
new_h = set()
with hc:
    st.markdown("**Hostile sexism**")
    for key, label, desc in HOSTILE_SUBS:
        if st.checkbox(label, value=key in initial["hostile_subcategories"],
                        help=desc, key=f"h_{key}_{sid}_{nonce}"):
            new_h.add(key)
new_b = set()
with bc:
    st.markdown("**Benevolent sexism**")
    for key, label, desc in BENEVOLENT_SUBS:
        if st.checkbox(label, value=key in initial["benevolent_subcategories"],
                        help=desc, key=f"b_{key}_{sid}_{nonce}"):
            new_b.add(key)

# Save on change. Compare against the form's initial values -- if nothing
# changed since render, this is just a passive page load and we must NOT
# commit a phantom consensus (otherwise navigating through unresolved items
# would silently auto-save Omar's defaults everywhere).
prospective = {
    "stance": new_stance,
    "hostile_subcategories": sorted(new_h),
    "benevolent_subcategories": sorted(new_b),
}
initial_form = {
    "stance": initial["stance"],
    "hostile_subcategories": sorted(initial["hostile_subcategories"]),
    "benevolent_subcategories": sorted(initial["benevolent_subcategories"]),
}
if prospective != initial_form and new_stance is not None:
    _set_consensus(new_stance, new_h, new_b)

# Notes (saved on change, doesn't drive rerun)
notes_val = (consensus.get(sid) or {}).get("notes", "")
new_notes = st.text_area("Notes (optional, attaches to consensus)",
                          value=notes_val, height=68, key=f"notes_{sid}_{nonce}")
if new_notes != notes_val and sid in consensus:
    consensus[sid]["notes"] = new_notes
    save_jsonl(CONSENSUS_PATH, consensus)

# Save & Next
status_col, next_col = st.columns([3, 1])
with status_col:
    if c_rec := consensus.get(sid):
        st.caption(f"Saved {c_rec['annotated_at']}  --  "
                   f"stance: {c_rec['stance']}; "
                   f"hostile: {fmt_subs(c_rec['hostile_subcategories'])}; "
                   f"benevolent: {fmt_subs(c_rec['benevolent_subcategories'])}")
    else:
        st.caption("Set stance to save a consensus.")
with next_col:
    if st.button("Next  ->", type="primary", use_container_width=True,
                  disabled=pos >= len(visible) - 1):
        # Bounds guard: visible may have shrunk between render and click if a
        # commit just landed and filter mode is "unresolved".
        next_pos = min(pos + 1, len(visible) - 1)
        st.session_state.idx = visible[next_pos]
        st.rerun()
