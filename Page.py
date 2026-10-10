"""Streamlit UI for the multi-agent research pipeline (dark + orange).

Run from the project folder with:  streamlit run app.py
Requires .streamlit/config.toml (dark theme) next to this file.
"""

import html
import inspect
import re
import time
import traceback
from urllib.parse import urlparse

import streamlit as st

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Research assistant", page_icon="⚡", layout="centered")

# agents.py raises ValueError at import time if an API key is missing.
# Catch it so the user sees a readable message instead of a stack trace.
try:
    from pipeline import run_research_pipeline
except Exception as import_error:
    st.error(
        f"Could not start the research pipeline: {import_error}\n\n"
        "Check that your .env file has GOOGLE_API_KEY and TAVILY_API_KEY, "
        "then restart the app."
    )
    st.stop()

# If pipeline.py accepts an `on_step` callback, the console updates live.
_PARAMS = inspect.signature(run_research_pipeline).parameters
SUPPORTS_PROGRESS = "on_step" in _PARAMS
SUPPORTS_MODE = "mode" in _PARAMS

STAGES = [
    ("search", "Search", "Plans queries and searches the web"),
    ("read", "Read", "Gets page text for the best sources"),
    ("write", "Write", "Drafts the report from that research"),
    ("critique", "Review", "A critic checks the draft"),
]

EXAMPLE_TOPICS = [
    "Solid-state batteries",
    "How UPI changed small businesses",
    "Long context in language models",
]

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

MODE_HELP = {
    "Fast": "About 1 AI call. Searches and writes, no review.",
    "Balanced": "About 3 AI calls. Searches, writes and reviews.",
    "Thorough": "Up to 10 AI calls. An agent searches, and the draft is revised.",
}


# ---------- styling ----------
# Palette: void (warm black page), panel (surfaces), line (borders),
# text / muted, orange (brand, primary action, finished stages),
# amber (work in progress, links), red (errors).
# Type: Archivo ExtraBold for headlines and numbers (dense, industrial),
# Instrument Sans for everything you read.

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@700;800;900&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
  --void: #0B0908;
  --panel: #141110;
  --panel-2: #1D1917;
  --line: #2F2925;
  --text: #F3EDE7;
  --muted: #A59D95;
  --orange: #FF6A1A;
  --orange-hi: #FF8A4A;
  --amber: #FFB84D;
  --hot: #FFE6D0;
  --ink-on-orange: #1A0D04;
  --red: #FF4D5E;
  --display: 'Archivo', 'Instrument Sans', system-ui, sans-serif;
  --sans: 'Instrument Sans', system-ui, sans-serif;
}

/* page: warm black, one orange glow, a faint grid */
.stApp {
  background:
    radial-gradient(900px 520px at 12% -6%, rgba(255, 106, 26, 0.30), transparent 62%),
    radial-gradient(700px 420px at 100% 0%, rgba(255, 184, 77, 0.10), transparent 60%),
    linear-gradient(rgba(255, 255, 255, 0.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px),
    var(--void);
  background-size: auto, auto, 56px 56px, 56px 56px, auto;
  background-attachment: fixed;
  color: var(--text); font-family: var(--sans);
}
/* Streamlit paints its own opaque backgrounds on inner containers; clear them
   so the glow and grid above show through. */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {
  background: transparent !important;
}
::selection { background: rgba(255, 106, 26, 0.55); color: #fff; }
.block-container { max-width: 960px; padding-top: 3.5rem; padding-bottom: 6rem; }
footer, #MainMenu { visibility: hidden; }

/* hero */
.hero-wrap { display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 1.5rem; align-items: center; margin-bottom: 2rem; }
.hero {
  font-family: var(--display); font-weight: 800; color: #fff;
  font-size: clamp(2.3rem, 6vw, 4.1rem); line-height: 1.02; letter-spacing: -0.03em; margin: 0 0 1.2rem;
}
.hero .hot { color: var(--orange); }
.lede { color: var(--muted); font-size: 1.12rem; line-height: 1.6; max-width: 50ch; margin: 0; }
.art { width: 100%; height: auto; overflow: visible; }
.art .ring-a { fill: none; stroke: var(--line); stroke-dasharray: 2 7; }
.art .flow { fill: none; stroke: var(--orange); stroke-width: 2; stroke-dasharray: 6 9; stroke-linecap: round; }
.art .back { stroke: var(--amber); opacity: 0.8; }
.art .node circle { fill: var(--panel-2); stroke: var(--orange); stroke-width: 2.5; filter: drop-shadow(0 0 9px rgba(255, 106, 26, 0.75)); }
.art .node.core circle { fill: var(--orange); }
.art .node text { fill: var(--hot); font: 800 13px var(--display); text-anchor: middle; dominant-baseline: central; }
.art .node.core text { fill: var(--ink-on-orange); }
@media (prefers-reduced-motion: no-preference) {
  .art .flow { animation: dash 2.6s linear infinite; }
}
@keyframes dash { to { stroke-dashoffset: -60; } }
@media (max-width: 760px) {
  .hero-wrap { grid-template-columns: 1fr; }
  .art { display: none; }
}

/* command bar */
[data-testid="stForm"] { border: none; padding: 0; background: transparent; }
[data-testid="stTextInput"] div[data-baseweb="input"] {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
  border-color: var(--orange);
  box-shadow: 0 0 0 3px rgba(255, 106, 26, 0.25), 0 0 40px rgba(255, 106, 26, 0.28);
}
[data-testid="stTextInput"] input {
  font-family: var(--sans); font-size: 1.1rem; padding: 0.95rem 1.1rem;
  color: var(--text); caret-color: var(--orange); background: transparent;
}
[data-testid="stTextInput"] input::placeholder { color: #6E665F; }

[data-testid="stFormSubmitButton"] button {
  background: var(--orange) !important; border: none !important; color: var(--ink-on-orange) !important;
  font-family: var(--sans); font-weight: 700; border-radius: 12px; min-height: 3.35rem;
  box-shadow: 0 12px 30px -12px rgba(255, 106, 26, 0.95);
  transition: transform 0.12s, background 0.12s, box-shadow 0.12s;
}
[data-testid="stFormSubmitButton"] button:hover {
  background: var(--orange-hi) !important; transform: translateY(-1px);
  box-shadow: 0 16px 36px -12px rgba(255, 106, 26, 1);
}
[data-testid="stDownloadButton"] button {
  background: transparent !important; color: var(--text) !important;
  border: 1px solid var(--line) !important; border-radius: 12px; font-weight: 600;
}
[data-testid="stDownloadButton"] button:hover { border-color: var(--orange) !important; color: var(--orange) !important; }
[data-testid="stDownloadButton"] { margin-top: 1.25rem; }

/* example chips */
[class*="st-key-ex"] button {
  background: transparent !important; border: 1px solid var(--line) !important; border-radius: 999px;
  color: var(--muted) !important; font-family: var(--sans); font-size: 0.86rem;
  min-height: 0; padding: 0.35rem 0.95rem;
}
[class*="st-key-ex"] button:hover { border-color: var(--orange) !important; color: var(--orange) !important; }

/* pipeline console: the centrepiece */
.console {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border: 1px solid var(--line); border-radius: 16px;
  padding: 1.5rem 1.7rem 1.5rem; margin: 2.5rem 0;
  box-shadow: 0 34px 70px -40px rgba(255, 106, 26, 0.5);
}
.meter { height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; margin-bottom: 1.6rem; }
.meter span {
  display: block; height: 100%; background: linear-gradient(90deg, var(--orange), var(--amber));
  box-shadow: 0 0 14px var(--orange);
}
.track { list-style: none; display: flex; margin: 0; padding: 0; }
.track li { flex: 1; position: relative; padding-right: 1rem; }
.track li::before {
  content: ""; position: absolute; top: 14px; left: 42px; right: 10px; height: 2px; background: var(--line);
}
.track li:last-child::before { display: none; }
.track li.done::before { background: var(--orange); box-shadow: 0 0 10px rgba(255, 106, 26, 0.8); }
.dot {
  display: flex; align-items: center; justify-content: center; position: relative; z-index: 1;
  width: 30px; height: 30px; border-radius: 50%; border: 2px solid var(--line);
  background: var(--panel-2); color: var(--muted); font-size: 0.8rem; font-weight: 700;
}
.done .dot {
  background: var(--orange); border-color: var(--orange); color: var(--ink-on-orange);
  box-shadow: 0 0 22px rgba(255, 106, 26, 0.7);
}
.active .dot { border-color: var(--amber); background: var(--panel); }
.active .dot::after {
  content: ""; width: 10px; height: 10px; border-radius: 50%;
  background: var(--amber); box-shadow: 0 0 12px var(--amber);
}
.error .dot { background: var(--red); border-color: var(--red); color: #fff; }
.skipped .dot { border-style: dashed; background: transparent; }
.skipped .step-name { color: var(--muted); }
@media (prefers-reduced-motion: no-preference) {
  .active .dot { animation: pulse 1.5s ease-out infinite; }
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 184, 77, 0.55); }
  100% { box-shadow: 0 0 0 14px rgba(255, 184, 77, 0); }
}
.step-name { font-weight: 700; margin-top: 0.8rem; color: #fff; }
.idle .step-name { color: var(--muted); }
.step-desc { color: var(--muted); font-size: 0.88rem; line-height: 1.45; margin-top: 0.2rem; }
.step-time { font-size: 0.86rem; font-weight: 600; margin-top: 0.4rem; min-height: 1.2em; color: var(--orange-hi); }
.active .step-time { color: var(--amber); }
.error .step-time { color: var(--red); }
@media (max-width: 640px) {
  .track { flex-direction: column; gap: 1.4rem; }
  .track li::before { display: none; }
}

/* result summary */
.meta { color: var(--muted); font-size: 0.95rem; margin: 0 0 1.1rem; }

/* tabs */
[data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; background: transparent; }
button[data-baseweb="tab"][aria-selected="true"] { color: #fff; }
[data-baseweb="tab-highlight"] { background: var(--orange) !important; height: 3px; }

/* the report */
.st-key-report_sheet {
  background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--orange);
  border-radius: 16px; padding: 2.4rem 2.6rem;
  box-shadow: 0 40px 70px -40px rgba(0, 0, 0, 0.95), 0 0 0 1px rgba(255, 106, 26, 0.04);
}
.st-key-report_sheet [data-testid="stMarkdownContainer"] p,
.st-key-report_sheet [data-testid="stMarkdownContainer"] li {
  font-family: var(--sans); font-size: 1.06rem; line-height: 1.78; color: #DDD6CF;
}
.st-key-report_sheet [data-testid="stMarkdownContainer"] strong { color: #fff; }
.st-key-report_sheet [data-testid="stMarkdownContainer"] h1,
.st-key-report_sheet [data-testid="stMarkdownContainer"] h2,
.st-key-report_sheet [data-testid="stMarkdownContainer"] h3 {
  font-family: var(--display); font-weight: 800; color: #fff; letter-spacing: -0.02em;
}
.st-key-report_sheet [data-testid="stMarkdownContainer"] h1 { font-size: 1.9rem; }
.st-key-report_sheet [data-testid="stMarkdownContainer"] h2 { font-size: 1.45rem; margin-top: 2rem; }
.st-key-report_sheet [data-testid="stMarkdownContainer"] h2::before {
  content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 0.65rem;
  background: var(--orange); border-radius: 2px; transform: translateY(-2px);
}
.st-key-report_sheet [data-testid="stMarkdownContainer"] h3 { font-size: 1.1rem; }
.st-key-report_sheet [data-testid="stMarkdownContainer"] a { color: var(--amber); }
@media (max-width: 640px) { .st-key-report_sheet { padding: 1.3rem; } }

/* sources */
.src { display: flex; align-items: center; gap: 1rem; padding: 0.95rem 0; border-bottom: 1px solid var(--line); }
.tile {
  flex: none; width: 40px; height: 40px; border-radius: 10px; display: grid; place-items: center;
  background: rgba(255, 106, 26, 0.12); border: 1px solid rgba(255, 106, 26, 0.35);
  color: var(--orange-hi); font-family: var(--display); font-weight: 800; font-size: 1.05rem;
}
.src a { color: #fff; font-weight: 700; text-decoration: none; }
.src a:hover { color: var(--orange); }
.src .url { color: var(--muted); font-size: 0.85rem; overflow-wrap: anywhere; margin-top: 0.1rem; }

/* review gauge */
.gauge { position: relative; width: 150px; height: 150px; margin: 0.5rem 0 1.6rem; }
.ring {
  position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--orange) calc(var(--p) * 1%), var(--panel-2) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
  filter: drop-shadow(0 0 10px rgba(255, 106, 26, 0.55));
}
.gauge-num { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.gauge-num b { font-family: var(--display); font-weight: 800; font-size: 2.8rem; line-height: 1; color: #fff; }
.gauge-num span { color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }
"""

st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)

# Decorative graphic for the hero: the four agents as a connected loop, with
# the critic feeding back to the writer. One line, no blank lines (Markdown).
HERO_ART = (
    '<svg class="art" viewBox="0 0 320 300" aria-hidden="true">'
    '<defs><radialGradient id="glow" cx="50%" cy="50%" r="50%">'
    '<stop offset="0" stop-color="#FF6A1A" stop-opacity="0.45"/>'
    '<stop offset="1" stop-color="#FF6A1A" stop-opacity="0"/></radialGradient></defs>'
    '<circle cx="160" cy="150" r="145" fill="url(#glow)"/>'
    '<circle class="ring-a" cx="160" cy="150" r="105"/>'
    '<circle class="ring-a" cx="160" cy="150" r="62"/>'
    '<path class="flow" d="M70 72 L252 92 L238 228 L88 214"/>'
    '<path class="flow back" d="M88 214 Q165 150 238 228"/>'
    '<g class="node"><circle cx="70" cy="72" r="17"/><text x="70" y="72">1</text></g>'
    '<g class="node"><circle cx="252" cy="92" r="17"/><text x="252" y="92">2</text></g>'
    '<g class="node"><circle cx="238" cy="228" r="17"/><text x="238" y="228">3</text></g>'
    '<g class="node"><circle cx="88" cy="214" r="17"/><text x="88" y="214">4</text></g>'
    '<g class="node core"><circle cx="160" cy="150" r="13"/></g>'
    "</svg>"
)


# ---------- helpers ----------

def as_text(value) -> str:
    """Normalise model output to a string (Gemini can return content blocks)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in value
        )
    return str(value)


def blank_stages() -> dict:
    return {key: {"state": "idle", "secs": None, "round": 1} for key, _, _ in STAGES}


def tracker_html(stages: dict) -> str:
    # Built without blank lines or indentation: Markdown would otherwise
    # treat indented HTML as a code block.
    done = sum(1 for s in stages.values() if s["state"] in ("done", "skipped"))
    working = any(s["state"] == "active" for s in stages.values())
    percent = (done + (0.5 if working else 0)) / len(STAGES) * 100

    items = []
    for number, (key, name, description) in enumerate(STAGES, start=1):
        stage = stages[key]
        glyph = {"done": "✓", "active": "", "error": "!", "skipped": "–"}.get(stage["state"], str(number))
        rounds = stage.get("round", 1)
        unit = "drafts" if key == "write" else "rounds"
        if stage["state"] == "done":
            note = f"{stage['secs']:.0f}s" if stage["secs"] is not None else "Done"
            if rounds > 1:
                note += f", {rounds} {unit}"
        elif stage["state"] == "active":
            note = f"Draft {rounds}, working" if rounds > 1 and key in ("write", "critique") else "Working"
        elif stage["state"] == "error":
            note = "Failed"
        elif stage["state"] == "skipped":
            note = "Skipped"
        else:
            note = ""
        items.append(
            f'<li class="step {stage["state"]}"><span class="dot">{glyph}</span>'
            f'<div class="step-name">{name}</div>'
            f'<div class="step-desc">{description}</div>'
            f'<div class="step-time">{note}</div></li>'
        )
    return (
        '<div class="console">'
        f'<div class="meter"><span style="width:{percent:.0f}%"></span></div>'
        '<ol class="track">' + "".join(items) + "</ol></div>"
    )


def make_progress_callback(stages: dict, slot, started: dict):
    """Callback the pipeline calls as each stage starts and finishes."""

    def on_step(stage: str, event: str, data=None):
        if stage not in stages:
            return
        entry = stages[stage]
        if event == "skip":
            entry["state"] = "skipped"
            slot.markdown(tracker_html(stages), unsafe_allow_html=True)
            return
        if event == "start":
            entry["state"] = "active"
            started[stage] = time.perf_counter()
            if isinstance(data, dict) and data.get("round"):
                entry["round"] = data["round"]
                # A new draft means the review starts over.
                if stage == "write" and data["round"] > 1:
                    stages["critique"]["state"] = "idle"
        else:
            entry["state"] = "done"
            spent = time.perf_counter() - started.get(stage, time.perf_counter())
            entry["secs"] = (entry["secs"] or 0) + spent
        slot.markdown(tracker_html(stages), unsafe_allow_html=True)

    return on_step


def set_topic(topic: str):
    st.session_state["topic_input"] = topic


def file_name_for(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return f"{slug[:50] or 'report'}.md"


def extract_sources(text: str) -> list:
    urls = []
    for url in URL_RE.findall(text):
        url = url.rstrip(".,;")
        if url not in urls:
            urls.append(url)
    return urls


def sources_html(urls: list) -> str:
    rows = []
    for url in urls:
        domain = urlparse(url).netloc.removeprefix("www.") or url
        letter = html.escape(domain[:1].upper() or "?")
        rows.append(
            f'<div class="src"><span class="tile">{letter}</span><div>'
            f'<a href="{html.escape(url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{html.escape(domain)}</a>'
            f'<div class="url">{html.escape(url)}</div></div></div>'
        )
    return "".join(rows)


def score_html(score: float) -> str:
    percent = max(0, min(10, score)) * 10
    return (
        f'<div class="gauge"><div class="ring" style="--p:{percent}"></div>'
        f'<div class="gauge-num"><b>{score:g}</b><span>out of 10</span></div></div>'
    )


def render_critique(feedback):
    """Show the critic output, whether it is plain text or a structured object."""
    if feedback is None:
        st.info("The critic did not return a review.")
        return

    # Structured output (pydantic object with score, issues, ...)
    if hasattr(feedback, "score"):
        st.markdown(score_html(feedback.score), unsafe_allow_html=True)
        verdict = getattr(feedback, "verdict", "")
        strengths = getattr(feedback, "strengths", [])
        claims = getattr(feedback, "unsupported_claims", [])
        improvements = getattr(feedback, "improvements", [])
        if verdict:
            st.markdown(f"**{verdict}**")
        if strengths:
            st.markdown("**What works**")
            for item in strengths:
                st.markdown(f"- {item}")
        if claims:
            st.markdown("**Claims the research does not support**")
            for item in claims:
                st.markdown(f"- {item}")
        if improvements:
            st.markdown("**What to improve**")
            for item in improvements:
                st.markdown(f"- {item}")
        if not claims and not improvements:
            st.success("The critic found nothing to fix.")
        return

    # Plain text output: pull the score out if the critic wrote "Score: X/10"
    text = as_text(feedback)
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if match:
        st.markdown(score_html(float(match.group(1))), unsafe_allow_html=True)
        text = re.sub(r"(?im)^\s*\**score\**:.*$", "", text, count=1)
    st.markdown(text)


def run(topic: str, slot, mode: str = "balanced"):
    """Run the pipeline, update the console live, store the result."""
    st.session_state.pop("result", None)
    stages = blank_stages()
    st.session_state["stages"] = stages
    started = {}
    began = time.perf_counter()

    try:
        kwargs = {}
        if SUPPORTS_MODE:
            kwargs["mode"] = mode
        if SUPPORTS_PROGRESS:
            slot.markdown(tracker_html(stages), unsafe_allow_html=True)
            kwargs["on_step"] = make_progress_callback(stages, slot, started)
            state = run_research_pipeline(topic, **kwargs)
        else:
            stages["search"]["state"] = "active"
            slot.markdown(tracker_html(stages), unsafe_allow_html=True)
            state = run_research_pipeline(topic, **kwargs)
            for stage in stages.values():
                stage["state"] = "done"
    except Exception as error:
        for stage in stages.values():
            if stage["state"] == "active":
                stage["state"] = "error"
        slot.markdown(tracker_html(stages), unsafe_allow_html=True)
        st.error(
            f"The run stopped with an error: {error}\n\n"
            "If this mentions quota or rate limits, wait a minute and try again."
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return

    slot.markdown(tracker_html(stages), unsafe_allow_html=True)
    st.session_state["result"] = {
        "topic": topic,
        "state": state,
        "elapsed": time.perf_counter() - began,
    }


# ---------- page ----------

st.markdown(
    '<div class="hero-wrap"><div>'
    '<h1 class="hero">Four agents.<br><span class="hot">One sourced report.</span></h1>'
    '<p class="lede">Type a topic. Agents search the web and read the pages, '
    "a writer drafts the report, and a critic checks it for weak claims.</p>"
    f"</div>{HERO_ART}</div>",
    unsafe_allow_html=True,
)

with st.form("research_form"):
    input_column, button_column = st.columns([5, 1.6], vertical_alignment="bottom")
    with input_column:
        topic_value = st.text_input(
            "Research topic",
            key="topic_input",
            max_chars=200,
            placeholder="What do you want researched?",
            label_visibility="collapsed",
        )
    with button_column:
        submitted = st.form_submit_button(
            "Start research", type="primary", use_container_width=True
        )

mode_label = st.radio(
    "Speed",
    list(MODE_HELP),
    index=1,
    horizontal=True,
    key="mode_choice",
    label_visibility="collapsed",
)
st.caption(MODE_HELP[mode_label])

chip_columns = st.columns(len(EXAMPLE_TOPICS))
for index, (column, example) in enumerate(zip(chip_columns, EXAMPLE_TOPICS)):
    column.button(
        example,
        key=f"ex{index}",
        on_click=set_topic,
        args=(example,),
        use_container_width=True,
    )

tracker_slot = st.empty()
tracker_slot.markdown(
    tracker_html(st.session_state.get("stages") or blank_stages()),
    unsafe_allow_html=True,
)
if not SUPPORTS_PROGRESS:
    st.caption("Stage updates appear live once pipeline.py accepts an on_step callback.")

if submitted:
    if not topic_value.strip():
        st.warning("Enter a topic to start.")
    else:
        run(topic_value.strip(), tracker_slot, mode_label.lower())

# Results are drawn from session_state, not from the button press. Streamlit
# reruns the whole script on every interaction (including a download click),
# so anything not stored in session_state would disappear.
result = st.session_state.get("result")

if result:
    state = result["state"]
    report = as_text(state.get("report"))
    sources = extract_sources(as_text(state.get("search_results")))

    plural = "" if len(sources) == 1 else "s"
    st.markdown(
        f'<p class="meta">Finished in {result["elapsed"]:.0f} seconds'
        + (f" with {state['llm_calls']} AI calls" if state.get("llm_calls") else "")
        + f". The search found {len(sources)} source{plural}.</p>",
        unsafe_allow_html=True,
    )

    tab_report, tab_sources, tab_review = st.tabs(["Report", "Sources", "Review"])

    with tab_report:
        with st.container(key="report_sheet"):
            st.markdown(report or "_No report was produced._")
        if report:
            st.download_button(
                "Download report (.md)",
                data=report,
                file_name=file_name_for(result["topic"]),
                mime="text/markdown",
            )

    with tab_sources:
        if sources:
            st.markdown(sources_html(sources), unsafe_allow_html=True)
        else:
            st.info("No links were found in the search results.")
        with st.expander("Text the writer was given"):
            st.markdown("**Search findings**")
            st.text(as_text(state.get("search_results")) or "Nothing returned.")
            st.markdown("**Page content**")
            st.text(as_text(state.get("scraped_content")) or "Nothing returned.")

    with tab_review:
        history = state.get("history") or []
        if len(history) > 1:
            st.caption(
                "Revision history: "
                + ", ".join(f"draft {h['round']} scored {h['score']}/10" for h in history)
            )
        if state.get("mode") == "fast":
            st.info("Fast mode skips the review. Choose Balanced or Thorough to get one.")
        else:
            render_critique(state.get("critique") or state.get("feedback"))