"""HeatPOD Trend Detective: Streamlit entry point.

A guided investigation in five steps. The app only reads data/results/
(precomputed by the pipeline); it never calls the NASA API.
"""

from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="HeatPOD Trend Detective", layout="wide")
st.markdown(f"<style>{(ROOT / 'app' / 'styles.css').read_text()}</style>", unsafe_allow_html=True)

from app import data_loader as dl  # noqa: E402  (after set_page_config)
from app.components import stripes  # noqa: E402
from app.views import Case, evidence, extremes, patterns, what, where  # noqa: E402

STEPS = {
    "what": ("1  What", what),
    "where": ("2  Where", where),
    "evidence": ("3  Evidence", evidence),
    "patterns": ("4  Patterns", patterns),
    "extremes": ("5  Extremes", extremes),
}
ORDER = list(STEPS)
DEMO = {"region": "florida", "variable": "HEAT_INDEX"}

ss = st.session_state
ss.setdefault("region", DEMO["region"])
ss.setdefault("variable", "T2M")
ss.setdefault("step", "what")
ss.setdefault("demo", False)


# --- demo callbacks (run before the next rerun, so widget state can change) ---

def start_demo() -> None:
    ss.region, ss.variable, ss.step, ss.demo = DEMO["region"], DEMO["variable"], "what", True


def move(delta: int) -> None:
    i = min(max(ORDER.index(ss.step) + delta, 0), len(ORDER) - 1)
    ss.step = ORDER[i]
    # The demo must end on a variable that has an extremes analysis.
    if ss.step == "extremes" and not catalog[ss.variable].extremes:
        ss.variable = DEMO["variable"]


def end_demo() -> None:
    ss.demo = False


index = dl.index()
catalog = dl.catalog()

# Links like ?region=phoenix&variable=T2M&step=evidence open a specific case.
if "from_link" not in ss:
    qp = st.query_params
    if qp.get("region") in index["regions"]:
        ss.region = qp["region"]
    if qp.get("variable") in catalog.variables:
        ss.variable = qp["variable"]
    if qp.get("step") in STEPS:
        ss.step = qp["step"]
    ss.from_link = True

# --- sidebar: the case steps and the pickers ----------------------------------
with st.sidebar:
    st.markdown('<div class="case-label">Case steps</div>', unsafe_allow_html=True)
    st.radio("Step", ORDER, format_func=lambda k: STEPS[k][0], key="step",
             label_visibility="collapsed")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.selectbox("Region", list(index["regions"]), key="region",
                 format_func=lambda k: dl.region(k)["name"])
    st.selectbox("Variable", list(catalog.variables), key="variable",
                 format_func=lambda c: catalog[c].name)
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.button("Watch a demo", on_click=start_demo, width="stretch")
    st.caption("Data: NASA POWER daily (MERRA-2), 1981 to 2025, about 55 km grid.")

st.query_params.update(region=ss.region, variable=ss.variable, step=ss.step)
case = Case(ss.region, dl.region(ss.region), catalog[ss.variable], dl.summary(ss.region, ss.variable))

# --- header: title, case line, warming stripes ---------------------------------

st.markdown('<div class="case-label">Case file / NASA Space Apps 2026</div>', unsafe_allow_html=True)
st.title("HeatPOD Trend Detective", anchor=False)
st.markdown(f'<div class="case-line mono">Case: {escape(case.region_info["name"])}, '
            f'{escape(case.var.name.lower())}</div>', unsafe_allow_html=True)
stripes.show(case.region, case.code, case.var.unit)
st.caption(f"Each bar is one year, {index['period']['start']} to {index['period']['end']}: "
           "blue below normal, red above normal (normal = 1991 to 2020).")

# --- demo controls ---------------------------------------------------------------

DEMO_NOTES = {
    "what": "Demo case: Florida's heat index, the heat people actually feel in a humid place.",
    "where": "We study the land squares of Florida. The sea squares are removed.",
    "evidence": "The verdict box gives the answer to question 1, with the test behind it.",
    "patterns": "Every square warms, but not equally. Pattern 2 splits the north from the south peninsula.",
    "extremes": "Question 3: are the hottest days becoming more common?",
}
if ss.demo:
    i = ORDER.index(ss.step)
    st.markdown(f'<div class="demo-bar"><span class="case-label">Demo, step {i + 1} of 5</span>'
                f"<p>{escape(DEMO_NOTES[ss.step])}</p></div>", unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("Back", on_click=move, args=(-1,), disabled=i == 0, width="stretch")
    b2.button("Next step", on_click=move, args=(1,), disabled=i == len(ORDER) - 1,
              type="primary", width="stretch")
    b3.button("End demo", on_click=end_demo, width="stretch")

# --- main area: the step view and margin notes -------------------------------------

main, notes = st.columns([3, 1], gap="large")
with main:
    STEPS[ss.step][1].render(case)

with notes:
    g = dl.glossary()
    keys = [k for k in g["steps"][ss.step] if k != "heat_index" or case.code == "HEAT_INDEX"]
    st.markdown('<div class="case-label">Margin notes</div>', unsafe_allow_html=True)
    for k in keys:
        t = g["terms"][k]
        st.markdown(f'<div class="note"><div class="note-title">{escape(t["title"])}</div>'
                    f'<p>{escape(t["text"])}</p></div>', unsafe_allow_html=True)
