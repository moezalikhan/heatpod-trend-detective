"""Step 1. What are we investigating?"""

from html import escape

import streamlit as st

from app import data_loader as dl


def render(case) -> None:
    st.header("1. What are we investigating?", anchor=False)
    v = case.var
    st.markdown(f'<div class="case-label">Variable</div>', unsafe_allow_html=True)
    st.subheader(f"{v.name} ({v.unit})", anchor=False)
    st.markdown(f"**What it is.** {v.definition}.")
    st.markdown(f"**Why it matters.** {v.why_it_matters}")
    if v.limitation:
        st.markdown(f"**Limitation.** {v.limitation}")
    st.markdown(
        "**The three questions.** Is it changing? Is it changing the same way "
        "everywhere? Are the worst days getting worse? The next steps answer them "
        "with 45 years of NASA data (1981 to 2025).")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<div class="case-label">All variables in this case file</div>', unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td class="mono">{escape(c)}</td><td>{escape(x.name)}</td>'
        f"<td>{escape(x.definition)}</td></tr>"
        for c, x in dl.catalog().variables.items())
    st.markdown(f'<table class="plain"><tr><th>Code</th><th>Variable</th><th>Plain meaning</th></tr>'
                f"{rows}</table>", unsafe_allow_html=True)
    st.caption("Pick a different variable in the sidebar.")
