"""Verdict box: flat ink border, stamp-style label, the key numbers in mono."""

from html import escape

import streamlit as st


def show(label: str, headline: str, details: list[str], sentence: str,
         caution: str | None = None) -> None:
    """Render a verdict box. `label` is SIGNIFICANT or NOT SIGNIFICANT."""
    muted = " verdict-muted" if label.startswith("NOT") else ""
    lines = "".join(f'<div class="verdict-detail mono">{escape(d)}</div>' for d in details)
    note = f'<div class="verdict-caution">{escape(caution)}</div>' if caution else ""
    st.markdown(
        f'<div class="verdict{muted}">'
        f'<div class="verdict-label">{escape(label)}</div>'
        f'<div class="verdict-headline mono">{escape(headline)}</div>'
        f"{lines}"
        f'<p class="verdict-sentence">{escape(sentence)}</p>'
        f"{note}</div>",
        unsafe_allow_html=True,
    )
