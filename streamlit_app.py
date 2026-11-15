"""HeatPOD Trend Detective: placeholder entry point (Module 0)."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
STYLES = ROOT / "app" / "styles.css"

# Wide layout, set before any other Streamlit call.
st.set_page_config(page_title="HeatPOD Trend Detective", layout="wide")

# Load the small custom stylesheet once per run.
st.markdown(f"<style>{STYLES.read_text()}</style>", unsafe_allow_html=True)

# Heading uses Source Serif 4 and body uses IBM Plex Sans (theme fonts).
st.title("HeatPOD Trend Detective")
st.markdown("Coming soon")
