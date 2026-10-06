"""Evaluation results (reports/results.md) with all figures."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _render import render_markdown  # noqa: E402

st.set_page_config(page_title="LPHRS results", page_icon="📊", layout="wide")
render_markdown(Path(__file__).resolve().parents[2] / "reports" / "results.md")
