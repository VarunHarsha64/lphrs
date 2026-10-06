"""Render a local Markdown file in Streamlit, showing its ![...](path) images inline."""

import re
from pathlib import Path

import streamlit as st

IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def render_markdown(path: Path) -> None:
    if not path.exists():
        st.error(f"{path.name} not found. Run the scripts first (see README).")
        return
    text = path.read_text(encoding="utf-8")
    pos = 0
    for m in IMAGE.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            st.markdown(before)
        img = (path.parent / m.group(2)).resolve()
        if img.exists():
            st.image(str(img), caption=m.group(1), width="stretch")
        else:
            st.caption(f"(missing figure: {m.group(2)})")
        pos = m.end()
    if text[pos:].strip():
        st.markdown(text[pos:])
