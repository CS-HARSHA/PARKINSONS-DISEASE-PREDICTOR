"""Streamlit Components V2 wrapper for browser-side keystroke timing capture."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

_COMPONENT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def _register_component():
    """Register the frontend component once per Streamlit server process."""
    if not hasattr(st.components, "v2"):
        raise RuntimeError(
            "This app requires Streamlit 1.51 or newer for Components V2. "
            "Upgrade with: python -m pip install --upgrade 'streamlit>=1.51,<2'."
        )

    html = (_COMPONENT_DIR / "typing_capture.html").read_text(encoding="utf-8")
    css = (_COMPONENT_DIR / "typing_capture.css").read_text(encoding="utf-8")
    js = (_COMPONENT_DIR / "typing_capture.js").read_text(encoding="utf-8")

    return st.components.v2.component(
        name="parkinsons_keystroke_timing_capture",
        html=html,
        css=css,
        js=js,
        isolate_styles=True,
    )


def render_typing_capture(
    *,
    passage: str,
    minimum_count: int = 200,
    maximum_count: int = 1000,
    key: str = "typing_capture",
):
    """Mount the typing capture component and return its component state."""
    component = _register_component()
    return component(
        data={
            "passage": passage,
            "minimum_count": int(minimum_count),
            "maximum_count": int(maximum_count),
        },
        default={"capture_result": None},
        on_capture_result_change=lambda: None,
        key=key,
        width="stretch",
        height=680,
    )
