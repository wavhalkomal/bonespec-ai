"""Separate research-validation page; the working app.py is unchanged."""
from pathlib import Path
import sys

import streamlit as st

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from reviewer1_comment2 import render_panel

st.set_page_config(page_title="Reviewer 1 Comment 2 validation", page_icon="🧪", layout="wide")
st.session_state.setdefault("r1c2_show", True)
render_panel(st)
