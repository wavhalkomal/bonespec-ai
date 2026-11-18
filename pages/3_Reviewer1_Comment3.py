"""Independent page; the original app.py is not modified."""
from pathlib import Path
import sys
import streamlit as st
root=str(Path(__file__).resolve().parents[1])
if root not in sys.path:sys.path.insert(0,root)
from reviewer1_comment3 import render_panel
st.set_page_config(page_title='Reviewer 1 Comment 3 SHAP stability',page_icon='🧭',layout='wide')
render_panel(st)
