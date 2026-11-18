"""Real Streamlit reruns using an in-memory synthetic workbook."""
import io
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_reviewer1_comment2 import fixture


class Comment2UI(unittest.TestCase):
    def test_separate_page_is_accessible_from_unchanged_working_app(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        at = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
        self.assertFalse(at.exception)
        at.switch_page("pages/2_Reviewer1_Comment2.py").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.number_input), 2)
        self.assertEqual(len(at.get("file_uploader")), 1)

    def test_completed_results_disappear_when_settings_or_input_changes(self):
        from streamlit.testing.v1 import AppTest
        buffer = io.BytesIO()
        fixture(6).to_excel(buffer, index=False, sheet_name="Raw")
        app = '''
import io
import streamlit as st
from reviewer1_comment2 import render_panel
class Upload(io.BytesIO):
    name = "synthetic.xlsx"
data = st.session_state.get("fixture_bytes", DEFAULT_BYTES)
render_panel(st, Upload(data))
'''.replace("DEFAULT_BYTES", repr(buffer.getvalue()))
        at = AppTest.from_string(app, default_timeout=60).run()
        at.sidebar.checkbox[0].check().run()
        at.number_input[0].set_value(6).run()
        at.multiselect[0].set_value(["SVM"]).run()
        at.button[0].click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("download_button")), 4)
        self.assertTrue(len(at.dataframe) >= 2)
        at.number_input[1].set_value(43).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("download_button")), 0)
        at.number_input[1].set_value(42).run()
        self.assertEqual(len(at.get("download_button")), 4)
        at.session_state["fixture_bytes"] = b"different workbook bytes"
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("download_button")), 0)

    def test_invalid_input_leaves_no_downloads(self):
        from streamlit.testing.v1 import AppTest
        buffer = io.BytesIO()
        bad = fixture(6); bad.loc[0, 1800] = float("nan")
        bad.to_excel(buffer, index=False, sheet_name="Raw")
        app = '''
import io
import streamlit as st
from reviewer1_comment2 import render_panel
class Upload(io.BytesIO):
    name = "invalid.xlsx"
render_panel(st, Upload(DEFAULT_BYTES))
'''.replace("DEFAULT_BYTES", repr(buffer.getvalue()))
        at = AppTest.from_string(app, default_timeout=60).run()
        at.sidebar.checkbox[0].check().run()
        at.number_input[0].set_value(6).run()
        at.button[0].click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertTrue(any("invalid/missing feature" in e.value for e in at.error))
        self.assertEqual(len(at.get("download_button")), 0)


if __name__ == "__main__":
    unittest.main()
