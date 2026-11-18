import io,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_reviewer1_comment3 import fixture

class UI(unittest.TestCase):
 def test_separate_page_opens(self):
  from streamlit.testing.v1 import AppTest
  p=Path(__file__).resolve().parents[1]/'pages/3_Reviewer1_Comment3.py'
  at=AppTest.from_file(str(p),default_timeout=60).run()
  self.assertFalse(at.exception)
  self.assertEqual(len(at.get('file_uploader')),1)

 def test_downloads_bound_to_current_input_and_configuration(self):
  from streamlit.testing.v1 import AppTest
  buf=io.BytesIO();fixture().to_excel(buf,index=False,sheet_name='Raw')
  script='''
import io
import streamlit as st
from reviewer1_comment3 import render_panel
class Upload(io.BytesIO):name='fixture.xlsx'
data=st.session_state.get('fixture_bytes',DEFAULT_BYTES)
original_uploader=st.file_uploader
try:
    st.file_uploader=lambda *args,**kwargs:Upload(data)
    render_panel(st)
finally:
    st.file_uploader=original_uploader
'''.replace('DEFAULT_BYTES',repr(buf.getvalue()))
  at=AppTest.from_string(script,default_timeout=60).run()
  at.number_input[0].set_value(4).run();at.number_input[1].set_value(2).run()
  at.multiselect[1].set_value(['Logistic Regression']).run()
  at.button[0].click().run(timeout=60)
  self.assertFalse(at.exception);self.assertEqual(len(at.get('download_button')),2)
  at.number_input[-1].set_value(43).run();self.assertEqual(len(at.get('download_button')),0)
  at.number_input[-1].set_value(42).run();self.assertEqual(len(at.get('download_button')),2)
  at.session_state['fixture_bytes']=b'different workbook';at.run()
  self.assertEqual(len(at.get('download_button')),0)
  at.button[0].click().run();self.assertTrue(at.error)
  self.assertEqual(len(at.get('download_button')),0)

if __name__=='__main__':unittest.main()
