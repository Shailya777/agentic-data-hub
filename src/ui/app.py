import sys
import os
import streamlit as st
import pandas as pd

# Adding Project Root to Python's Search Path:
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing SQL Engine Module:
from src.engines.sql_engine import execute_text_to_sql

# Page Config:
st.set_page_config(
    page_title= 'Agentic Data Hub',
    page_icon= '📊',
    layout= 'wide',
)

st.title('📊 Agentic Data Intelligence Hub')
st.markdown('Ask natural language questions about your e-commerce data. It will get converted to SQL, and you can verify the logic before trusting the results.')
st.divider()