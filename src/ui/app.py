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

# Sidebar for User Input:
with st.sidebar:
    st.header('Query Engine')
    user_query= st.text_area('Ask a business question:',
                             placeholder= 'e.g., What is our total revenue by product category?')
    run_button= st.button('Generate Insights',
                          type= 'primary')

# Main Execution Logic:
if run_button and user_query:
    with st.spinner('Analyzing request and generating SQL..'):

        # Calling SQL Engine:
        result= execute_text_to_sql(user_query)
        df = result.get('dataframe')
        metadata= result.get('metadata')

        # Displaying Result and Chart:
        if df is not None and not df.empty and metadata is not None:
            # Displaying Generated SQL for Varification:
            with st.expander('🔍 View AI-Generated SQL', expanded= False):
                st.code(metadata.sql_query, language= 'sql')
                st.caption(f'Chart logic selected: {metadata.chart_type.upper()} | X: {metadata.x_axis} | Y: {metadata.y_axis}')
