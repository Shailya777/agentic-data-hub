import sys
import os
import streamlit as st
import pandas as pd

# Adding Project Root to Python's Search Path:
#print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing Intent-Routing, SQL-Engine, RAG-Engine Module:
from src.agents.intent_router import route_query
from src.engines.sql_engine import execute_text_to_sql
from src.engines.rag_engine import execute_rag_query

# Page Config:
st.set_page_config(
    page_title= 'Agentic Data Hub',
    page_icon= '🧠',
    layout= 'wide',
)

st.title('📊 Agentic Data Intelligence Hub')
st.markdown('Ask natural language questions. The Intent Router will autonomously dispatch your query to the correct database engines (SQL, Vector, or both).')
st.divider()

# Sidebar for User Input:
with st.sidebar:
    st.header('Query Engine')
    user_query= st.text_area('Ask a business question:',
                             placeholder= 'e.g., What was our total revenue last quarter, and what are customers complaining about regarding delivery?')
    run_button= st.button('Generate Insights',
                          type= 'primary')

# Main Execution Logic:
if run_button and user_query:

    #Intent-Routing:
    with st.spinner('Analyzing Intent...'):
        routing_result= route_query(user_query= user_query)

    with st.expander("Router Logic", expanded= True):
        st.info(f"**Selected Engines:** {routing_result.engines}")
        st.caption(f"**Reasoning: {routing_result.reasoning}")

    st.divider()

    # Parallel Execution (Handling Multi-Intent if Necessary):

    # RAG Engine Trigger:
    if 'RAG_ENGINE' in routing_result.engines:
        st.subheader('Qualitative Insights (Customer Reviews)')
        with st.spinner('Querying Vector Database and Synthesizing...'):
            rag_response= execute_rag_query(user_query= user_query, n_results= 10)
            st.write(rag_response)
        st.divider()

    # SQL Engine Trigger:
    if 'SQL_ENGINE' in routing_result.engines:
        st.subheader('Quantitative Insights (Structured Data)')
        with st.spinner('Generating SQL and Executing...'):
            sql_result= execute_text_to_sql(user_query= user_query)
            df = sql_result.get('dataframe')
            metadata= sql_result.get('metadata')

            # Displaying Result and Chart:
            if df is not None and not df.empty and metadata is not None:
                # Displaying Generated SQL for Varification:
                with st.expander('🔍 View AI-Generated SQL', expanded= False):
                    st.code(metadata.sql_query, language= 'sql')
                    st.caption(f'Chart logic selected: {metadata.chart_type.upper()} | X: {metadata.x_axis} | Y: {metadata.y_axis}')

                # Rendering Result Dataframe:
                col1, col2= st.columns([1,2])

                with col1:
                    st.subheader('Results')
                    st.dataframe(data= df,
                             use_container_width= True)

                with col2:
                    st.subheader('Visualization')

                    # Deciding Chart Type based on Metadata:
                    if metadata.needs_chart and metadata.chart_type is not None:
                        try:
                            if metadata.chart_type == 'bar':
                                st.bar_chart(data= df, x= metadata.x_axis, y= metadata.y_axis)

                            elif metadata.chart_type == 'line':
                                st.line_chart(data= df, x= metadata.x_axis, y= metadata.y_axis)

                            elif metadata.chart_type == 'scatter':
                                st.scatter_chart(data= df, x= metadata.x_axis, y= metadata.y_axis)

                            else:
                                st.warning(f'Requested chart type {metadata.chart_type} not supported')

                        except Exception as e:
                            st.error(f'Failed to render chart. Please check axis mapping. Error: {e}')

                    else:
                        st.info('This Data does not require chart (e.g., single metric return).')

            else:
                st.error('The engine failed to return data. Check the terminal logs for database or self-correction errors.')

        st.divider()

    # Predictive Engine Trigger (Placeholder):
    if 'PREDICTIVE_ENGINE' in routing_result.engines:
        st.warning('🔮 **Predictive Engine:** Forecasting and Anomaly Detection are yet to be Implemented.')

    # Unknown Route:
    if 'UNKNOWN' in routing_result.engines:
        st.error("❓ I couldn't determine how to answer this question. Please ask a question related to e-commerce revenue or customer reviews.")