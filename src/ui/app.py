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
from src.engines.predictive.router import route_predictive_task
from src.engines.predictive.forecast_predictor import ForecastPredictor
from src.engines.predictive.rfm_predictor import RFMPredictor
from src.engines.predictive.delivery_predictor import predict_delivery_delay

# Importing Logger:
from src.utils.logger import hub_logger

# Initializing Forecast and RFM Predictors:
@st.cache_resource
def load_predictors():
    return ForecastPredictor(), RFMPredictor()

forecast_predictor, rfm_predictor = load_predictors()

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
    hub_logger.info('Initializing query execution lifecycle.')

    #Intent-Routing:
    with st.spinner('Analyzing Intent and Decomposing Query (if necessary)...'):
        routing_result= route_query(user_query= user_query)

    with st.expander("Router Logic", expanded= True):
        st.caption(f"**Reasoning: {routing_result.reasoning}")
        for task in routing_result.tasks:
            st.info(f"**Engine:** {task.engine_name} | **Sub-Query:** {task.sub_query}")

    st.divider()

    # Parallel Execution (Handling Multi-Intent if Necessary):
    for task in routing_result.tasks:

        # RAG Engine Trigger:
        if task.engine_name == 'RAG_ENGINE':
            hub_logger.info('Executing Vector RAG Engine.')
            st.subheader('Qualitative Insights')
            with st.spinner('Querying Vector Database and Synthesizing...'):
                collection_to_query= task.target_collection or 'customer_reviews'
                rag_response= execute_rag_query(user_query= task.sub_query, collection_name= collection_to_query,  n_results= 6)
                st.write(rag_response)
            st.divider()

    # SQL Engine Trigger:
        elif task.engine_name == 'SQL_ENGINE':
            hub_logger.info('Executing SQL Engine.')
            st.subheader('Quantitative Insights (Structured Data)')
            with st.spinner('Generating SQL and Executing...'):
                sql_result= execute_text_to_sql(user_query= task.sub_query)
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
        if task.engine_name == 'PREDICTIVE_ENGINE':
            hub_logger.info('Executing Predictive Analytics Inference Pipeline.')
            st.subheader('🔮 Predictive Analytics')
            with st.spinner(f'Running Predictive Task: {task.predictive_task}...'):
                st.markdown(f'**🧠 Predictive Engine Task:** {task.predictive_task}')

                try:

                    # 1. Forecasting (Revenue or Inventory):
                    if task.predictive_task in ['revenue_forecast', 'inventory_forecast']:
                       forecast_type= 'revenue' if task.predictive_task == 'revenue_forecast' else 'inventory'
                       category_entity= str(task.predictive_entity) if task.predictive_entity else ""

                       prediction= forecast_predictor.predict(
                           category= category_entity,
                           forecast_type= forecast_type
                       )

                       if prediction['status'] == 'success':
                           st.metric(
                               label= f"Projected 7-Day {forecast_type.title()} ({prediction['category'].title()})",
                               value= prediction['value']
                           )
                           st.caption(f"**Target Date:** Week of {prediction['target_date']}")
                       else:
                           st.error(prediction.get('message', 'Forecast Failed.'))

                    # 2. RFM Churn Profiling:
                    


                except Exception as e:
                    st.info(f'Inference Failed. Error: {e}')

            hub_logger.info('Query Execution lifecycle completed successfully.')
            st.divider()


        # Unknown Route:
        if task.engine_name == 'UNKNOWN':
            st.error("❓ I couldn't determine how to answer this question. Please ask a question related to e-commerce revenue or customer reviews.")