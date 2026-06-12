import sys
import os
import streamlit as st
import pandas as pd

# Adding Project Root to Python's Search Path:
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

# Page Config:
st.set_page_config(
    page_title= 'Agentic Data Hub',
    page_icon= '🧠',
    layout= 'wide',
)

# Initializing Forecast and RFM Predictors:
@st.cache_resource
def load_predictors():
    return (
        ForecastPredictor(models_dir= 'models', data_dir= 'data/processed'),
        RFMPredictor(data_dir='data/processed'))
forecast_predictor, rfm_predictor = load_predictors()

st.title('📊 Agentic Data Intelligence Hub')
st.caption('Enterprise Analytics Platform powered by Orchestrated Language Models & Predictive Intelligence')
st.markdown("""
    This intelligent interface automatically parses natural language questions and dispatches them to specialized backend engines. 
    It seamlessly manages structured relational databases (**SQL**), unstructured textual feedback (**Vector RAG**), and advanced predictive pipelines.
    """)
st.divider()

# Sidebar for User Input:
with st.sidebar:
    st.header('📥 Query Workspace')
    user_query= st.text_area('Ask an analytical or predictive question:',
                             placeholder= 'e.g., What was our total revenue last quarter, and what are customers complaining about regarding delivery?',
                             height= 150)

    # Showcasing System Capabilities in Sidebar:
    st.markdown('---')
    st.markdown('### 💡 Supported Capabilities')
    st.markdown(
        """
        - **Quantitative:** Revenue, sales metrics, and growth trends via dynamic Text-to-SQL.
        - **Qualitative:** Multilingual translation and semantic customer review analysis via Advanced RAG.
        - **Predictive:** 7-day forecasting, RFM customer churn scoring, and delivery risk analysis.
        """
    )

    run_button= st.button('Run Core Orchestrator',
                      type= 'primary',
                      use_container_width= True)

# Main Execution Logic:
if run_button and user_query:
    hub_logger.info('Initializing query execution lifecycle.')

    # Step 1: Intent-Routing Execution:
    with st.spinner('Analyzing Intent and Decomposing Query (if necessary)...'):
        routing_result= route_query(user_query= user_query)

    # Trace Console (Collapsed):
    with st.expander("🛠️ View Agentic Reasoning Trace & Execution Plan", expanded= False):
        st.markdown("### 🗺️ Orchestrator Execution Plan")
        st.markdown(f"**Core Reasoning:** *{routing_result.reasoning}*")
        st.markdown("---")

        for i, task in enumerate(routing_result.tasks, start= 1):
            st.markdown(f"**Task {i} Destination:** `{task.engine_name}`")
            st.code(f"Sub-Query: {task.sub_query}", language= "text")

            if task.target_collection:
                st.caption(f"Target Vector Collection: `{task.target_collection}`")

            if task.predictive_task:
                st.caption(f"Predictive Pipeline Task: `{task.predictive_task}` | Target Entity: `{task.predictive_entity}`")

            st.markdown("---")

    # # Step 2: Looping and Executing Dispatched Tasks:
    for task in routing_result.tasks:

        # ==========================================
        # RAG Engine Trigger:
        # ==========================================
        if task.engine_name == 'RAG_ENGINE':
            hub_logger.info('Executing Vector RAG Engine.')

            st.subheader('Qualitative Insights')
            with st.spinner('Querying Vector Database and Synthesizing...'):
                collection_to_query= task.target_collection or 'customer_reviews'
                rag_response= execute_rag_query(user_query= task.sub_query, collection_name= collection_to_query,  top_k= 5)

                # Rendering RAG Response:
                st.info(rag_response)
            st.divider()

        # ==========================================
        # SQL Engine Trigger:
        # ==========================================
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
                    with st.expander('🔍 View Generated SQL Statement', expanded= False):
                        st.code(metadata.sql_query, language= 'sql')
                        st.caption(f'Chart Engine Mapping: {metadata.chart_type.upper()} | X-Axis: {metadata.x_axis} | Y-Axis: {metadata.y_axis}')

                    # Rendering Result Dataframe and Chart:
                    col1, col2= st.columns([1,2])

                    with col1:
                        st.markdown('🔢 Results')
                        st.dataframe(data= df,
                             use_container_width= True)

                    with col2:
                        st.markdown('📉 Dynamic Visualization')

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
                            st.info('Scalar or single-metric matrix returned. Chart rendering skipped.')

                else:
                    st.error('The deterministic SQL execution layer failed to return results. Please review system console logs.')

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

                       # Guardrail if Category is "":
                       if not category_entity or category_entity.lower() in ['null', 'none']:
                           st.warning(
                               f"⚠️ **Missing Information:** I need a specific product category to run a {forecast_type} forecast. Please refine your question to include a category (e.g., 'Health & Beauty').")
                       else:
                           # Run the prediction only if we have a category
                           prediction = forecast_predictor.predict(
                               category=category_entity,
                               forecast_type=forecast_type
                           )

                           if prediction['status'] == 'success':
                               st.metric(
                                   label=f"Projected 7-Day {forecast_type.title()} ({prediction['category'].title()})",
                                   value=prediction['value']
                               )
                               st.caption(f"**Target Date:** Week of {prediction['target_date']}")
                           else:
                               st.error(prediction.get('message', 'Forecast failed.'))

                    # 2. RFM Churn Profiling:
                    elif task.predictive_task == 'rfm_churn':
                        customer_entity= str(task.predictive_entity) if task.predictive_entity else ""

                        # Guardrail if Customer ID is "":
                        if not customer_entity or customer_entity.lower() in ['null', 'none']:
                            st.warning(
                                "⚠️ **Missing Information:** Please provide a specific Customer ID to check their churn risk profile.")
                        else:
                            prediction= rfm_predictor.predict(
                                customer_unique_id= customer_entity
                            )

                            if prediction['status'] == 'success':
                                risk= prediction['value']
                                if 'High Risk' in risk:
                                    st.error(f"🚨 **Retention Alert:** Customer status is **{risk}**.")
                                elif 'VIP' in risk or 'Champion' in risk:
                                    st.success(f"👑 **VIP Profile:** Customer status is **{risk}**.")
                                else:
                                    st.info(f"👤 **Customer Profile:** Customer status is **{risk}**.")

                            else:
                                st.error(prediction.get('message', 'RFM Lookup Failed.'))

                    # 3. Delivery Delay Prediction:
                    elif task.predictive_task == 'delivery_delay':
                        order_entity= str(task.predictive_entity) if task.predictive_entity else ""

                        # Guardrail if Order ID is "":
                        if not order_entity or order_entity.lower() in ['null', 'none']:
                            st.warning(
                                "⚠️ **Missing Information:** Please provide a specific Order ID to predict delivery delays.")
                        else:
                            prediction= predict_delivery_delay(
                                input_data= order_entity
                            )

                            if prediction['status'] == 'success':
                                risk_level= prediction['risk_level']

                                col1, col2= st.columns([1,2])
                                with col1:
                                    st.metric(label= 'Predicted Delay Risk', value= prediction['delay_probability'])
                                with col2:
                                    if 'High Risk' in risk_level:
                                        st.error(f'**Predicted Status:** {risk_level.upper()} 🚨')
                                        st.info(f"**Strategic Recommendation:** {prediction['recommendation']}")
                                    else:
                                        st.success(f'**Predicted Status:** {risk_level.upper()} ✅')
                                        st.info(f"**Strategic Recommendation:** {prediction['recommendation']}")

                            else:
                                st.error(prediction.get('error', 'Delivery Prediction Failed.'))

                    else:
                        st.warning(f"Unknown predictive task: {task.predictive_task}")

                except Exception as e:
                    st.info(f'Inference Failed. Error: {e}')

            hub_logger.info('Query Execution lifecycle completed successfully.')
            st.divider()

        # Unknown Route:
        if task.engine_name == 'UNKNOWN':
            st.error("❓ I couldn't determine how to answer this question. Please ask a question related to e-commerce revenue or customer reviews.")