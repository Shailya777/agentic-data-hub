import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from src.engines.predictive.delivery_predictor import predict_delivery_delay

load_dotenv()
openai= OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

class PredictiveRoutingDecision(BaseModel):
    """
    Schema for Choosing Predictive Model and Arguments.
    """

    model_type: str= Field(
        description= "Must be 'DELAY', 'CHURN', or 'FORECAST'"
    )
    model: str= Field(
        description= "Must be 'OPERATIONAL' (looking up an ID) or 'SIMULATION' (manual feature simulation)"
    )
    order_id: Optional[str] = Field(
        default= None,
        description= 'he extracted alphanumeric order ID if mode is OPERATIONAL'
    )
    simulation_features: Optional[Dict[str, Any]]= Field(
        default= None,
        description= 'Dictionary of manually simulated metrics if mode is SIMULATION. Keys should exactly match: estimated_days_to_deliver, total_freight_value, total_weight_g, total_volume_cm3, purchase_month, purchase_day_of_week, is_interstate, seller_historical_delay_rate'
    )

def route_predictive_task(sub_query: str) -> dict:
    """
    Parses a predictive sub-query, determines the correct ML target,
    and returns the prediction result.
    :param sub_query: Sub Query from Main Intent Router.
    :return: Dictionary of Predictive ML model results.
    """

    system_prompt= """
    You are an expert ML Router. Your job is to parse an abstract data request and format it for a predictive backend pipeline.
    
    Determine the model_type:
    - 'DELAY': Questions about shipping delays, timing, packages arriving late, or specific order delivery tracking.
    - 'CHURN': Questions about customer retention, loyalty tier drops, or high-risk buyer behavior.
    - 'FORECAST': Questions about future demand trends, revenue projections, or sales inventory requirements.
    
    Determine the mode:
    - Use 'OPERATIONAL' if the user provides an alphanumeric hash ID (like an order_id or customer_id).
    - Use 'SIMULATION' if the user asks a hypothetical 'What if' question using numeric constraints (weight, volume, months, etc.).
    """