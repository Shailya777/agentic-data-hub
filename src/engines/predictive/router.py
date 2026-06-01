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