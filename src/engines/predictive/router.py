import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from src.engines.predictive.delivery_predictor import predict_delivery_delay

load_dotenv()
openai= OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

class SimulationFeatures(BaseModel):
    """
    Schema for simulation features.
    """
    estimated_days_to_deliver: int = Field(default=11,
                                           description="Estimated shipping window in days. Default to 11 if unmentioned.")
    total_freight_value: float = Field(default=20.0, description="Freight cost value. Default to 20.0 if unmentioned.")
    total_weight_g: int = Field(description="Weight in grams. If user gives kg, multiply by 1000 to convert to grams.")
    total_volume_cm3: int = Field(description="Volume in cubic centimeters (cm3).")
    purchase_month: int = Field(default=6, description="Month of purchase (1-12). Default to 6 if unmentioned.")
    purchase_day_of_week: int = Field(default=2,
                                      description="Day of week (1=Sunday, 2=Monday, etc.). Default to 2 (Monday) if unmentioned.")
    is_interstate: int = Field(description="Set to 1 if shipping interstate/across states, 0 if within the same state.")
    seller_historical_delay_rate: float = Field(default=0.08,
                                                description="Historical failure rate of the vendor. Default to our dataset baseline of 0.08 if unmentioned.")

class PredictiveRoutingDecision(BaseModel):
    """
    Schema for Choosing Predictive Model and Arguments.
    """

    model_type: str= Field(
        description= "Must be 'DELAY', 'CHURN', or 'FORECAST'"
    )
    mode: str= Field(
        description= "Must be 'OPERATIONAL' (looking up an ID) or 'SIMULATION' (manual feature simulation)"
    )
    order_id: Optional[str] = Field(
        default= None,
        description= 'The extracted alphanumeric order ID if mode is OPERATIONAL'
    )
    simulation_features: Optional[SimulationFeatures]= Field(
        default= None,
        description= 'Populated ONLY if mode is SIMULATION.'
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

    # Parsing LLM Response:
    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': sub_query},
        ],
        response_format= PredictiveRoutingDecision,
        temperature= 0.0
    )

    decision= response.choices[0].message.parsed

    print(f'Model: {decision.model_type} & Mode: {decision.mode}')

    # Dispatching to Correct Machine Learning Model:
    if decision.model_type == 'DELAY':
        if decision.mode == 'OPERATIONAL' and decision.order_id:
            return predict_delivery_delay(decision.order_id)
        elif decision.mode == 'SIMULATION' and decision.simulation_features:
            return predict_delivery_delay(decision.simulation_features.model_dump())
        else:
            return {'error': 'Invalid combination of mode and arguments for delay prediction.'}

    elif decision.model_type == 'CHURN':
        return {'status': 'Yet to be Implemented.', 'message': 'The Customer Churn model training pipeline is currently offline.'}

    elif decision.model_type == 'FORECAST':
        return {'status': 'Yet to be Implemented.', 'message': 'The Time-Series Demand Forecaster is currently offline.'}

    return {'error': 'Unknown Predictive Model Destination Targeted.'}

if __name__ == '__main__':

    # Testing Delay Prediction by Order ID:
    print('--- Test 1: OPERATIONAL Delay Prediction ---')
    q1= 'Check tracking for order ID 00010242fe8c5a6d1ba2dd792cb16214 to see if it will arrive late.'
    print(route_predictive_task(q1))

    # Testing Delay Prediction by Order Features:
    print('--- Test 2: SIMULATION Delay Prediction ---')
    q2= "Will an item with a volume of 50000cm3 and weight of 10kg be delayed if sent interstate this December?"
    print(route_predictive_task(q2))