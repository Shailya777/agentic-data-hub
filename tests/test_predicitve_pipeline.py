import sys
import os
import pytest

# Updating Sys Path to find Project Root Folder:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.predictive.delivery_predictor import predict_delivery_delay

def test_delivery_delay_simulation_mode():
    """
    Tests that the XGBoost pipeline successfully ingests a raw feature
    dictionary, processes it without crashing, and returns the expected UI keys.
    """

    # Dummy Features for Testing:
    dummy_features= {
        "estimated_days_to_deliver": 11,
        "total_freight_value": 20.0,
        "total_weight_g": 10000,
        "total_volume_cm3": 50000,
        "purchase_month": 12,
        "purchase_day_of_week": 2,
        "is_interstate": 1,
        "seller_historical_delay_rate": 0.08
    }

    result= predict_delivery_delay(dummy_features)

    # Assert: Varifying the Output is a Dictionary Containing Exactly what the UI Expects:
    assert isinstance(result, dict), 'Engine must return a dictionary'
    assert 'status' in result, "Dictionary must contain 'status' key"
    assert 'confidence' in result, "Dictionary must contain 'confidence' key"
    assert 'recommendation' in result, "Dictionary must contain 'recommendation' key"

    # Printing to Terminal:
    print(f'\nTest Passed: Output: {result}')