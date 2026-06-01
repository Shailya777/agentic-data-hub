import os
import joblib
import pandas as pd

def predict_delivery_delay(order_data: dict) -> dict:
    """
    Loads the trained XGBoost pipeline, runs inference on real-time order data,
    and applies business logic thresholds to predict delivery delays.
    :param order_data: Order Data to use as Features.
    :return: Dictionary of predicted Order Delay Staus with Probability.
    """
    try:
        # Loading The Saved Model:
        current_dir= os.path.dirname(os.path.abspath(__file__))
        model_path= os.path.abspath(os.path.join(current_dir, '../../models/delivery_delay_pipeline.pkl'))


        if not os.path.exists(model_path):
            return {
                'error': 'Model File Not Found, ',
            }
        pipeline = joblib.load(model_path)

        # Converting Incoming Data Dictionary to Pandas Dataframe:
        df_live= pd.DataFrame([order_data])

        # Predicting Probabilities:
        probabilities = pipeline.predict_proba(df_live)
        delay_probability= probabilities[0][1]

        # Threshold Tuning:
        risk_threshold= 0.75 # XGBoost defaults to 0.50. We raise it to 0.75 to prevent false alarms.

        is_high_risk= delay_probability >= risk_threshold

        # Returning Results:
        return {
            'status': 'High Risk of Delay' if is_high_risk else 'On Track',
            'confidence': f'{delay_probability * 100:.1f}%',
            'recommendation': 'Upgrade to Expedited Shipping.' if is_high_risk else 'Standard Processing.'
        }

    except Exception as e:
        return {'error': f'Prediction Failed: {str(e)}'}

if __name__ == '__main__':
    # Simulating a heavy, interstate order placed on a Friday (Day 6) right before Christmas (Month 12)
    test_order = {
        'estimated_days_to_deliver': 14,
        'total_freight_value': 45.50,
        'total_weight_g': 15000,
        'total_volume_cm3': 80000,
        'purchase_month': 12,
        'purchase_day_of_week': 6,
        'is_interstate': 1,
        'seller_historical_delay_rate': 0.35  # A notoriously slow seller
    }
    print('Testing Predictive Engine...')
    result= predict_delivery_delay(test_order)
    print(f'Result: {result}')