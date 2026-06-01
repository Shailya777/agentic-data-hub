import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

def get_order_features_from_db(order_id: str) -> dict:
    """
    Queries MySQL to pull live features for a specific order_id.
    :param order_id: Order ID to query.
    :return: Dictionary of live features for a specific order_id.
    """

    # Database Engine:
    engine_url= URL.create(
        drivername= 'mysql+pymysql',
        username= os.getenv('DB_USER'),
        password= os.getenv('DB_PASSWORD'),
        host= os.getenv('DB_HOST'),
        database= os.getenv('DB_NAME')
    )
    db_engine = create_engine(engine_url)

    # Same Query as Delivery Data Extraction, filtered for a single Order ID:
    query = f"""
        WITH SellerDelays AS (
            SELECT 
                oi.seller_id,
                COUNT(o.order_id) as total_orders,
                SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) as delayed_orders
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY oi.seller_id
        )
        SELECT 
            MONTH(o.order_purchase_timestamp) AS purchase_month,
            DAYOFWEEK(o.order_purchase_timestamp) AS purchase_day_of_week,
            DATEDIFF(o.order_estimated_delivery_date, o.order_purchase_timestamp) AS estimated_days_to_deliver,
            SUM(oi.freight_value) AS total_freight_value,
            SUM(p.product_weight_g) AS total_weight_g,
            SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS total_volume_cm3,
            CASE WHEN c.customer_state != s.seller_state THEN 1 ELSE 0 END AS is_interstate,
            MAX(sd.delayed_orders / NULLIF(sd.total_orders, 0)) AS seller_historical_delay_rate
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN sellers s ON oi.seller_id = s.seller_id
        LEFT JOIN SellerDelays sd ON s.seller_id = sd.seller_id
        WHERE o.order_id = '{order_id}'
        GROUP BY o.order_id, o.order_purchase_timestamp, o.order_estimated_delivery_date, c.customer_state, s.seller_state;
        """

    df= pd.read_sql(sql= query, con= db_engine)
    if df.empty:
        return None
    return df.to_dict(orient= 'records')[0]

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
        model_path= os.path.abspath(os.path.join(current_dir, '../../../models/delivery_delay_pipeline.pkl'))


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
        'seller_historical_delay_rate': 0.35
    }
    print('Testing Predictive Engine...')
    result= predict_delivery_delay(test_order)
    print(f'Result: {result}')

    # Testing Order Features from DB:
    test_order_id= '00010242fe8c5a6d1ba2dd792cb16214'
    print(get_order_features_from_db(test_order_id))