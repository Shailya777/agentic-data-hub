import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Loading Environment Variables:
load_dotenv()

def extract_delivery_data(engine, output_dir):
    """
    Extracts feature-engineered data from MySQL to train the Delivery Delay Predictor.
    :param engine: SQLAlchemy engine object.
    :param output_dir: Path to directory where extracted training data will be saved.
    """
    print('Connecting to MySQL database...')
    engine_url = URL.create(
        drivername= 'mysql+pymysql',
        username= os.getenv('DB_USER'),
        password= os.getenv('DB_PASSWORD'),
        host= os.getenv('DB_HOST'),
        database= os.getenv('DB_NAME')
    )
    db_engine = create_engine(engine_url)

    extraction_query= """
    SELECT 
        o.order_id,
        DATEDIFF(o.order_estimated_delivery_date, o.order_purchase_timestamp) AS estimated_days_to_deliver,
        DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) AS actual_days_to_deliver,
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 
            ELSE 0 
        END AS is_delayed,
        SUM(oi.freight_value) AS total_freight_value,
        SUM(p.product_weight_g) AS total_weight_g,
        SUM(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS total_volume_cm3,
        CASE 
            WHEN c.customer_state != s.seller_state THEN 1 
            ELSE 0 
        END AS is_interstate
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY 
        o.order_id, 
        o.order_estimated_delivery_date, 
        o.order_purchase_timestamp, 
        o.order_delivered_customer_date,
        c.customer_state,
        s.seller_state;
    """

    print('Executing Delivery Data Extraction query...')
    df= pd.read_sql(sql= extraction_query,
                   con= db_engine)

    # Saving Extracted Data to CSV:
    df.to_csv(os.path.join(output_dir, 'delivery_training_data.csv'), index= False)

    print(f" -> Saved {len(df)} records. Baseline Delay Rate: {(df['is_delayed'].mean() * 100):.2f}%")


if __name__ == '__main__':
    # Output Directory to Store Extracted Delivery Delay Data:
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))
    os.makedirs(output_dir, exist_ok=True)
    extract_delivery_data()