import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

# Loading Environment Variables
load_dotenv()

# Initializing Database Engine:
engine_url= URL.create(
    drivername='mysql+pymysql',
    username= os.getenv('DB_USER'),
    password= os.getenv('DB_PASSWORD'),
    host= os.getenv('DB_HOST'),
    database= os.getenv('DB_NAME')
)
db_engine = create_engine(engine_url)

# Initializing OpenAI API
openai = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

# Defining SQL Schema for Context for LLM:
SCHEMA_CONTEXT= """
The database 'ecommerce_db' contains the following tables:
1. customers (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
2. orders (order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
3. order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
4. products (product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
5. order_payments (order_id, payment_sequential, payment_type, payment_installments, payment_value)
6. sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state)
7. product_category_name_translation (product_category_name, product_category_name_english)

Explicit Relationships:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_items.product_id = products.product_id
- order_items.seller_id = sellers.seller_id
- order_payments.order_id = orders.order_id
- products.product_category_name = product_category_name_translation.product_category_name
"""

class SQLResponse(BaseModel):
    """
    The structured output schema forcing the LLM to provide SQL and Charting Metadata for possible Visualization Artifect.
    """
    sql_query: str = Field(
        description="The raw, highly optimized MySQL query to execute.",
    )
    needs_chart: bool = Field(
        description= "True if the result represents a time series, comparison, or categorical aggregation that benefits from a chart."
    )
    chart_type: str = Field(
        description= "Must be 'bar', 'line', 'scatter', or 'none'."
    )
    x_axis: str = Field(
        description= "The exact column name from the SQL SELECT statement to use for the X-axis, or 'none'."
    )
    y_axis: str = Field(
        description= "The exact column name from the SQL SELECT statement to use for the Y-axis, or 'none'."
    )