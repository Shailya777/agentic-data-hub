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

Business Rules & Definitions:
1. REVENUE: Total revenue must ONLY be calculated using SUM(order_items.price) where orders.order_status = 'delivered'. 
2. TOTAL ORDER VALUE: This is different from Revenue. Total Order Value includes the item price PLUS the freight_value.
3. TRANSLATIONS: Always use product_category_name_translation.product_category_name_english for product categories. The base products table is in Portuguese.
4. DELIVERY TIME: Calculated as the difference in days between order_purchase_timestamp and order_delivered_customer_date.
5. ACTIVE SELLERS: A seller is only considered "active" if they have an associated order in the order_items table linked to a 'delivered' order.
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

def generate_sql_and_metadata(user_query: str, error_message: str = None, previous_sql: str = None) -> SQLResponse:
    """
    Generates a MySQL query from User Query in Natural Language and Suggests charting instructions via Structured Outputs.
    :param user_query: User Query in Natural Language.
    :param error_message: Error message from previous SQL query.
    :param previous_sql: Previous SQL query that gave error.
    :return: SQLResponse Object
    """

    prompt= f"Write a highly optimized MYSQL Query to answer this request: {user_query}"

    if error_message:
        prompt += f"\n\nYour previous query:\n{previous_sql}\n\nFailed with this MySQL error:\n{error_message}\n\nPlease fix the SQL query and provide updated charting metadata."

    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': f"You are a Senior Data Engineer. You strictly write highly optimized MySQL queries and determine the best way to visualize the results.\n\n{SCHEMA_CONTEXT}"},
            {'role': 'user', 'content': prompt}
        ],
        response_format= SQLResponse,
        #temperature= 0.0
    )

    return response.choices[0].message.parsed

def execute_text_to_sql(user_query: str, max_retries: int= 3) -> dict:
    """
    Executes the self-correcting text-to-sql loop and returns both the data and visualization metadata.
    :param user_query: User Query in Natural Language.
    :param max_retries: Maximum number of retries to execute generated SQL query.
    :return: Dictionary of SQL Query Result and Chart Metadata.
    """

    error_message = None
    previous_sql = None

    for attempt in range(max_retries):
        print(f"Attempt #{attempt+1}")
        try:
            # Getting Pydantic object SQLResponse from LLM:
            llm_response= generate_sql_and_metadata(user_query= user_query,
                                                     error_message=error_message,
                                                     previous_sql=previous_sql)
            sql_query= llm_response.sql_query

            print(f"Generated SQL query:\n{sql_query}\n")
            print(f"Proposed Chart: {llm_response.chart_type.upper()} (X: {llm_response.x_axis}, Y: {llm_response.y_axis})\n")

            # Executing The Generated SQL Query:
            df= pd.read_sql(sql= sql_query, con= db_engine)
            print("Execution Successful!")

            # Return a dictionary containing the result dataframe AND the chart metadata
            return {
                'dataframe': df,
                'metadata': llm_response,
            }

        except Exception as e:
            error_message = str(e)
            previous_sql = sql_query
            print(f"Status: Execution failed. Error: {error_message}")
            print("Action: Engaging Self-Correction Loop...\n")

    print("Fatal: Max retries reached. Could not resolve the query.")
    return {"dataframe": pd.DataFrame(), "metadata": None}

if __name__ == "__main__":
    test_query= 'What is the total revenue generated by the top 5 product categories in English?'
    result= execute_text_to_sql(user_query= test_query)

    if not result['dataframe'].empty:
        print("Final Data:")
        print(result["dataframe"])
        print("\nUI Rendering Instructions:")
        print(f"Render a {result['metadata'].chart_type} chart.")
        print(f"X-Axis: {result['metadata'].x_axis}")
        print(f"Y-Axis: {result['metadata'].y_axis}")