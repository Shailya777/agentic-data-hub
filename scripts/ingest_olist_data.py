import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Loading Environment Variables:
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# SQLAlchemy Engine using pymysql:
engine_url = URL.create(
    drivername= 'mysql+pymysql',
    host= DB_HOST,
    password= DB_PASSWORD,
    username= DB_USER,
    database= DB_NAME
)
engine = create_engine(engine_url)

# From CSV files to MySQL:
def ingest_csv_to_mysql(csv_path: str, table_name: str):
    """
    Reads a CSV and uploads it to the MySQL database in chunks.
    :param csv_path: Path of the CSV file.
    :param table_name: Table name to use in MySQL database.
    """
    print(f"Ingesting {csv_path} into {table_name}")
    try:
        df = pd.read_csv(csv_path)
        # Upload to SQL in chunks of 10,000 to prevent memory bottlenecks.
        # if_exists='replace' drops the table if it exists and recreates it.
        df.to_sql(name= table_name, con= engine, if_exists= 'replace', index= False, chunksize= 10000)
        print(f"Successfully ingested {len(df)} rows into {table_name}.\n")

    except Exception as e:
        print(f"Error ingesting {csv_path} into {table_name}: {e}")

if __name__ == "__main__":

    # Defining Mapping of CSV files to MySQL Table Names:
    data_files= {
        '../data/raw/olist_customers_dataset.csv': 'customers',
        '../data/raw/olist_orders_dataset.csv': 'orders',
        '../data/raw/olist_order_items_dataset.csv': 'order_items',
        '../data/raw/olist_products_dataset.csv': 'products',
        '../data/raw/olist_order_payments_dataset.csv': 'order_payments',
        '../data/raw/olist_sellers_dataset.csv': 'sellers',
        '../data/raw/product_category_name_translation.csv': 'product_category_name_translation',
        '../data/raw/olist_order_reviews_dataset.csv': 'order_reviews'
    }

    for csv_path, table_name in data_files.items():
        if os.path.exists(csv_path):
            ingest_csv_to_mysql(csv_path= csv_path, table_name= table_name)
        else:
            print(f"File {csv_path} not found. Skipping.\n")