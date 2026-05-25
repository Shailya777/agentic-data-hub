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
    driver= 'mysql+pymysql',
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
