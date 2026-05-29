import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# Loading Environment Variables:
load_dotenv()

def extract_delivery_data():
    """
    Extracts feature-engineered data from MySQL to train the Delivery Delay Predictor.
    :return:
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
    query= 'SELECT * FROM customers LIMIT 10;'
    df = pd.read_sql(query, db_engine)
    print(df)

if __name__ == '__main__':
    extract_delivery_data()