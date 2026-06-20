import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from dotenv import load_dotenv

load_dotenv()

def migrate_data_to_sqlite():
    """

    :return:
    """

    # Connecting to MySQL Database (Source):
    print('Connecting to MySQL database...')
    mysql_url= URL.create(
        drivername= 'mysql+pymysql',
        username= os.getenv("DB_USER"),
        password= os.getenv('DB_PASSWORD'),
        host= os.getenv("DB_HOST"),
        database= os.getenv("DB_NAME")
    )
    mysql_engine= create_engine(mysql_url)

    # Connecting to SQLite (Destination):
    sqlite_path= os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ecommerce_db.sqlite'))
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError('SQLite Database not found. Please run create_sqlite_schema.py first.')

    sqlite_engine= create_engine(f"sqlite:///{sqlite_path}")

    # Enforcing Foreign Keys:
    # (SQLite requires PRAGMA foreign_keys = ON to be set per-connection)
    with sqlite_engine.connect() as connection:
        connection.execute(text('PRAGMA foreign_keys = ON;'))

    # Inserting Data following Strict Order (Parent tables > Child Tables > Grandchild Tables):
    insertion_order= [
        'customers',
        'sellers',
        'products',
        'product_category_name_translation',
        'orders',
        'order_items',
        'order_payments',
        'order_reviews'
    ]

    print('\nMigrating data to SQLite...')

    for table_name in insertion_order:
        print(f'Extracting [{table_name}] from MySQL...')
        df= pd.read_sql_table(table_name, con= mysql_engine)

        print(f"Loading {len(df)} rows into SQLite...")
        df.to_sql(name= table_name,
                  con= sqlite_engine,
                  if_exists= 'append', # if_exists='append' ensures we do not overwrite the constraints
                  index= False,
                  chunksize= 10000)

    print(f"Database Migration finished Successfully!!")

if __name__ == '__main__':
    migrate_data_to_sqlite()