import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

# Loading Environment Variables:
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
