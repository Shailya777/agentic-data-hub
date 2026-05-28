import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Loading Environment Variables:
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initializing OpenAI Client and ChromaDB:
openai= OpenAI(api_key= OPENAI_API_KEY)
DB_PATH= os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/vector_db"))
#print(DB_PATH)
chroma_client= chromadb.PersistentClient(path= DB_PATH)

# Embedding:
embedding= embedding_functions.OpenAIEmbeddingFunction(
    api_key= OPENAI_API_KEY,
    model_name= 'text_embedding-3-small'
)

# Get or Create Collection in Vector DB:
collection= chroma_client.get_or_create_collection(
    name= 'customer_reviews',
    embedding_function= embedding
)

# Building Vector Database:
def build_vector_database(csv_path: str, sample_size: int= 1000):
    """
    Reads the reviews CSV, filters for actual text comments, and embeds them into ChromaDB.
    Run this ONCE to populate the local vector database.
    :param csv_path: Path to the reviews CSV file.
    :param sample_size: Number of reviews to store in Vector DB.
    """

    print('Loading reviews dataset...')
    df= pd.read_csv(csv_path)

    # Filtering Out Empty Reviews:
    df= df.dropna(subset=['review_comment_message'])

    # Sampling sample_size reviews from data:
    df= df.tail(sample_size).reset_index(drop= True)
