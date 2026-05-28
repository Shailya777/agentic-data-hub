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