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
    model_name= 'text-embedding-3-small'
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

    # Embedding reviews and Storing it in Vector Store:
    documents= []
    metadatas= []
    ids= []
    print(f'Embedding {len(df)} reviews into ChromaDB...')
    for index, row in df.iterrows():
        # Combining Title and Message if both exists:
        title= str(row['review_comment_title']) if pd.notna(row['review_comment_title']) else ''
        message= str(row['review_comment_message'])
        full_text= f"Title: {title}\nMessage: {message}".strip()

        documents.append(full_text)
        metadatas.append({'score': int(row['review_score']), 'review_id': str(row['review_id'])})
        ids.append(f"doc_{index}")

    # Adding to ChromaDB:
    collection.add(
        documents= documents,
        metadatas= metadatas,
        ids= ids
    )

    print('Vector database Successfully Populated!')

# Executing RAG Query and Getting Most Relevant Reviews:
def execute_rag_query(user_query: str, n_results: int=5) -> str:
    """
    Takes a natural language question, finds the most relevant reviews,
    and uses an LLM to synthesize a qualitative answer.
    :param user_query: User's Query in natural language format.
    :param n_results: Number of Closest Vectors to return while DB Search.
    :return: Answer to User's Query Synthesized by LLM Using found Context from VectorDB.
    """

    # Retrieving Relevant Documents from ChromaDB:
    results= collection.query(
        query_texts= [user_query],
        n_results= n_results
    )
    retrieved_reviews= results['documents'][0]
    metadata_list= results['metadatas'][0]


if __name__ == '__main__':
    reviews_file= os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/raw/olist_order_reviews_dataset.csv"))
    if collection.count() == 0:
        build_vector_database(csv_path= reviews_file)
