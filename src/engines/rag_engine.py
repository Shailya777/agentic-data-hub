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

    if not retrieved_reviews:
        return "No relevant customer reviews found for this query."

    # Formatting Context for LLM:
    context= ""
    for index, review in enumerate(retrieved_reviews):
        score= metadata_list[index]['score']
        context += f"--- Review {index+1} (Rating: {score} stars) ---\n{review}\n\n"

    # Synthesizing Answer Using Found Context:
    system_prompt= """
    You are a Customer Experience Analyst for a Brazilian e-commerce company.
    Answer the user's query using ONLY the provided customer review context.
    
    CRITICAL RULES:
    1. MULTILINGUAL SUPPORT: The source reviews are in Portuguese. You MUST write your entire final analysis in English. If you quote a specific phrase, provide the English translation.
    2. SEAMLESS SYNTHESIS: NEVER cite internal reference markers (e.g., "In Review 1", "According to Review 3"). Synthesize the insights naturally into a cohesive professional summary.
    3. FACTUAL BOUNDARIES: If the provided context does not contain the answer, explicitly state that you do not have enough information. Do not guess.
    """
    user_prompt= f"User Query: {user_query}\n\nCustomer Review Context:\n{context}"

    response= openai.chat.completions.create(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature= 0.3, #Low temperature for factual synthesis, slight creativity for readability
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    # ONLY RUN THIS UNCOMMENTED THE FIRST TIME to build the DB
    #reviews_file= os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/raw/olist_order_reviews_dataset.csv"))
    #if collection.count() == 0:
    #    build_vector_database(csv_path= reviews_file)

    # Testing Retrieval and Answer Synthesis:
    test_query= 'What are the most common complaints about delivery?'
    print(f'User Query: {test_query}\n')
    print('Synthesizing Answer using VectorDB..\n')
    answer= execute_rag_query(user_query= test_query, n_results= 10)
    print(answer)
