import os
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
chroma_client= chromadb.PersistentClient(path= DB_PATH)

# Embedding:
embedding= embedding_functions.OpenAIEmbeddingFunction(
    api_key= OPENAI_API_KEY,
    model_name= 'text-embedding-3-small'
)


# Executing RAG Query and Getting Most Relevant Reviews:
def execute_rag_query(user_query: str, collection_name: str, n_results: int=5) -> str:
    """
    Takes a natural language question, finds the most relevant documents in the specified collection,
    and uses an LLM to synthesize a qualitative answer.
    :param user_query: User's Query in natural language format.
    :param collection_name: Name of the collection to query.
    :param n_results: Number of Closest Vectors to return while DB Search.
    :return: Answer to User's Query Synthesized by LLM Using found Context from VectorDB.
    """

    # Fetching requested Collection:
    try:
        collection= chroma_client.get_collection(
            name= collection_name,
            embedding_function= embedding,
        )
    except Exception as e:
        return f"Error: Could not access vector collection '{collection_name}'. Has it been ingested? Details: {e}"

    # Retrieving Relevant Documents from ChromaDB:
    results= collection.query(
        query_texts= [user_query],
        n_results= n_results
    )
    retrieved_docs= results['documents'][0]

    # Handling Different Metadata for Reviews and Policies Collection:
    if collection_name == 'customer_reviews':
        metadata_list= results['metadatas'][0]
    else:
        metadata_list= None

    if not retrieved_docs:
        return f"No relevant information found in {collection_name} for this query."

    # Formatting Context for LLM:
    context= ""
    for index, doc in enumerate(retrieved_docs):
        if collection_name == 'customer_reviews':
            score= metadata_list[index]['score']
            context += f"--- Review {index+1} (Rating: {score} stars) ---\n{doc}\n\n"
        else:
            context= f"--- Policy Document Excerpt {index+1} ---\n{doc}\n\n"


    # Dynamic Prompting using Target Collection:
    if collection_name == 'customer_reviews':
        system_prompt= """
        You are a Customer Experience Analyst for a Brazilian e-commerce company.
        Answer the user's query using ONLY the provided customer review context.
    
        CRITICAL RULES:
        1. MULTILINGUAL SUPPORT: The source reviews are in Portuguese. You MUST write your entire final analysis in English. If you quote a specific phrase, provide the English translation.
        2. SEAMLESS SYNTHESIS: NEVER cite internal reference markers (e.g., "In Review 1", "According to Review 3"). Synthesize the insights naturally into a cohesive professional summary.
        3. FACTUAL BOUNDARIES: If the provided context does not contain the answer, explicitly state that you do not have enough information. Do not guess.
        """
    else:
        system_prompt= """
        You are the Chief Operations Officer at Olist.
        Answer the user's query regarding internal corporate policy using ONLY the provided document context.
        
        CRITICAL RULES:
        1. AUTHORITATIVE TONE: Respond with clear, direct, and professional corporate language.
        2. CITE METRICS: If the context contains specific operational thresholds (like percentages, days, or volumes), you must explicitly include those numbers in your answer.
        3. FACTUAL BOUNDARIES: If the provided context does not contain the answer, explicitly state that the policy manual does not cover this scenario. Do not guess.
        """

    user_prompt= f"User Query: {user_query}\n\nContext:\n{context}"

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
    # Testing Retrieval and Answer Synthesis for Customer Reviews:
    print('--- Testing Customer Reviews RAG ---')
    test_query_1= 'What are the most common complaints about delivery?'
    print(execute_rag_query(user_query= test_query_1,
                            collection_name= 'customer_reviews',
                            n_results= 5))