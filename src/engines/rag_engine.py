import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry_if_result

from src.utils.logger import hub_logger

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

# Refining The User Query before Context Fetching:
def _expand_query(user_query: str) -> List[str]:
    """
    Generates synonymous, refined queries to widen the retrieval net.
    :param user_query: User's Query in Natural Language format.
    :return: List of Refined queries.
    """
    hub_logger.info('Expanding User Query...')

    expansion_prompt= f"""
    You are an AI data retrieval assistant. Your job is to take a user's query and generate 
    TWO additional, refined, slightly different versions of it to help search a vector database.
    Focus on synonyms and different ways a user might phrase the problem.
    Output ONLY the two new queries, separated by a newline. Do not number them.
    
    Original Query: {user_query}
    """

    response= openai.chat.completions.create(
        model= 'gpt-4o-mini',
        messages= [
            {'role': 'user', 'content': expansion_prompt}
        ],
        temperature= 0.7,
    )

    expanded= response.choices[0].message.content.strip().split('\n')
    queries= [user_query] + [q.strip() for q in expanded if q.strip()]
    hub_logger.info(f'Generated Search Queries: {queries}')
    return queries


def _retrieve_and_deduplicate(search_queries: List[str], collection, chunks_per_query: int= 5) -> List[str]:
    """
    Hits the vector DB with multiple queries and deduplicates the results.
    :param search_queries: List of Original User Query with refined versions of it.
    :param collection: Collection to query.
    :param chunks_per_query: Number of Chunks to retrieve from Vector DB for each query.
    :return: List of Chunks, de-duplicated.
    """
    hub_logger.info('Retrieving Context for all queries...')
    retrieved_chunks= {}

    for query in search_queries:
        results= collection.query(
            query_texts= [query],
            n_results= chunks_per_query
        )

        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id not in retrieved_chunks:
                retrieved_chunks[doc_id]= results['documents'][0][i]

    raw_chunks= list(retrieved_chunks.values())
    hub_logger.info(f"Retrieved {len(raw_chunks)} unique Chunks from Vector DB before Re-Ranking.")
    return raw_chunks


# Re-Ranking Retrieved Chunks based on their relevance:
def _rerank_chunks(user_query: str, raw_chunks: List[str], top_k: int= 5) -> List[str]:
    """
    Uses an LLM to score and filter chunks based on actual relevance.
    :param user_query: Original User Query.
    :param raw_chunks: Chunks retrieved from Vector DB using Original and Refined Uer Queries.
    :param top_k: Number of top chunks to return after re-ranking.
    :return: List of Chunks.
    """
    hub_logger.info('Re-Ranking Chunks by Relevance...')

    rerank_prompt= f"""
    You are an expert relevance evaluator. I will provide a user query and a list of text chunks.
    Score each chunk from 0 to 10 based on how helpful it is for answering the user's query.
    Output ONLY a comma-separated list of integer scores in the exact order the chunks are presented.
    
    User Query: {user_query}
    
    Chunks:
    """

    for idx, chunk in enumerate(raw_chunks):
        rerank_prompt+= f"\n[{idx}] {chunk}\n"

    response= openai.chat.completions.create(
        model= 'gpt-4o-mini',
        messages= [
            {'role': 'user', 'content': rerank_prompt}
        ],
        temperature= 0.0,
    )

    try:
        scores= [int(s.strip()) for s in response.choices[0].message.content.split(',')]
        print(scores)
        scored_chunks= list(zip(raw_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse= True)

        # Leaving out chunks with scores of less or equal to 3:
        best_chunks= [chunk for chunk, score in scored_chunks[:top_k] if score > 3]
        hub_logger.info(f"Retained top {len(best_chunks)} Chunks for final response generation.")

        return best_chunks

    except Exception as e:
        hub_logger.warning(f"Re-Ranking Failed. Falling back to raw chunks.\nError: {e}")
        return raw_chunks[:top_k]

# Generating Answer to User Query using Re-Ranked Best Chunks:
def _synthesize_answer(user_query: str, best_chunks: List[str]) -> str:
    """
    Generates the final response of User Query based on the Ranked Retrieved Context.
    :param user_query: User's Query.
    :param best_chunks: Chunks returned from re-ranking.
    :return: Response of User's Query.
    """
    hub_logger.info('Synthesizing final answer...')

    context= "\n\n".join([f"Review Excerpt: {chunk}" for chunk in best_chunks])

    sys_prompt= """
    You are a Senior Customer Experience Analyst for an e-commerce platform. 
    Analyze the provided customer reviews and answer the user's question professionally.
    
    CRITICAL RULES:
    1. Base your answer STRICTLY on the provided context. Do not use outside knowledge.
    2. If the context does not contain the answer, say "The provided reviews do not contain enough information to answer this."
    3. Do NOT mention "According to the context". Write naturally as if you did the research yourself.
    4. Synthesize the findings across different languages into a single, cohesive English report.
    """

    response= openai.chat.completions.create(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"Context:\n{context}\n\nUser Question: {user_query}"}
        ],
        temperature= 0.3,
    )

    return response.choices[0].message.content

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
            context += f"--- Policy Document Excerpt {index+1} ---\n{doc}\n\n"


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
        2. CITE METRICS: If the context contains specific operational thresholds, you must explicitly include those numbers.
        3. FACTUAL DEDUCTION: You must apply the rules in the context to the user's specific scenario. If the policy states a limit of 50,000 and the user asks about 60,000, you must explicitly deduce that the user's package exceeds the limit and state the resulting penalty. 
        4. BOUNDARIES: If the provided context does not contain the rules required to answer, explicitly state that the policy manual does not cover this scenario. Do not guess outside the provided rules.
        """

    user_prompt= f"User Query: {user_query}\n\nContext:\n{context}"

    # ADD THIS LINE TO PEEK INSIDE THE BLACK BOX
    #print(f"\n[DEBUG] Target Collection: {collection_name}")
    #print(f"[DEBUG] User Sub-Query: {user_prompt}")
    #print(f"[DEBUG] Retrieved Context:\n{context}\n")

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
    #print('--- Testing Customer Reviews RAG ---')
    #test_query_1= 'What are the most common complaints about delivery?'
    #print(execute_rag_query(user_query= test_query_1,
    #                       collection_name= 'customer_reviews',
    #                      n_results= 5))
    collection= chroma_client.get_collection('olist_corporate_policies', embedding_function= embedding)
    lst= _expand_query(user_query= 'What is the company policy for late delivery?')
    chunks= _retrieve_and_deduplicate(lst, collection= collection)
    best_chunks= _rerank_chunks(user_query= 'What is the company policy for late delivery?', raw_chunks= chunks, top_k= 5)
    res= _synthesize_answer(user_query= 'What is the company policy for late delivery?', best_chunks= best_chunks)
    print(res)