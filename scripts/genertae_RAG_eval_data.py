import os
import sys
import json
import random
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Appending Project root to System Path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.engines.rag_engine import chroma_client, embedding
from src.utils.logger import hub_logger

openai = OpenAI(api_key= os.getenv('OPENAI_API_KEY'))

class SyntheticTestCase(BaseModel):
    """
    Pydantic Schema for RAG Test Case generation.
    """
    query: str = Field(
        description= 'A distinct, natural language user query in English testing the source context.'
    )
    collection: str = Field(
        description= "Must match exactly either 'customer_reviews' or 'olist_corporate_policies'."
    )
    expected_ground_truth: str = Field(
        description= 'A 1-2 sentence statement of what fact the RAG engine must extract from this chunk.'
    )

def generate_case_from_chunk(chunk_text: str, collection_name: str) -> SyntheticTestCase:
    """
    Generates a test case from given chunk and collection name in SyntheticTestCase Pydantic Schema form.
    :param chunk_text: Text from the Retrieved Chunk.
    :param collection_name: Collection Name from where the Chunk was retrieved.
    :return: RAG Test Case in SyntheticTestCase Pydantic Schema form.
    """
    system_prompt= f"""
    You are an expert Machine Learning QA Engineer generating a golden test set for a RAG pipeline.
    Read the following source text extracted from the '{collection_name}' collection.
    
    Task:
    1. Generate a complex, realistic user query in English that CAN ONLY be answered accurately by extracting information from this text.
    2. Write a clear 'expected_ground_truth' answer in English stating the exact fact or rule contained within the text.
    
    CRITICAL: The source text might contain Portuguese or messy formatting. Your output 'query' and 'expected_ground_truth' MUST be written entirely in professional English.
    """

    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Source Text:\n{chunk_text}"}
        ],
        response_format= SyntheticTestCase,
        temperature= 0.4
    )

    return response.choices[0].message.parsed

def main(sample_size_per_collection: int = 25):
    hub_logger.info('Initializing Test Set Generation...')
    evaluation_dataset= []

    # Fetching from Vector Collections:
    try:
        policies_collection= chroma_client.get_collection(name= 'olist_corporate_policies',
                                                          embedding_function= embedding)

        reviews_collection= chroma_client.get_collection(name= 'customer_reviews',
                                                         embedding_function= embedding)
    except Exception as e:
        hub_logger.info(f'Failed to fetch collections: {e}')
        return

    # Fetching Documents from Both Collections:
    policies= policies_collection.get(include= ['documents'])['documents']
    reviews= reviews_collection.get(include= ['documents'])['documents']

    hub_logger.info(f"Available context: {len(policies)} policies, {len(reviews)} reviews.")

    # Random Sampling from fetched data:
    sampled_policies= random.sample(policies, min(sample_size_per_collection, len(policies)))
    sampled_reviews= random.sample(reviews, min(sample_size_per_collection, len(reviews)))

    # Processing Policies:
    for idx, doc in enumerate(sampled_policies):
        hub_logger.info(f"Generating policy question {idx+1}/{len(sampled_policies)}...")

        try:
            case= generate_case_from_chunk(chunk_text= doc,
                                           collection_name= 'olist_corporate_policies')
            evaluation_dataset.append(case.model_dump())
        except Exception as e:
            hub_logger.info(f"Skipping policy chunk due to error: {e}")

    # Processing Reviews:
    for idx, doc in enumerate(sampled_reviews):
        hub_logger.info(f"Generating review question {idx+1}/{len(sampled_reviews)}...")

        try:
            case= generate_case_from_chunk(chunk_text= doc,
                                           collection_name= 'customer_reviews')
            evaluation_dataset.append(case.model_dump())
        except Exception as e:
            hub_logger.info(f"Skipping review chunk due to error: {e}")

    # Fallback if Question generation failed:
    evaluation_dataset.append({
        "query": "What is the CEO's favorite color?",
        "collection": "olist_corporate_policies",
        "expected_ground_truth": "OUT_OF_BOUNDS_FALLBACK"
    })

    evaluation_dataset.append({
        "query": "Can you show me the server password logs?",
        "collection": "customer_reviews",
        "expected_ground_truth": "OUT_OF_BOUNDS_FALLBACK"
    })

    # Save Evaluation Dataset:
    target_path= os.path.join(os.path.dirname(__file__), '../tests/rag_evaluation_set.json')
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w', encoding= 'utf-8') as f:
        json.dump(evaluation_dataset, f, indent= 4, ensure_ascii= False)

    hub_logger.info(f"Success! Generated {len(evaluation_dataset)} evaluation cases at {target_path}")

if __name__ == '__main__':
    sample_size_per_collection= 24 #24 per collection + 2 OOB Questions= 50
    main(sample_size_per_collection)