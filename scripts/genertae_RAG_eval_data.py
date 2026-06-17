import os
import sys
import json
import random
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Appending Project root to System Path:
sys.path.append(os.path.abspath(os.path.dirname(__file__), '../'))
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