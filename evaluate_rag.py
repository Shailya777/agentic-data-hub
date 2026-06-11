import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.engines.rag_engine import (
    _expand_query,
    _retrieve_and_deduplicate,
    _rerank_chunks,
    _synthesize_answer,
    chroma_client,
    embedding
)

# Loading Environment Variables:
load_dotenv()

# Initializing OpenAI Client:
client= OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Evaluation Dataset:
EVAL_QUESTIONS= [
{
        "query": "What are the most common complaints about delivery?",
        "collection": "customer_reviews"
    },
    {
        "query": "What is our corporate policy for handling returns after 30 days?",
        "collection": "olist_corporate_policies"
    },
    {
        "query": "Summarize the feedback from customers who bought watches.",
        "collection": "customer_reviews"
    },
    {
        "query": "What is the CEO's favorite color?", # Trick Question
        "collection": "olist_corporate_policies"
    }
]
