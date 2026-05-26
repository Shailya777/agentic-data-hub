import os
from dotenv import load_dotenv
import openai as o
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Loading Environment Variables
load_dotenv()
print(f"OpenAI Version: {o.__version__}")

# Initializing OpenAI Client:
openai = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Intent Route Class:
class IntentRoute(BaseModel):
    """
    The structured output schema for multi-intent routing.
    """
    engines : List[str] = Field(
        description= "A list of required engines. Valid options: 'SQL_ENGINE', 'RAG_ENGINE', 'PREDICTIVE_ENGINE', 'UNKNOWN'.",
    )
    reasoning: str= Field(
        description= "A one-sentence justification for why these specific engines were selected."
    )

def route_query(user_query: str) -> IntentRoute:
    """
    Analyzes a user's natural language query and routes it to one OR MORE specialized engines.
    :param user_query: User's natural language query.
    :return: IntentRoute object.
    """

    system_prompt= """
    You are the central Intent Orchestrator for an E-commerce Data Intelligence Hub.
    Your job is to analyze the user's query and determine which specialized engines are required to answer it fully.
    A single query might require multiple engines.

    Available Engines:
    1. SQL_ENGINE: Hard numbers, aggregations, timelines, relational data (revenue, order counts, product dimensions).
    2. RAG_ENGINE: Qualitative analysis, reading customer reviews, subjective feedback.
    3. PREDICTIVE_ENGINE: Future forecasting, machine learning predictions, anomaly detection.

    Rules:
    - If a query asks for data from multiple domains (e.g., revenue AND customer sentiment), return multiple engines.
    - If the query is completely unrelated to e-commerce, return ['UNKNOWN'].
    """

    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query},
        ],
        response_format= IntentRoute,
        temperature= 0.0
    )

    # Parsing the JSON string response back into our Pydantic model
    return response.choices[0].message.parsed

if __name__ == '__main__':
    # Test Cases to Check The Intent Routing:
    test_queries = [
        "What was our total revenue for the top 5 product categories last quarter?",
        "Why are customers giving 1-star reviews in São Paulo?",
        "What was our revenue in Q1, and why were people complaining about shipping during that time?",
        "Can you forecast our expected sales for the next 30 days?"

    ]

    for query in test_queries:
        print(f"\nUser Query: {query}")
        result = route_query(query)
        print(f"Routed To: {result.engines}")
        print(f"Reasoning: {result.reasoning}")