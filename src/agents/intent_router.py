import os
from dotenv import load_dotenv
import openai as o
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Loading Environment Variables
load_dotenv()

# Initializing OpenAI Client:
openai = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


class EngineTask(BaseModel):
    """
    Structured Output Schema to pass while Intent Routing.
    """
    engine_name: str = Field(
        description= "Must be 'SQL_ENGINE', 'RAG_ENGINE', 'PREDICTIVE_ENGINE', or 'UNKNOWN'"
    )
    sub_query: str= Field(
        description= "The specific portion of the user's request rewritten as a standalone question for this engine. Do not include parts of the prompt meant for other engines."
    )

class RoutingResponse(BaseModel):
    """
    List of Structured Outputs (EngineTask).
    """
    tasks : List[EngineTask] = Field(
        description= "List of Tasks to Execute",
    )
    reasoning: str= Field(
        description= "Why these engines and sub-queries were chosen."
    )

def route_query(user_query: str) -> RoutingResponse:
    """
    Analyzes a user's natural language query and routes it to one OR MORE specialized engines.
    :param user_query: User's natural language query.
    :return: RoutingResponse object.
    """

    system_prompt= """
    You are a Master Orchestrator for an E-commerce Data Hub.
    Your job is to read the user's prompt, determine which database engines are required, 
    and DECOMPOSE the prompt into specific sub-queries for each engine.

    Available Engines:
    - SQL_ENGINE: For quantitative data, math, revenue, counting, and structured database queries.
    - RAG_ENGINE: For qualitative data, synthesizing customer reviews, and text analysis.
    - PREDICTIVE_ENGINE: For forecasting and machine learning.
    - UNKNOWN: If the query is completely unrelated to e-commerce.

    CRITICAL RULE: If a query requires multiple engines, create a separate task for each. 
    Rewrite the `sub_query` so the target engine ONLY sees the part of the question it is responsible for answering.
    """

    response= openai.chat.completions.parse(
        model= 'gpt-4o',
        messages= [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query},
        ],
        response_format= RoutingResponse,
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
        "Can you forecast our expected sales for the next 30 days?",
        "What were the top 10 cities by total revenue in the last financial year, and how much did each contribute percentage-wise to total sales?",
        "Summarize the most frequent customer complaints about delivery experience from recent reviews.",
        "Predict next month’s demand for gaming laptops using the last 24 months of sales data.",
        "Which products have the highest refund rates, and what reasons are customers mentioning in reviews for refunds?",
        "Identify stores whose sales behavior is anomalous compared to their historical seasonal patterns.",
        "What are customers saying about the new mobile app update after version 5.2 release?",
        "Calculate monthly repeat purchase rate and average basket size for users acquired through Facebook campaigns.",
        "Which customers are most likely to churn within the next 30 days?",
        "Why are enterprise customers dissatisfied with onboarding support?",
        "Find product categories where sales are consistently declining, forecast next quarter performance, and summarize customer feedback related to those categories.",
        "Show the daily active users trend for the past 6 months segmented by platform.",
        "Which warehouses are likely to experience stock shortages next month based on current inventory movement?",
        "Compare average order value, return percentage, and customer lifetime value across loyalty tiers.",
        "Read support tickets from the last year and identify newly emerging technical issues that were previously uncommon.",
        "Detect customers whose purchasing behavior significantly deviates from their normal buying patterns and summarize any related support complaints."
    ]

    for query in test_queries:
        print(f"\nUser Query: {query}")
        result = route_query(query)
        print(f'Reasoning: {result.reasoning}\n')
        for task in result.tasks:
            print(f"Engine: {task.engine_name} | Sub-Query: {task.sub_query}")