import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Updating Sys Path to find Project Root Folder:
sys.path.append(os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

from src.agents.intent_router import route_query, RoutingResponse, EngineTask

@patch('src.agents.intent_router.openai.chat.completions.parse')
def test_intent_router_sql_routing(mock_parse):
    """
    Tests that the intent router successfully parses a mock OpenAI response
    and returns our rigid Pydantic schemas without crashing.
    :param mock_parse:
    :return:
    """

    # Creating Fake Pydantic Output we want OpenAI to Return:
    mock_task= EngineTask(engine_name= 'SQL_ENGINE',
                          sub_query= 'Calculate total revenue.')
    mock_routing_response= RoutingResponse(
        tasks= [mock_task],
        reasoning= 'The user asked for quantitative revenue data.'
    )

    # Wiring The Fake Response into Deep Object Structure OpenAI returns:
    mock_parse.return_value.choices= [
        MagicMock(
            message= MagicMock(parsed= mock_routing_response)
        )
    ]

    # Calling Intent Router Function with Test Query:
    result= route_query('What was the total revenue last year?')

    # Asserting the Correct Pydantic Object:
    assert len(result.tasks) == 1
    assert result.tasks[0].engine_name == 'SQL_ENGINE'
    print('\nMock LLM Tets Passed!')