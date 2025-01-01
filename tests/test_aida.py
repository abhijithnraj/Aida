from aida.preprocessor import QueryPreprocessor
import pytest
from aida.core import Aida
from aida.config import AidaConfig
import subprocess
import json
import re
from langchain_ollama import ChatOllama

@pytest.fixture
def test_support_agent():
    """Fixture that provides a TestSupportAgent instance"""
    return TestSupportAgent("qwen2.5-coder:32b")

@pytest.fixture
def valid_config():
    """Fixture that provides a valid AidaConfig instance"""
    return AidaConfig(core_model="llama3.2:3b", preprocessor_model="llama3.2:3b")

@pytest.fixture
def aida_instance(valid_config):
    """Fixture that provides an Aida instance with valid configuration"""
    return Aida(config=valid_config)

@pytest.fixture
def invalid_config():
    """Fixture that provides an invalid AidaConfig instance"""
    return AidaConfig(core_model="nonexistent_model")

class TestSupportAgent:
    def __init__(self, model="llama3.2:3b"):
        self.model = model
        self.llm = ChatOllama(model=self.model)

    def verify_response(self, query, response):
        verification_query = f'''You are a Test support agent whose job is to check whether the reply to a query makes sense. 
        Your answer should always be true or false and nothing else.
        Example:
        Query: 'How many users are logged in?', Response: 'There are 10 users logged in.', Is this true or false?
        Answer: true
        Query: 'How many users are logged in?', Response: 'Weather is nice today', Is this true or false?
        Answer: false
        Query: '{query}', Response: '{response}'. Is this true or false?'''
        verification_response = self.llm.invoke(verification_query).content.strip()
        print("Response from TestSupportAgent: ", verification_response)
        return verification_response.lower() == "true"

class TestAida:
    """Test class for Aida functionality"""

    def test_aida_unavailable_model(self, invalid_config):
        """Test that Aida raises an error when an unavailable model is specified"""
        with pytest.raises(ValueError):
            Aida(config=invalid_config)

    def test_aida_available_model(self, valid_config):
        """Test that Aida initializes successfully when a valid model is specified"""
        aida = Aida(config=valid_config)
        assert aida is not None

    def test_aida_process_query(self, aida_instance):
        """Test that Aida processes queries correctly"""
        query = "How many users are logged in?"
        response = aida_instance.process_query(query)
        assert response is not None or response != "", "Response should not be empty"

        query = "What's the weather like today?"
        response = aida_instance.process_query(query)
        assert "not related to server management" in response.lower(), "Response should indicate irrelevance"

        query = ""
        response = aida_instance.process_query(query)
        assert "empty query" in response.lower(), "Response should indicate that the query is empty"

    def test_aida_response_with_ai(self, aida_instance, test_support_agent):
        """Test that Aida's responses make sense according to the TestSupportAgent"""
        queries = [
            "How long has the server been running?",
            "What is the current CPU usage?",
            "How many users are logged in?",
            "What is the status of the web server?",
            "Are there any disk space issues?"
        ]
        
        for query in queries:
            response = aida_instance.process_query(query)
            is_response_valid = test_support_agent.verify_response(query, response)
            assert is_response_valid, f"The response for query '{query}' should make sense according to the TestSupportAgent"
