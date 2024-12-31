from dataclasses import dataclass
from typing import Optional
from langchain_ollama import ChatOllama
from .config import AidaConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessorResult:
    """Result from the preprocessor containing relevance check and optional response"""
    is_relevant: bool
    query: str
    response: Optional[str] = None

class QueryPreprocessor:
    def __init__(self, model_name: Optional[str] = None, config: Optional[AidaConfig] = None):
        """Initialize the preprocessor with a language model"""
        self.config = config or AidaConfig()
        self.llm = ChatOllama(model=model_name or self.config.preprocessor_model, temperature=0)
    
    def process_query(self, query: str) -> PreprocessorResult:
        """
        Process a query to determine if it's related to server management
        
        Args:
            query: The user's query string
            
        Returns:
            PreprocessorResult containing relevance check and optional response
        """
        if not query.strip():
            return PreprocessorResult(
                is_relevant=False,
                query=query,
                response="Empty query provided. Please ask a question about server management."
            )
        
        # Prompt the LLM to determine if query is server management related
        prompt = f"""Determine if this query is related to server management or administration.
        Query: "{query}"
        
        Rules:
        1. The query should be about managing, monitoring, or administering a server
        2. Valid topics include: processes, users, disk usage, memory, CPU, network, services, logs, etc.
        3. Invalid topics: weather, general knowledge, math, jokes, or anything not related to servers
        
        Respond with either:
        RELEVANT: <reason> 
        or 
        NOT_RELEVANT: <reason>
        
        Example valid queries:
        - "How many users are logged in?" -> RELEVANT: Query about server user management
        - "Show disk usage" -> RELEVANT: Query about server storage
        
        Example invalid queries:
        - "What's the weather?" -> NOT_RELEVANT: Weather is not related to server management
        - "Tell me a joke" -> NOT_RELEVANT: Entertainment is not related to server management
        - "Who invented the linux command 'ls'?" -> NOT_RELEVANT: General knowledge is not related to server management
        """
        
        try:
            response = self.llm.invoke(prompt).content
            logger.debug(f"LLM Response: {response}")
            
            is_relevant = response.strip().startswith("RELEVANT:")
            reason = response.split(":", 1)[1].strip() if ":" in response else "No reason provided"
            
            if is_relevant:
                return PreprocessorResult(
                    is_relevant=True,
                    query=query
                )
            else:
                return PreprocessorResult(
                    is_relevant=False,
                    query=query,
                    response=f"This query is not related to server management: {reason}"
                )
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return PreprocessorResult(
                is_relevant=False,
                query=query,
                response=f"Error processing query: {str(e)}"
            ) 