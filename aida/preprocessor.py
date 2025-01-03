from dataclasses import dataclass
from typing import Optional, List
from .providers import LLMProviderFactory
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
    """Preprocessor for queries to determine if they're relevant to server management"""
    
    def __init__(self, config: AidaConfig, conversation):
        """Initialize the preprocessor with a config and conversation manager
        
        Args:
            config: AidaConfig instance
            conversation: ConversationManager instance for maintaining history
        """
        self.config = config
        self.conversation = conversation
        self.llm = LLMProviderFactory.get_provider(
            provider_type=config.preprocessor_provider,
            model=config.preprocessor_model,
            temperature=0
        )
    
    def process_query(self, query: str) -> PreprocessorResult:
        """Process a query to determine if it's relevant to server management
        
        Args:
            query: The query to process
            
        Returns:
            PreprocessorResult containing relevance check and optional response
        """
        if not query:
            return PreprocessorResult(
                is_relevant=False,
                query=query,
                response="Empty query. Please ask a question."
            )
            
        # Get conversation context from the shared conversation manager
        conversation_context = self.conversation.get_recent_messages()
            
        prompt = f"""You are a query preprocessor for a server management AI assistant.
        Your job is to determine if a query is related to server management or not.
        

        
        Rules:
        1. If the query is about server management, system administration, or Linux commands, respond with "RELEVANT:"
        2. If the query is NOT about server management, respond with "NOT RELEVANT:"
        3. After the prefix, briefly explain why in one sentence
        4. For follow-up questions, consider the context from previous messages to determine relevance
        
        Examples of relevant queries and follow-ups:
        - How many users are logged in?
        - What's the current disk usage?
        - Show me the running processes
        - Check server uptime
        - List all open ports
        - How long have they been running? (when asked after a process-related question)
        - When did they log in? (when asked after a user-related question)
        
        Examples of irrelevant queries:
        - What's the weather like today?
        - Tell me a joke
        - What's the capital of France?
        - How do I make pasta?
        - What's 2+2?

        Previous conversation context:
        {conversation_context}
        
        Current query: {query}
        Response: """
        print("From Preprocessor: ", prompt)
        try:
            response = self.llm.invoke(prompt).content
            logger.debug(f"LLM Response: {response}")
            
            is_relevant = response.strip().startswith("RELEVANT:")
            reason = response.split(":", 1)[1].strip() if ":" in response else "No reason provided"
            
            # Add preprocessor result to conversation history
            self.conversation.add_preprocessor_result(is_relevant, reason)
            
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