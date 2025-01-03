from typing import Optional
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_community.tools import ShellTool
from .preprocessor import QueryPreprocessor
from .config import AidaConfig
from .providers import LLMProviderFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Aida:
    def __init__(self, config: Optional[AidaConfig] = None):
        self.config = config or AidaConfig()
        
        # Initialize core LLM provider
        self.llm = LLMProviderFactory.get_provider(
            provider_type=self.config.core_provider,
            model=self.config.core_model,
            temperature=0
        )
        
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        
        # Initialize preprocessor with its own LLM provider
        self.preprocessor = QueryPreprocessor(
            config=self.config
        )
    
    def _setup_tools(self) -> list[Tool]:
        return [
            Tool(
                name="shell",
                func= ShellTool().run,
                description="""Execute shell commands on the server. Use this tool to run commands and get their output.
                Example:
                Action: shell
                Action Input: who
                Observation: user1    pts/0    2024-01-31 10:00 (:0)
                Thought: The 'who' command shows user1 is logged in
                """
            )
        ]
    
    def _setup_agent(self):
        # Using ZERO_SHOT_REACT_DESCRIPTION which follows a thought-action-observation pattern
        return initialize_agent(
            tools=self.tools,
            llm=self.llm.llm,  # Access the underlying LangChain LLM
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="force",
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": """You are AIDA, a helpful AI assistant that helps users manage their server.
                When asked a question, you MUST use the available tools to help the user.
                NEVER make up or hallucinate command outputs.
                ALWAYS use the shell tool to execute commands and get real output.
                
                Important rules:
                1. ALWAYS use the shell tool to execute commands
                2. NEVER pretend to execute a command - actually use the tool
                3. If a command fails, show the error and explain what went wrong
                4. Never Execute the same command more than once
                5. After getting command output, explain what it means
                6. Always have at the very least Thought and Final Answer in response.
                7. Always end the response with a Final Answer. This is very important and it has to answer the query posed by the user.
                8. Only the Final answer is shown to the user, so it should include all the information needed to answer the query.
                Example interaction:
                Example 1:
                    Question: How many users are logged in?
                    Thought: I need to use the shell tool with 'who | wc -l' command to check logged in users
                    Action: shell
                    Action Input: who | wc -l
                    Observation: 3
                    Final Answer: There are 3 users currently logged in.
                    
                Example 2:
                    Question: Who are the currently logged in users?
                    Thought: I need to use the shell tool with 'who' command to check logged in users
                    Action: shell
                    Action Input: who
                    Observation: user1    pts/0    2024-01-31 10:00 (:0)
                    user2    pts/1    2024-01-31 10:05 (:0)
                    user3    pts/2    2024-01-31 10:10 (:0)
                    Final Answer: There are 3 users currently logged in: user1, user2, and user3. Each is connected through a pseudo-terminal (pts).
                """,
                "format_instructions": """To use a tool, please use the following format:
                Thought: I need to use X tool because...
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times). It always has to follow this format.
                Thought: I now know what to respond
                Final Answer: the final response to the human"""
            }
        )
    
    def _validate_response(self, response: str) -> bool:
        """
        Validate that the response contains a Final Answer
        
        Args:
            response: The response from the agent
            
        Returns:
            bool: True if response contains Final Answer, False otherwise
        """
        return "Final Answer:" in response
    
    def process_query(self, query: str) -> str:
        """Process a user query and return a response"""
        if not query:
            return "Empty query. Please ask a question."
            
        # First check if query is relevant using preprocessor
        preprocessor_result = self.preprocessor.process_query(query)
        if not preprocessor_result.is_relevant:
            return preprocessor_result.response or "This query is not related to server management."
            
        try:
            # Run the agent to process the query
            response = self.agent.invoke({"input": query})
            logger.info(f"Response: {response}")
            
            # Skip validation for strong models
            if not  self.llm.is_strong():
                # Validate response has Final Answer
                if not self._validate_response(response):
                    # Print the response variable for debugging
                    print(f"Response before validation: {response}")
                    
                    # If no Final Answer, try to get one
                    final_response = self.llm.invoke(
                        f"""Based on this conversation and output, please provide a Final Answer that directly answers the user's question: "{query}"
                        
                        Previous output:
                        {response}
                        
                        Remember to start with "Final Answer:" and provide a clear, direct response. Don't say anything about agent."""
                    ).content
                    response = final_response.lstrip("Final Answer:").strip()
                return response
            else:
                return response["output"]
        except Exception as e:
            logger.error("Error processing query: %s", str(e))
            return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    aida = Aida()
    response = aida.process_query("How long has the server been running?")
    print("AIDA Response:", response)
    print("DONE")