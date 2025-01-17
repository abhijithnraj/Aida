from typing import Optional, List, Dict
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_community.tools import ShellTool,DuckDuckGoSearchRun
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from .preprocessor import QueryPreprocessor
from .config import AidaConfig
from .providers import LLMProviderFactory
import logging
import re
from .tools.coder_tool import PythonCoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# from .tools.coder_tool import WriteCodeAndExecute
from .tools.validated_shelltool import shell_tool

class ConversationManager:
    """Manages conversation history for both preprocessor and core model"""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})
    
    def add_preprocessor_result(self, is_relevant: bool, reason: str):
        self.messages.append({"role": "system", "content": f"Preprocessor: {'Relevant' if is_relevant else 'Not relevant'} - {reason}"})
    
    def get_recent_messages(self, count: int = 5) -> str:
        """Get the most recent messages formatted as a string"""
        recent = self.messages[-count:] if len(self.messages) > count else self.messages
        formatted = []
        for msg in recent:
            prefix = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg["role"], msg["role"])
            formatted.append(f"{prefix}: {msg['content']}")
        return "\n".join(formatted)
    
    def get_memory_messages(self) -> List[HumanMessage | AIMessage]:
        """Convert messages to LangChain message format"""
        memory_messages = []
        for msg in self.messages:
            if msg["role"] == "user":
                memory_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                memory_messages.append(AIMessage(content=msg["content"]))
        return memory_messages

class Aida:
    def __init__(self, config: Optional[AidaConfig] = None, gui_validator=None):
        self.config = config or AidaConfig()
        
        # Initialize conversation manager
        self.conversation = ConversationManager()
        
        # Initialize core LLM provider
        self.llm = LLMProviderFactory.get_provider(
            provider_type=self.config.core_provider,
            model=self.config.core_model,
            temperature=0
        )
        
        self.gui_validator = gui_validator
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        
        # Initialize preprocessor with conversation manager
        self.preprocessor = QueryPreprocessor(
            config=self.config,
            conversation=self.conversation
        )
    
    def _setup_tools(self) -> list[Tool]:
        return [
            shell_tool,
            Tool(name="duckduckgo",
                 func = DuckDuckGoSearchRun().run,
                 description="Use this to search for information when you need it or cant get a job done."),
            Tool(name="python_coder",
                 func=PythonCoder(llm=self.llm.llm).process_query,
                 description="""This code will use an agent to write the code and execute it. You only need to pass in the query. The generated code will be in generated_code.py
                 This tool can handle installing packages and executing code. 

                 You must provide a very detailed description of the code you want to write.
                 
                 Example:
                 Action: write_code
                 Action Input: Write a python function to find the 7th prime number. Then call the function with 7 as the argument and print the result.
                 Observation: Code written to file generated_code.py
                 Action: shell
                 Action Input: python generated_code.py
                 Observation: The 7th prime number is 17
                 Final Answer: The 7th prime number is 17
                 """)
        ]
    
    def _setup_agent(self):
        # Using ZERO_SHOT_REACT_DESCRIPTION which follows a thought-action-observation pattern
        return initialize_agent(
            tools=self.tools,
            llm=self.llm.llm,  # Access the underlying LangChain LLM
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            early_stopping_method="force",
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": """You are AIDA, a helpful AI assistant.
                When asked a question, you MUST use the available tools to help the user.
                NEVER make up or hallucinate command outputs.
                ALWAYS use the shell tool to execute commands and get real output.
                
                You have access to chat history, so you can refer to previous questions and answers.
                When answering follow-up questions, make sure to consider the context from previous interactions.
                
                Important rules:
                1. ALWAYS use the shell tool to execute commands
                2. NEVER pretend to execute a command - actually use the tool
                3. If a command fails, show the error and explain what went wrong
                4. Never Execute the same command more than once
                5. After getting command output, explain what it means
                6. Always have at the very least Thought and Final Answer in response.
                7. Always end the response with a Final Answer. This is very important and it has to answer the query posed by the user.
                8. Only the Final answer is shown to the user, so it should include all the information needed to answer the query.
                9. For follow-up questions, consider the context from previous interactions.
                10. You can install a new package if required. But always follow these below rlules:
                     - Before you install anything, verify that the package does not exist on the system and 
                     - Always find out which OS is running on the server to use the correct package manager.

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
                    
                Example 3 (Follow-up):
                    Question: When did they log in?
                    Thought: From the previous 'who' command output, I can see the login times
                    Final Answer: Looking at the previous information: user1 logged in at 10:00, user2 at 10:05, and user3 at 10:10 on January 31st, 2024.
                
                Example 4:
                    Question: Plot the iris dataset
                    Thought: I need to use the python_coder tool to write the code to plot the iris dataset
                    Action: python_coder
                    Action Input: Plot the iris dataset
                    Observation: Code written to file generated_code.py
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: The iris dataset has been plotted
                    Final Answer: The iris dataset has been plotted
                
                Example 5:
                    Question: Write the code to find the 7th prime number
                    Thought: I need to use the python_coder tool to write the code to find the 7th prime number
                    Action: python_coder
                    Action Input: Find the 7th prime number
                    Observation: Code written to file generated_code.py
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: The 7th prime number is 17
                    Final Answer: The 7th prime number is 17
                 
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
        
        # Add user query to conversation history
        self.conversation.add_user_message(query)
        
        # Construct the prompt for the query using conversation history
        prompt = self.conversation.get_recent_messages() + f"\nUser: {query}"
        
        # First check if query is relevant using preprocessor
        #TODO: We need to move the preprocessor check out of AIDA. Its too restrictive.
        # preprocessor_result = self.preprocessor.process_query(query)
        # if not preprocessor_result.is_relevant:
        #     response = preprocessor_result.response or "This query is not related to server management."
        #     self.conversation.add_assistant_message(response)
        #     return response
            
        try:
            # Run the agent to process the query
            response = self.agent.invoke({"input": prompt})  # Use constructed prompt
            logger.info(f"Response: {response}")
            
            # Skip validation for strong models
            if not self.llm.is_strong():
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
                
                # Add assistant response to conversation history
                self.conversation.add_assistant_message(response)
                return response
            else:
                # Add assistant response to conversation history
                self.conversation.add_assistant_message(response["output"])
                return response["output"]
        except Exception as e:
            logger.error("Error processing query: %s", str(e))
            error_response = f"Error processing query: {str(e)}"
            self.conversation.add_assistant_message(error_response)
            return error_response

if __name__ == "__main__":
    aida = Aida()
    response = aida.process_query("How long has the server been running?")
    print("AIDA Response:", response)
    print("DONE")