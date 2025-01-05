from aida.providers.factory import LLMProviderFactory
from aida.tools.validated_shelltool import shell_tool
import re
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_code_to_file(code: str) -> str:
    """Write the code to a file"""
    if "```python" in code:
        regex = r"(?s)(?<=```python\n)(.*?)(?=\n```)"
        extracted_code = re.search(regex, code).group(0)
        logger.info(f"Extracted code: {extracted_code}")
    else:
        extracted_code = code
    with open("generated_code.py", "w") as file:
        file.write(extracted_code)
    return "Code written to file. Run python generated_code.py to execute and test the code"

class PythonCoder:
    """A tool that uses an AI to write and save code into a file based on an input query."""

    def __init__(self, llm, file_path: str = "generated_code.py"):
        """
        Initializes the WriteCodeAndExecute tool.

        Args:
            llm: The AI language model instance.
            shell_tool: An instance of ValidatedShellTool to execute shell commands.
            file_path: The path where the generated code will be saved.
        """
        self.llm = LLMProviderFactory.get_provider(
            provider_type="gemini",
            model="gemini-1.5-flash",
            temperature=0
        )
        self.shell_tool = shell_tool
        self.file_path = file_path

        self.agent = initialize_agent(
            tools=[shell_tool,Tool(name="write_code_to_file", func=write_code_to_file,description="Use this function to write the code to a file")],
            llm=self.llm.llm,  # Access the underlying LangChain LLM
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
                # Start of Selection
                max_iterations=20 if self.llm.is_strong() else 10,
            early_stopping_method="force",
            return_intermediate_steps=False,
            agent_kwargs={
                "prefix": """You are an AI Software Engineer agent that has 10 years experience in python development.
                You are given a task to write code, execute and solve the problem. 
                - You have access to the shell tool to execute commands and get real output.
                - You can install a new package if required. But always follow these below rlules:
                     - Before you install anything, verify that the package does not exist on the system and 
                     - Always find out which OS is running on the server to use the correct package manager.
                - When you are writing the code, always use the write_code_to_file tool with just the executable code and no other strings. 
                - Document the code really well and make sure it is executable.
                - Don't execute the code directly wit the shell tool. Execute it using python generated_code.py
                - Use `pip list` to check if the package is installed.
                Never assume external media files are available, for sound unless specified always generate it.
                
                Example interaction:
                Example 1:
                    Question: Write the code to print hello world
                    
                    Thought: I need to write the code to print hello world
                    Action: write_code_to_file
                    Action Input: print("Hello World)
                    Observation: The code has been written and saved to generated_code.py
                    
                    Thought: I need to execute the code to see if it works
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: SyntaxError: unterminated string literal (detected at line 1)

                    Thought: I need to fix the code to print hello world
                    Action: write_code_to_file
                    Action Input: print("Hello World")
                    Observation: The code has been written and saved to generated_code.py

                    Thought: I need to execute the code to see if it works
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: Hello World

                    Final Answer: The code has been written and executed successfully

                Example 2:
                    Question: Write the code to show the scatterplot for the iris dataset
                    
                    Thought: I need to write the code to show the scatterplot for the iris dataset
                    Action: write_code_to_file
                    Action Input: import matplotlib.pyplot as plt
                                import seaborn as sns
                                sns.scatterplot(x='sepal_length', y='sepal_width', data=iris)
                                plt.show()
                    Observation: The code has been written and saved to generated_code.py

                    Thought: I need to execute the code to see if it works
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: No module named 'seaborn'

                    Thought: I need to install the seaborn package
                    Action: shell
                    Action Input: pip install seaborn
                    Observation: The seaborn package has been installed

                    Thought: I need to execute the code to see if it works
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: The scatterplot has been shown

                    Final Answer: The scatterplot has been shown

                Example 3:
                    Question: Write the code to show a random cat image
                    Thought: I need to write the code to show a random cat image
                    Action: write_code_to_file
                    Action Input:
                    import requests
                    response = requests.get('https://api.thecatapi.com/v1/images/search')
                    image_url = response.json()[0]['url']
                    # write the image to a file
                    with open('cat_image.jpg', 'wb') as file:
                        file.write(requests.get(image_url).content)
                    # display the image
                    display(Image(filename='cat_image.jpg'))
                    Observation: The code has been written and saved to generated_code.py
                    Thought: I need to execute the code to see if it works
                    Action: shell
                    Action Input: python generated_code.py
                    Observation: The cat image has been shown
                    Final Answer: The cat image has been shown

                """
            }
        )

    def process_query(self, query: str) -> str:
        """Process a user query and return a response"""
        if not query:
            return "Empty query. Please ask a question."
        
        response = self.agent.invoke({"input": query})
        return response

if __name__ == "__main__":
    llm = LLMProviderFactory.get_provider(
            provider_type="ollama",
            model="llama3.2:latest",
            temperature=0
        ).llm
    print(llm)
    coder_tool = PythonCoder(llm, shell_tool)
    coder_tool.process_query("Write the code to show the scatterplot for the iris dataset")