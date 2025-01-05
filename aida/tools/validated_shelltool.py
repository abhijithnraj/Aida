from langchain_community.tools import ShellTool
from langchain.agents import Tool
class ValidatedShellTool:
    """A wrapper around ShellTool that requires user validation before execution"""
    def __init__(self):
        self.shell_tool = ShellTool()
        self.result = None
    
    def run(self, command: str) -> str:
            # Terminal validation
            print(f"\nCommand to execute: {command}")
            user_input = input("Do you want to execute this command? (y/n/modify): ").lower().strip()
            
            if user_input == 'modify':
                command = input("Enter the modified command: ").strip()
                if not command:
                    return "Command execution cancelled by user"
            elif user_input != 'y':
                return "Command execution cancelled by user"
            
            return self.shell_tool.run(command)
        

shell_tool= Tool(
            name="shell",
            func=ValidatedShellTool().run,
            description="""Execute shell commands on the server. Use this tool to run commands and get their output.
            The command will be shown to the user for validation before execution.
            Example:
            Action: shell
            Action Input: who
            Observation: user1    pts/0    2024-01-31 10:00 (:0)
            Thought: The 'who' command shows user1 is logged in
            """
        )