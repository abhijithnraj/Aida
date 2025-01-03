import subprocess
import logging
from typing import Any
from langchain_ollama import ChatOllama
from .base import LLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation"""
    
    def __init__(self, model: str, temperature: float = 0):
        """Initialize the Ollama provider with a model and temperature"""
        self.model = model
        self.temperature = temperature
        if not self.validate_model(model):
            raise ValueError(f"Model '{model}' is not available in Ollama")
        self.llm = ChatOllama(model=model, temperature=temperature)
    
    def invoke(self, prompt: str) -> Any:
        """Invoke the Ollama LLM with a prompt"""
        return self.llm.invoke(prompt)
    
    def validate_model(self, model: str) -> bool:
        """Validate if the specified model is available in Ollama"""
        logger.info("Checking available Ollama models...")
        process = subprocess.Popen(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        
        if process.returncode != 0:
            logger.error("Failed to run 'ollama list': %s", error.strip())
            return False
            
        logger.info("Available models:\n%s", output)
        available_models = [model.split()[0] for model in output.splitlines()[1:] if model.strip()]
        return model in available_models 