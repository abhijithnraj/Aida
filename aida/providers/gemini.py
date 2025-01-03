import os
import logging
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from .base import LLMProvider

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation"""
    
    AVAILABLE_MODELS = ["gemini-1.5-flash","gemini-2.0-flash-exp","gemini-2.0-flash-thinking-exp","gemini-2.0-flash-exp"]
    
    def __init__(self, model: str, temperature: float = 0):
        """Initialize the Gemini provider with a model and temperature"""
        self.model = model
        self.temperature = temperature
        
        if not self.validate_model(model):
            raise ValueError(f"Model '{model}' is not available in Gemini. Available models: {self.AVAILABLE_MODELS}")
            
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider")
            
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key
        )
    
    def invoke(self, prompt: str) -> Any:
        """Invoke the Gemini LLM with a prompt"""
        return self.llm.invoke(prompt)
    
    def validate_model(self, model: str) -> bool:
        """Validate if the specified model is available in Gemini"""
        return model in self.AVAILABLE_MODELS 