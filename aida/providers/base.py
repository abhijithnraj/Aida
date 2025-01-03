from abc import ABC, abstractmethod
from typing import Optional, Any

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def __init__(self, model: str, temperature: float = 0):
        """Initialize the LLM provider with a model and temperature"""
        pass
    
    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt and return the response"""
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """Validate if the specified model is available"""
        pass
    
    @abstractmethod
    def is_strong(self) -> bool:
        """Return whether this model is considered strong enough to skip validation steps"""
        pass 