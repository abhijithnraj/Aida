from typing import Type
from .base import LLMProvider
from .ollama import OllamaProvider
from .gemini import GeminiProvider

class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    
    _providers = {
        "ollama": OllamaProvider,
        "gemini": GeminiProvider
    }
    
    @classmethod
    def get_provider(cls, provider_type: str, model: str, temperature: float = 0) -> LLMProvider:
        """Get an instance of the specified LLM provider
        
        Args:
            provider_type: Type of provider ("ollama" or "gemini")
            model: Name of the model to use
            temperature: Temperature parameter for the model
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If the provider type is not supported
        """
        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider type: {provider_type}. Available providers: {list(cls._providers.keys())}")
            
        return provider_class(model=model, temperature=temperature)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get a list of available provider types"""
        return list(cls._providers.keys()) 