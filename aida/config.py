from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from typing import Optional

@dataclass
class AidaConfig:
    """Configuration for AIDA"""
    
    # Core LLM settings
    core_provider: str = "ollama"
    core_model: str = "llama3.2:3b"
    
    # Preprocessor LLM settings
    preprocessor_provider: str = "gemini"
    preprocessor_model: str = "gemini-1.5-flash"
    
    # Debug mode
    debug: bool = False
    
    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> 'AidaConfig':
        """Load configuration from a YAML file
        
        Args:
            config_path: Path to config file. If None, tries default locations
            
        Returns:
            AidaConfig instance
        """
        # Try environment variable first
        if not config_path:
            env_path = os.getenv("AIDA_CONFIG_PATH")
            if env_path:
                config_path = Path(env_path)
                
        # Try default locations
        if not config_path:
            default_paths = [
                Path.home() / ".config" / "aida" / "config.yaml",
                Path.home() / ".aida.yaml",
                Path("config.yaml")
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break
        
        # If no config file found, return default config
        if not config_path or not config_path.exists():
            return cls()
            
        # Load and parse YAML
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        return cls(
            core_provider=config_data.get("core_provider", cls.core_provider),
            core_model=config_data.get("core_model", cls.core_model),
            preprocessor_provider=config_data.get("preprocessor_provider", cls.preprocessor_provider),
            preprocessor_model=config_data.get("preprocessor_model", cls.preprocessor_model),
            debug=config_data.get("debug", cls.debug)
        )
    
    def update_from_args(self, args) -> None:
        """Update config from command line arguments
        
        Args:
            args: Parsed command line arguments
        """
        if hasattr(args, "core_model") and args.core_model:
            self.core_model = args.core_model
            
        if hasattr(args, "preprocessor_model") and args.preprocessor_model:
            self.preprocessor_model = args.preprocessor_model
            
        if hasattr(args, "debug") and args.debug:
            self.debug = args.debug
        
        if hasattr(args, "provider") and args.provider:
            self.core_provider = args.provider
            self.preprocessor_provider = args.provider
            
        # Update from environment variables
        self.core_provider = os.getenv("AIDA_CORE_PROVIDER", self.core_provider)
        self.core_model = os.getenv("AIDA_CORE_MODEL", self.core_model)
        self.preprocessor_provider = os.getenv("AIDA_PREPROCESSOR_PROVIDER", self.preprocessor_provider)
        self.preprocessor_model = os.getenv("AIDA_PREPROCESSOR_MODEL", self.preprocessor_model)
        
        if not hasattr(args, "provider") or not args.provider:
            self.core_provider = os.getenv("AIDA_PROVIDER", self.core_provider)
            self.preprocessor_provider = os.getenv("AIDA_PROVIDER", self.preprocessor_provider)
