from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Optional

@dataclass
class AidaConfig:
    core_model: str = "llama2"
    preprocessor_model: str = "llama2"
    debug: bool = False
    
    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> 'AidaConfig':
        """Load configuration from file"""
        if config_path is None:
            config_path = Path.home() / ".aida" / "config.yml"
            
        if not config_path.exists():
            return cls()
            
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        return cls(
            core_model=config_data.get("core_model", cls.core_model),
            preprocessor_model=config_data.get("preprocessor_model", cls.preprocessor_model),
            debug=config_data.get("debug", cls.debug)
        )
    
    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file"""
        if config_path is None:
            config_path = Path.home() / ".aida" / "config.yml"
            
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "core_model": self.core_model,
            "preprocessor_model": self.preprocessor_model,
            "debug": self.debug
        }
        
        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f)
            
    def update_from_args(self, args) -> 'AidaConfig':
        """Update config from command line arguments"""
        if hasattr(args, "core_model") and args.core_model:
            self.core_model = args.core_model
        if hasattr(args, "preprocessor_model") and args.preprocessor_model:
            self.preprocessor_model = args.preprocessor_model
        if hasattr(args, "debug"):
            self.debug = args.debug
        return self 