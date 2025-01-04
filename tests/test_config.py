import os
import pytest
from pathlib import Path
from aida.config import AidaConfig
from argparse import Namespace
import yaml
import tempfile

def test_default_config():
    """Test default configuration values"""
    config = AidaConfig()
    assert config.core_provider == "gemini"
    assert config.core_model == "gemini-1.5-flash"
    assert config.preprocessor_provider == "gemini"
    assert config.preprocessor_model == "gemini-1.5-flash"
    assert config.debug is False

def test_config_from_yaml():
    """Test loading configuration from YAML file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'core_provider': 'test-provider',
            'core_model': 'test-model',
            'preprocessor_provider': 'test-prep-provider',
            'preprocessor_model': 'test-prep-model',
            'debug': True
        }, f)
    
    config = AidaConfig.from_file(Path(f.name))
    os.unlink(f.name)
    
    assert config.core_provider == "test-provider"
    assert config.core_model == "test-model"
    assert config.preprocessor_provider == "test-prep-provider"
    assert config.preprocessor_model == "test-prep-model"
    assert config.debug is True

def test_nonexistent_config_file():
    """Test behavior with nonexistent config file"""
    config = AidaConfig.from_file(Path("nonexistent.yaml"))
    assert config.core_provider == "gemini"  # Should use defaults
    assert config.core_model == "gemini-1.5-flash"
    assert config.preprocessor_provider == "gemini"
    assert config.preprocessor_model == "gemini-1.5-flash"
    assert config.debug is False

def test_env_var_override():
    """Test environment variable overrides"""
    os.environ["AIDA_CORE_PROVIDER"] = "env-provider"
    os.environ["AIDA_CORE_MODEL"] = "env-model"
    os.environ["AIDA_PREPROCESSOR_PROVIDER"] = "env-prep-provider"
    os.environ["AIDA_PREPROCESSOR_MODEL"] = "env-prep-model"
    
    config = AidaConfig()
    config.update_from_args(Namespace())
    
    assert config.core_provider == "env-provider"
    assert config.core_model == "env-model"
    assert config.preprocessor_provider == "env-prep-provider"
    assert config.preprocessor_model == "env-prep-model"
    
    # Cleanup
    del os.environ["AIDA_CORE_PROVIDER"]
    del os.environ["AIDA_CORE_MODEL"]
    del os.environ["AIDA_PREPROCESSOR_PROVIDER"]
    del os.environ["AIDA_PREPROCESSOR_MODEL"]

def test_args_override():
    """Test command line argument overrides"""
    args = Namespace(
        provider="args-provider",
        core_model="args-model",
        preprocessor_model="args-prep-model",
        debug=True
    )
    
    config = AidaConfig()
    config.update_from_args(args)
    
    assert config.core_provider == "args-provider"
    assert config.core_model == "args-model"
    assert config.preprocessor_provider == "args-provider"
    assert config.preprocessor_model == "args-prep-model"
    assert config.debug is True

def test_config_path_from_env():
    """Test loading config file path from environment variable"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'core_provider': 'env-path-provider',
            'core_model': 'env-path-model'
        }, f)
    
    os.environ["AIDA_CONFIG_PATH"] = f.name
    config = AidaConfig.from_file()
    os.unlink(f.name)
    del os.environ["AIDA_CONFIG_PATH"]
    
    assert config.core_provider == "env-path-provider"
    assert config.core_model == "env-path-model"

def test_default_config_locations():
    """Test loading from default config locations"""
    # Create a temporary config in one of the default locations
    config_dir = Path.home() / ".config" / "aida"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    
    config_data = {
        'core_provider': 'default-path-provider',
        'core_model': 'default-path-model'
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    config = AidaConfig.from_file()
    
    # Cleanup
    config_file.unlink()
    
    assert config.core_provider == "default-path-provider"
    assert config.core_model == "default-path-model" 