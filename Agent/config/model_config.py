"""
Model Configuration Loader
Loads and provides access to LLM model configurations from JSON file.
"""
import json
import os
from typing import Dict, Optional, Any
from pathlib import Path


class ModelConfig:
    """
    Singleton class to load and access LLM model configurations.
    """
    
    _instance = None
    _config_data = None
    _config_file = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelConfig._config_data is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if ModelConfig._config_file is None:
            # Get the directory where this file is located
            config_dir = Path(__file__).parent
            ModelConfig._config_file = config_dir / "llm_models.json"
        
        try:
            with open(ModelConfig._config_file, 'r') as f:
                ModelConfig._config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model configuration file not found: {ModelConfig._config_file}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model configuration file: {e}")
    
    def get_provider_default_model(self, provider: str) -> Optional[str]:
        """
        Get the default model for a provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'gemini')
            
        Returns:
            Default model name or None if provider not found
        """
        providers = ModelConfig._config_data.get('providers', {})
        provider_info = providers.get(provider.lower())
        return provider_info.get('default_model') if provider_info else None
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete information about a model.
        
        Args:
            model_name: Model name (e.g., 'gpt-4o', 'gemini-2.5-flash')
            
        Returns:
            Dictionary with model information or None if not found
        """
        models = ModelConfig._config_data.get('models', {})
        return models.get(model_name)
    
    def get_model_pricing(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary with 'input' and 'output' pricing per 1M tokens, or None
        """
        model_info = self.get_model_info(model_name)
        return model_info.get('pricing') if model_info else None
    
    def get_model_max_context(self, model_name: str) -> Optional[int]:
        """
        Get maximum context tokens for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Maximum context tokens or None if not found
        """
        model_info = self.get_model_info(model_name)
        return model_info.get('max_context_tokens') if model_info else None
    
    def get_all_models_by_provider(self, provider: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all models for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'gemini')
            
        Returns:
            Dictionary of model_name -> model_info for the provider
        """
        models = ModelConfig._config_data.get('models', {})
        return {
            name: info for name, info in models.items()
            if info.get('provider') == provider.lower()
        }
    
    def get_all_providers(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available providers.
        
        Returns:
            Dictionary of provider information
        """
        return ModelConfig._config_data.get('providers', {})
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available models.
        
        Returns:
            Dictionary of all models
        """
        return ModelConfig._config_data.get('models', {})
    
    def get_pricing_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Get pricing dictionary for all models (compatible with legacy code).
        
        Returns:
            Dictionary mapping model_name -> {'input': float, 'output': float}
        """
        models = ModelConfig._config_data.get('models', {})
        pricing_dict = {}
        for model_name, model_info in models.items():
            if 'pricing' in model_info:
                pricing_dict[model_name] = model_info['pricing']
        return pricing_dict
    
    def get_max_context_dict(self) -> Dict[str, int]:
        """
        Get max context dictionary for all models (compatible with legacy code).
        
        Returns:
            Dictionary mapping model_name -> max_context_tokens
        """
        models = ModelConfig._config_data.get('models', {})
        max_context_dict = {}
        for model_name, model_info in models.items():
            if 'max_context_tokens' in model_info:
                max_context_dict[model_name] = model_info['max_context_tokens']
        return max_context_dict
    
    def reload_config(self):
        """Reload configuration from file (useful if file changes)."""
        ModelConfig._config_data = None
        self._load_config()

