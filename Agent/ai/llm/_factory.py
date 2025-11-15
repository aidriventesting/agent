from typing import Optional, Callable, Dict
from Agent.ai.llm._baseclient import BaseLLMClient
from Agent.ai.llm._openaiclient import OpenAIClient
from Agent.ai.llm._anthropic import AnthropicClient
from Agent.ai.llm._gemini import GeminiClient
from Agent.config.model_config import ModelConfig
from Agent.config.config import Config


class LLMClientFactory:
    """
    Factory class to create and return LLM client instances.
    Supports multiple providers: OpenAI, Anthropic (Claude), and Google Gemini.
    """

    _model_config = ModelConfig()
    DEFAULT_MODELS = {
        "openai": _model_config.get_provider_default_model("openai"),
        "anthropic": _model_config.get_provider_default_model("anthropic"),
        "gemini": _model_config.get_provider_default_model("gemini"),
    }

    # Registry of provider factories for extension
    _registry: Dict[str, Callable[[Optional[str], Config], BaseLLMClient]] = {}

    # Pre-register built-in providers
    _registry.update({
        "openai": lambda model, cfg: OpenAIClient(model=model or LLMClientFactory.DEFAULT_MODELS.get("openai"), api_key=cfg.OPENAI_API_KEY),
        "anthropic": lambda model, cfg: AnthropicClient(model=model or LLMClientFactory.DEFAULT_MODELS.get("anthropic"), api_key=cfg.ANTHROPIC_API_KEY),
        "claude": lambda model, cfg: AnthropicClient(model=model or LLMClientFactory.DEFAULT_MODELS.get("anthropic"), api_key=cfg.ANTHROPIC_API_KEY),
        "gemini": lambda model, cfg: GeminiClient(model=model or LLMClientFactory.DEFAULT_MODELS.get("gemini")),
        "google": lambda model, cfg: GeminiClient(model=model or LLMClientFactory.DEFAULT_MODELS.get("gemini")),
    })

    @staticmethod
    def register_client(name: str, factory: Callable[[Optional[str], Config], BaseLLMClient]) -> None:
        """Register a custom provider factory callable.

        The callable receives (model, config) and must return a BaseLLMClient instance.
        """
        LLMClientFactory._registry[name.lower()] = factory

    @staticmethod
    def list_providers() -> Dict[str, str]:
        return {name: "registered" for name in LLMClientFactory._registry.keys()}

    @staticmethod
    def create_client(
        client_name: str = "openai",
        model: Optional[str] = None,
    ) -> BaseLLMClient:
        client_name_lower = client_name.lower()
        config = Config()

        factory = LLMClientFactory._registry.get(client_name_lower)
        if not factory:
            supported = ", ".join(sorted(LLMClientFactory._registry.keys()))
            raise ValueError(
                f"Unsupported LLM client: {client_name}. Registered providers: {supported}"
            )
        return factory(model, config)


