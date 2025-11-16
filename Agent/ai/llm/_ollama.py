from openai import OpenAI
from openai.types.chat import ChatCompletion
from typing import Optional, Dict, List, Union
import os
from robot.api import logger
from Agent.ai.llm._baseclient import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    Ollama client for running local LLMs.
    
    Ollama provides a local server that's compatible with OpenAI's API format,
    making it easy to run models like Llama, Mistral, CodeLlama, etc. locally.
    
    No API key required - just needs Ollama running locally.
    Default endpoint: http://localhost:11434/v1
    
    Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    
    def __init__(
        self, 
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        max_retries: int = 3,
    ):
        self.default_model = model
        self.base_url = base_url
        self.max_retries = max_retries
        
        # Ollama doesn't require API key, but OpenAI SDK needs something
        # Using "ollama" as a placeholder
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Dummy key, not used by Ollama
            max_retries=max_retries
        )
        
        logger.debug(f"Ollama client initialized with base_url: {base_url}")

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Optional[ChatCompletion]:
        """
        Create a chat completion using local Ollama server.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (if None, uses default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            OpenAI ChatCompletion object (Ollama is compatible)
        """
        try:
            self._validate_parameters(temperature, top_p)
            
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Log usage
            logger.debug(
                f"Ollama API call successful. Tokens used: {response.usage.total_tokens}",
                True
            )
            logger.debug(f"Response: {response}")
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg:
                logger.error(
                    "Cannot connect to Ollama server. Is Ollama running? "
                    "Start it with: ollama serve",
                    True
                )
            else:
                logger.error(f"Ollama API Error: {error_msg}", True)
            raise

    def _validate_parameters(self, temperature: float, top_p: float):
        """Validate API parameters."""
        if not (0 <= temperature <= 2):
            logger.error(f"Invalid temperature {temperature}. Must be between 0 and 2")
            raise ValueError(f"Invalid temperature {temperature}. Must be between 0 and 2")
        if not (0 <= top_p <= 1):
            logger.error(f"Invalid top_p {top_p}. Must be between 0 and 1")
            raise ValueError(f"Invalid top_p {top_p}. Must be between 0 and 1")

    def format_response(
        self, 
        response: ChatCompletion,
        include_tokens: bool = True,
        include_reason: bool = False
    ) -> Dict[str, Union[str, int]]:
        """
        Format Ollama response to a standardized dictionary.
        
        Args:
            response: OpenAI ChatCompletion object (from Ollama)
            include_tokens: Whether to include token usage information
            include_reason: Whether to include stop reason
            
        Returns:
            Standardized response dictionary
        """
        if not response or not response.choices:
            logger.error(f"Invalid response or no choices in the response", True)
            return {}
            
        result = {
            "content": response.choices[0].message.content,
        }
        
        if include_tokens and response.usage:
            logger.debug(f"Tokens used: {response.usage}")
            result.update({
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            })
            
        if include_reason:
            logger.debug(f"Finish reason: {response.choices[0].finish_reason}")
            result["finish_reason"] = response.choices[0].finish_reason
            
        return result

