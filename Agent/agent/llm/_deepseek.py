from anthropic import Anthropic, APIError
from typing import Optional, Dict, List, Union
import os
from Agent.utilities._logger import RobotCustomLogger
from Agent.agent.llm._baseclient import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek client using Anthropic API compatibility.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_retries: int = 3,
    ):
        self.logger = RobotCustomLogger()
        self.api_key: str = api_key
        
        if not self.api_key:
            from Agent.config.config import Config
            config = Config()
            self.api_key = config.DEEPSEEK_API_KEY
            self.logger.info(f"API key loaded from config file")
            
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or in the environment variables.")
            
        self.default_model = model
        self.max_retries = max_retries
        
        # Initialize Anthropic client with DeepSeek's base URL
        self.client = Anthropic(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/anthropic",
            max_retries=max_retries
        )

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ):
        """
        Create a chat completion using DeepSeek's Anthropic-compatible API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (if None, uses default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2.0 for DeepSeek)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Anthropic Message object (from DeepSeek)
        """
        try:
            self._validate_parameters(temperature, top_p)
            
            # DeepSeek follows Anthropic's format - separate system messages
            system_message = None
            user_messages = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content")
                else:
                    transformed_content = self._transform_content(msg.get("content"))
                    user_messages.append({
                        "role": msg.get("role"),
                        "content": transformed_content
                    })
            
            # Prepare API call parameters
            api_params = {
                "model": model or self.default_model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                **kwargs
            }
            # Only add temperature or top_p, not both
            if temperature != 1.0:
                api_params["temperature"] = temperature
            elif top_p != 1.0:
                api_params["top_p"] = top_p
            else:
                # If both are default, use temperature
                api_params["temperature"] = temperature
            
            if system_message:
                api_params["system"] = system_message
            
            response = self.client.messages.create(**api_params)
            
            # Log usage
            self.logger.info(
                f"DeepSeek API call successful. Tokens used: {response.usage.input_tokens + response.usage.output_tokens}",
                True
            )
            self.logger.info(f"Response: {response}")
            
            return response
            
        except APIError as e:
            self.logger.error(f"DeepSeek API Error: {str(e)}", True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", True)
            raise

    def _transform_content(self, content):
        """
        Transform content to DeepSeek's format (follows Anthropic format).
        
        Supports:
        - String content (passed through)
        - List content with text and images (if supported)
        - OpenAI-style image_url format -> Anthropic's image format
        - Native Anthropic image format (passed through)
        
        Args:
            content: String or list of content items
            
        Returns:
            Transformed content in Anthropic/DeepSeek format
        """
        # If content is a simple string, return as-is
        if isinstance(content, str):
            return content
        
        # If content is not a list, return as-is
        if not isinstance(content, list):
            return content
        
        # Transform list content
        transformed = []
        for item in content:
            if not isinstance(item, dict):
                continue
                
            item_type = item.get("type")
            
            # Handle text content
            if item_type == "text":
                transformed.append({
                    "type": "text",
                    "text": item.get("text", "")
                })
            
            # Handle OpenAI-style image_url format
            # Note: DeepSeek's docs show image support as "Not Supported"
            # but we'll include the transformation for future compatibility
            elif item_type == "image_url":
                image_url_data = item.get("image_url", {})
                if isinstance(image_url_data, dict) and "url" in image_url_data:
                    url = image_url_data["url"]
                    
                    # Check if it's a base64 data URL
                    if url.startswith("data:"):
                        # Extract media type and base64 data
                        try:
                            header, data = url.split(",", 1)
                            media_type = header.split(";")[0].split(":")[1]
                            transformed.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            })
                        except (ValueError, IndexError) as e:
                            self.logger.error(f"Invalid base64 image URL format: {e}")
                            continue
                    else:
                        # Regular URL
                        transformed.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url
                            }
                        })
            
            # Handle native Anthropic image format
            elif item_type == "image":
                if "source" in item:
                    transformed.append(item)
                else:
                    self.logger.warning("Image item missing 'source' field")
            
            # Pass through any other content types
            else:
                transformed.append(item)
        
        return transformed if transformed else content

    def _validate_parameters(self, temperature: float, top_p: float):
        """Validate API parameters. DeepSeek supports temperature 0-2.0."""
        if not (0 <= temperature <= 2.0):
            self.logger.error(f"Invalid temperature {temperature}. Must be between 0 and 2.0 for DeepSeek")
            raise ValueError(f"Invalid temperature {temperature}. Must be between 0 and 2.0")
        if not (0 <= top_p <= 1):
            self.logger.error(f"Invalid top_p {top_p}. Must be between 0 and 1")
            raise ValueError(f"Invalid top_p {top_p}. Must be between 0 and 1")

    def format_response(
        self, 
        response,
        include_tokens: bool = True,
        include_reason: bool = False
    ) -> Dict[str, Union[str, int]]:
        """
        Format DeepSeek response to a standardized dictionary.
        
        Args:
            response: Anthropic Message object (from DeepSeek)
            include_tokens: Whether to include token usage information
            include_reason: Whether to include stop reason
            
        Returns:
            Standardized response dictionary
        """
        if not response or not response.content:
            self.logger.error(f"Invalid response or no content in the response", True)
            return {}
        
        # Extract text content (follows Anthropic's format)
        content_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                content_text += block.text
        
        result = {
            "content": content_text,
        }
        
        if include_tokens and response.usage:
            self.logger.info(f"Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
            result.update({
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            })
            
        if include_reason:
            self.logger.info(f"Stop reason: {response.stop_reason}")
            result["finish_reason"] = response.stop_reason
            
        return result

