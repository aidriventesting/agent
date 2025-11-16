from anthropic import Anthropic, APIError
from typing import Optional, Dict, List, Union
from robot.api import logger
from Agent.ai.llm._baseclient import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_retries: int = 3,
    ):
        self.api_key: str = api_key

        if not self.api_key:
            from Agent.config.config import Config
            config = Config()
            self.api_key = config.ANTHROPIC_API_KEY

        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or in the environment variables.")

        self.default_model = model
        self.max_retries = max_retries
        self.client = Anthropic(api_key=self.api_key, max_retries=max_retries)

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ):
        try:
            self._validate_parameters(temperature, top_p)

            system_message = None
            user_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content")
                else:
                    transformed_content = self._transform_content(msg.get("content"))
                    user_messages.append({"role": msg.get("role"), "content": transformed_content})

            api_params = {
                "model": model or self.default_model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                **kwargs,
            }
            if temperature != 1.0:
                api_params["temperature"] = temperature
            elif top_p != 1.0:
                api_params["top_p"] = top_p
            else:
                api_params["temperature"] = temperature

            if system_message:
                api_params["system"] = system_message

            response = self.client.messages.create(**api_params)

            logger.debug(
                f"Anthropic API call successful. Tokens used: {response.usage.input_tokens + response.usage.output_tokens}",
                True,
            )
            logger.debug(f"Response: {response}")

            return response

        except APIError as e:
            logger.error(f"Anthropic API Error: {str(e)}", True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", True)
            raise

    def _transform_content(self, content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        transformed = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                transformed.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "image_url":
                image_url_data = item.get("image_url", {})
                if isinstance(image_url_data, dict) and "url" in image_url_data:
                    url = image_url_data["url"]
                    if url.startswith("data:"):
                        try:
                            header, data = url.split(",", 1)
                            media_type = header.split(";")[0].split(":")[1]
                            transformed.append(
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": media_type, "data": data},
                                }
                            )
                        except (ValueError, IndexError) as e:
                            logger.error(f"Invalid base64 image URL format: {e}")
                            continue
                    else:
                        transformed.append({"type": "image", "source": {"type": "url", "url": url}})
            elif item_type == "image":
                if "source" in item:
                    transformed.append(item)
                else:
                    logger.warn("Image item missing 'source' field")
            else:
                transformed.append(item)
        return transformed if transformed else content

    def _validate_parameters(self, temperature: float, top_p: float):
        if not (0 <= temperature <= 1):
            logger.error(f"Invalid temperature {temperature}. Must be between 0 and 1 for Anthropic")
            raise ValueError(f"Invalid temperature {temperature}. Must be between 0 and 1")
        if not (0 <= top_p <= 1):
            logger.error(f"Invalid top_p {top_p}. Must be between 0 and 1")
            raise ValueError(f"Invalid top_p {top_p}. Must be between 0 and 1")

    def format_response(
        self,
        response,
        include_tokens: bool = True,
        include_reason: bool = False,
    ) -> Dict[str, Union[str, int]]:
        if not response or not response.content:
            logger.error(f"Invalid response or no content in the response", True)
            return {}

        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text

        result = {
            "content": content_text,
        }

        if include_tokens and response.usage:
            logger.debug(
                f"Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}"
            )
            result.update(
                {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                }
            )

        if include_reason:
            logger.debug(f"Stop reason: {response.stop_reason}")
            result["finish_reason"] = response.stop_reason

        return result


