from openai import OpenAI
from openai.types.chat import ChatCompletion
from typing import Optional, Dict, List, Union
from robot.api import logger
from Agent.ai.llm._baseclient import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key=None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        base_backoff: int = 2,
    ):
        self.api_key: str = api_key
        if not self.api_key:
            from Agent.config.config import Config
            config = Config()
            self.api_key = config.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or in the environment variables.")

        self.default_model = model
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.client = OpenAI(api_key=self.api_key)

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> Optional[ChatCompletion]:
        try:
            self._validate_parameters(temperature, top_p)

            params = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                **kwargs
            }

            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice

            response = self.client.chat.completions.create(**params)
            logger.debug(f"OpenAI API call successful. Tokens used: {response.usage.total_tokens}", True)
            logger.debug(f"messages: {response}")
            return response
        except Exception as e:
            logger.error(f"OpenAI API Error: {str(e)}", True)
            raise

    def _validate_parameters(self, temperature: float, top_p: float):
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
        include_reason: bool = False,
    ) -> Dict[str, Union[str, int]]:
        if not response or not response.choices:
            logger.error(f"Invalid response or no choices in the response", True)
            return {}

        result = {
            "content": response.choices[0].message.content or "",
        }

        # Extract tool calls if present
        if response.choices[0].message.tool_calls:
            tool_calls = []
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            result["tool_calls"] = tool_calls

        if include_tokens:
            logger.debug(f"Tokens used: {response.usage}")
            result.update(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        if include_reason:
            logger.debug(f"Finish reason: {response.choices[0].finish_reason}")
            result["finish_reason"] = response.choices[0].finish_reason

        return result


