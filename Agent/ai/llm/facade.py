from typing import Any, Dict, List, Optional
import json

from robot.api import logger
from Agent.utilities._jsonutils import extract_json_safely
from Agent.ai.llm._factory import LLMClientFactory
from Agent.ai.llm._openaiclient import OpenAIClient
from Agent.ai.llm._ollama import OllamaClient


class UnifiedLLMFacade:
    """Single entrypoint for all LLMs via one simple API.

    Hides provider/model selection and response parsing behind send_request_and_parse_response.
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None) -> None:
        self._client = LLMClientFactory.create_client(provider, model=model)
        self._provider = provider.lower()

    #TODO : unite the two methods(send_ai_request_and_return_response and send_ai_request_with_tools) into one
    def send_ai_request_and_return_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sends a request to the AI model and returns a parsed JSON response."""
        logger.debug("🚀 Sending request to AI model...")
        
        # Only OpenAI and Ollama support response_format parameter
        if isinstance(self._client, (OpenAIClient, OllamaClient)):
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self._client.create_chat_completion(
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        logger.debug("📥 Raw AI response received.")
        formatted = self._client.format_response(response)
        content = formatted.get("content", "{}")
        logger.debug(f"   Raw content: {content}")
        parsed = extract_json_safely(content)
        logger.debug(f"✅ Parsed JSON response: {parsed}")
        return parsed

    def send_ai_request_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sends a request to the AI model with tool calling enabled."""
        logger.debug("🚀 Sending request to AI model with tools...")
        logger.debug(f"   Tools: {[t['function']['name'] for t in tools]}")
        
        response = self._client.create_chat_completion(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        
        logger.debug("📥 Raw AI response received.")
        formatted = self._client.format_response(response)
        
        # Check if response has tool calls
        if "tool_calls" in formatted and formatted["tool_calls"]:
            logger.debug(f"✅ Tool calls received: {len(formatted['tool_calls'])} call(s)")
            # Parse the arguments from JSON string to dict for easier use
            for tool_call in formatted["tool_calls"]:
                if isinstance(tool_call["function"]["arguments"], str):
                    try:
                        tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        logger.warn(f"Failed to parse tool call arguments: {e}")
                        tool_call["function"]["arguments"] = {}
            return formatted
        else:
            # Fallback to content if no tool calls
            logger.debug("⚠️ No tool calls in response, returning content")
            return formatted

