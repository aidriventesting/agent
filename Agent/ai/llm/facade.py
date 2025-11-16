from typing import Any, Dict, List, Optional

from robot.api import logger
from Agent.utilities._jsonutils import extract_json_safely
from Agent.ai.llm._factory import LLMClientFactory


class UnifiedLLMFacade:
    """Single entrypoint for all LLMs via one simple API.

    Hides provider/model selection and response parsing behind send_request_and_parse_response.
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None) -> None:
        self._client = LLMClientFactory.create_client(provider, model=model)

    def send_ai_request_and_return_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sends a request to the AI model and returns a parsed JSON response."""
        logger.debug("🚀 Sending request to AI model...")
        response = self._client.create_chat_completion(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs,
        )
        logger.debug("📥 Raw AI response received.")
        formatted = self._client.format_response(response)
        content = formatted.get("content", "{}")
        logger.debug(f"   Raw content: {content}")
        parsed = extract_json_safely(content)
        logger.debug(f"✅ Parsed JSON response: {parsed}")
        return parsed
