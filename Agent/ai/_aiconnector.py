from typing import Any, Dict, List, Optional

from Agent.ai.llm.facade import UnifiedLLMFacade
from Agent.ai._promptcomposer import AgentPromptComposer
from robot.api import logger
from Agent.utilities.imguploader.imghandler import ImageUploader


class AiConnector:
    """AI connector only: accepts prepared messages, returns parsed JSON result."""

    def __init__(self, provider: str = "openai", model: Optional[str] = "gpt-4o-mini") -> None:
        self.llm = UnifiedLLMFacade(provider=provider, model=model)
        self.prompt = AgentPromptComposer(locale="fr")
        # Image uploader never crashes; fallbacks to base64 if no provider configured
        self.image_uploader = ImageUploader(service="auto")

    # ----------------------- Public API -----------------------
    def ask_ai_do(
        self,
        instruction: str,
        ui_elements: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:

        messages = self.prompt.compose_do_messages(
            instruction=instruction,
            ui_elements=ui_elements,
        )
        return self._run(messages, temperature=temperature)


    def ask_ai_visual_check(
        self,
        instruction: str,
        image_base_or_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        messages = self.prompt.compose_visual_check_messages(
            instruction=instruction,
            image_url=image_base_or_url,
        )
        return self._run(messages, temperature=temperature)

    # ----------------------- Internals -----------------------
    def _run(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> Dict[str, Any]:
        logger.debug("⏳ Sending prepared messages to AI...")
        result = self.llm.send_ai_request_and_return_response(messages, temperature=temperature)
        logger.debug(f"📦 AI response parsed: {result}")
        return result



