from typing import Any, Dict, List, Optional
from Agent.ai.llm.facade import UnifiedLLMFacade
from Agent.ai._promptcomposer import AgentPromptComposer


class AiConnector:
    """AI connector for LLM requests."""

    def __init__(self, provider: str = "openai", model: Optional[str] = "gpt-4o") -> None:
        self.llm = UnifiedLLMFacade(provider=provider, model=model)
        self.prompt = AgentPromptComposer()

    def ask_ai_do(
        self,
        instruction: str,
        ui_elements: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        messages = self.prompt.compose_do_messages(instruction, ui_elements)
        tools = self.prompt.get_do_tools()
        return self.llm.send_ai_request_with_tools(
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature
        )

    def ask_ai_visual_check(
        self,
        instruction: str,
        image_url: str,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        messages = self.prompt.compose_visual_check_messages(instruction, image_url)
        tools = self.prompt.get_visual_check_tools()
        return self.llm.send_ai_request_with_tools(
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature
        )



