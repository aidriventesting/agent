from typing import List, Dict, Optional, Any
from Agent.platforms import DeviceConnector


class AgentPromptComposer:
    """Builds prompts for agent actions and visual checks."""

    def __init__(self) -> None:
        self.catalog = AgentKeywordCatalog()

    def compose_do_messages(
        self,
        instruction: str,
        ui_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build DO action messages using tool calling approach."""
        ui_text = self._render_ui_candidates(ui_elements)
        system_content = (
            "You are a MOBILE app test automation engine (Appium).\n"
            "Your job: analyze the instruction and call the appropriate function to interact with the mobile UI.\n\n"
            f"{self.catalog._render_catalog_text()}\n\n"
            "IMPORTANT: You are working with MOBILE apps (Android/iOS), NOT web browsers.\n"
            "Select the element index from the numbered list by calling the appropriate function."
        )
        user_content = f"Instruction: {instruction}\n\nMobile UI Elements:\n{ui_text}"
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def get_do_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions for DO actions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "tap_element",
                    "description": "Tap or click on a mobile UI element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_index": {
                                "type": "integer",
                                "description": "The index number of the element from the UI elements list (1-based)",
                                "minimum": 1
                            }
                        },
                        "required": ["element_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "input_text",
                    "description": "Clear and input text into a mobile UI element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element_index": {
                                "type": "integer",
                                "description": "The index number of the element from the UI elements list (1-based)",
                                "minimum": 1
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to input into the element"
                            }
                        },
                        "required": ["element_index", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll_down",
                    "description": "Scroll down the mobile screen",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    def compose_visual_check_messages(
        self,
        instruction: str,
        image_url: str,
    ) -> List[Dict[str, Any]]:
        """Build visual check messages using tool calling approach."""
        system_content = (
            "You are a mobile app visual verification engine. "
            "Analyze the screenshot and verify if it matches the instruction. "
            "Use the verify_visual_match function to report your findings."
        )
        user_content = [
            {"type": "text", "text": f"Verify: {instruction}"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def get_visual_check_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions for visual check actions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "verify_visual_match",
                    "description": "Report the results of visual verification against the given instruction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "verification_result": {
                                "type": "boolean",
                                "description": "Whether the screenshot matches the instruction (true) or not (false)"
                            },
                            "confidence_score": {
                                "type": "number",
                                "description": "Confidence level of the verification from 0.0 (no confidence) to 1.0 (completely confident)",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "analysis": {
                                "type": "string",
                                "description": "Detailed analysis explaining why the verification passed or failed"
                            },
                            "found_elements": {
                                "type": "array",
                                "description": "Optional list of UI elements found in the screenshot",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "element_type": {"type": "string"},
                                        "description": {"type": "string"},
                                        "location": {"type": "string"},
                                        "confidence": {"type": "number"}
                                    }
                                }
                            },
                            "issues": {
                                "type": "array",
                                "description": "Optional list of issues or problems found",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["verification_result", "confidence_score", "analysis"]
                    }
                }
            }
        ]

    def _render_ui_candidates(self, ui_elements: Optional[List[Dict[str, Any]]]) -> str:
        if not ui_elements:
            return "(no UI elements found)"
        
        rendered = []
        for i, el in enumerate(ui_elements[:20], 1):
            parts = []
            if el.get("text"):
                parts.append(f"text='{el['text']}'")
            if el.get("resource_id"):
                parts.append(f"id='{el['resource_id']}'")
            if el.get("content_desc"):
                parts.append(f"desc='{el['content_desc']}'")
            
            rendered.append(f"{i}. {' | '.join(parts) if parts else el.get('class_name', 'unknown')}")
        
        return "\n".join(rendered)





class AgentKeywordCatalog:
    """Catalog of available mobile actions."""

    def __init__(self) -> None:
        self.actions = [
            ("tap", "Click Element", "Tap/click an element"),
            ("input", "Input Text", "Clear and type text into element"),
            ("scroll_down", "Swipe", "Scroll down the screen"),
        ]

    def _render_catalog_text(self) -> str:
        lines = ["Available actions:"]
        for action, rf_kw, desc in self.actions:
            lines.append(f"- {action}: {desc}")
        return "\n".join(lines)
