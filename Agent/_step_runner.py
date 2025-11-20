from typing import Any, Dict, List, Optional

from Agent.platforms import DeviceConnector
from Agent.ai._aiconnector import AiConnector
from Agent.utilities.imguploader.imghandler import ImageUploader
from robot.api import logger

class AgentStepRunner:
    """Orchestrates the Agent.Do and Agent.VisualCheck flows without relying on Robot Framework.

    This class encapsulates:
      - capturing the UI context and screenshots
      - composing prompts (Do/VisualCheck)
      - calling the LLM (strict JSON response)
      - executing actions and visual verification

    Objective: allow `AgentKeywords` to delegate cleanly, to facilitate
    architectural evolution without breaking existing functionality.
    """

    def __init__(self, llm_client: str = "openai", llm_model: str = "gpt-4o-mini", platform: Optional[DeviceConnector] = None) -> None:
        # Platform
        self.platform: DeviceConnector = platform or DeviceConnector()
        # Agent component
        self.agent = AiConnector(provider=llm_client, model=llm_model)
        self.image_uploader = ImageUploader(service="auto")
        
    # ----------------------- Public API -----------------------
    def do(self, instruction: str) -> None:
        logger.info(f"🚀 Starting Agent.Do: '{instruction}'")

        ui_candidates = self.platform.collect_ui_candidates()
        
        result = self.agent.ask_ai_do(
            instruction=instruction,
            ui_elements=ui_candidates,
            temperature=0,
        )

        logger.info(f"AI response: {result}")
        self._execute_do_from_tool_calls(result, ui_candidates, instruction)
        logger.info("✅ Agent.Do completed")

    def visual_check(self, instruction: str) -> None:
        logger.info(f"👁️ Starting Agent.VisualCheck with instruction: '{instruction}'")

        # Log to Robot Framework
        logger.debug("=" * 80)
        logger.debug("AGENT VISUAL CHECK STARTED")
        logger.debug("=" * 80)
        logger.debug(f"Instruction: {instruction}")
        logger.debug("Capturing screenshot for AI analysis...")

        # Capture screenshot
        logger.debug("📸 Capturing screenshot...")
        screenshot_base64 = self.platform.get_screenshot_base64()
        
        # Embed screenshot to Robot Framework log
        self.platform.embed_image_to_log(screenshot_base64)
        logger.debug("Screenshot captured and sent to AI for analysis")
        image_url = self.image_uploader.upload_from_base64(screenshot_base64)

        result = self.agent.ask_ai_visual_check(
            instruction=instruction,
            image_url=image_url,
            temperature=0,
        )

        logger.debug("Executing visual verification...")
        self._execute_visual_check_from_tool_calls(result)
        logger.debug("Agent.VisualCheck completed successfully")

    # ----------------------- Internals -----------------------
    def _run_rf_keyword(self, keyword_name: str, *args: Any) -> Any:
        from robot.libraries.BuiltIn import BuiltIn
        try:
            args_str = " ".join([str(a) for a in args]) if args else ""
            logger.debug(f"EXECUTING: {keyword_name} {args_str}".strip())
            logger.info(f"▶️ RF: {keyword_name} {args_str}")

            result = BuiltIn().run_keyword(keyword_name, *args)

            logger.info(f"SUCCESS: {keyword_name} executed successfully")
            return result
        except Exception as exc:
            raise

    def _extract_text_from_instruction(self, instruction: str) -> Optional[str]:
        import re

        patterns = [
            r'input this text[^:]*:\s*(.+)$',
            r'type this text[^:]*:\s*(.+)$',
            r'enter this text[^:]*:\s*(.+)$',
            r'write this text[^:]*:\s*(.+)$',
            r'input\s*:\s*(.+)$',
            r'type\s*:\s*(.+)$',
            r'enter\s*:\s*(.+)$',
            r'with text\s*["\']([^"\']+)["\']',
            r'["\']([^"\']+)["\']',
        ]

        instruction_lower = instruction.lower()
        for pattern in patterns:
            match = re.search(pattern, instruction_lower, re.IGNORECASE)
            if match:
                text = match.group(1).strip().strip('\"\'')
                if text:
                    return text
        return None

    def _execute_do_from_tool_calls(self, result: Dict[str, Any], ui_candidates: List[Dict[str, Any]], instruction: str) -> None:
        """Execute actions from tool calls returned by the LLM."""
        tool_calls = result.get("tool_calls", [])
        
        if not tool_calls:
            logger.error("No tool calls in response")
            raise AssertionError("AI did not return any tool calls")
        
        # Execute the first tool call (typically there's only one for DO actions)
        tool_call = tool_calls[0]
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        
        logger.info(f"Executing tool: {function_name} with args: {arguments}")
        
        if function_name == "tap_element":
            element_index = arguments.get("element_index")
            if element_index is None or element_index < 1 or element_index > len(ui_candidates):
                raise AssertionError(f"Invalid element_index: {element_index}. Must be 1-{len(ui_candidates)}")
            
            element = ui_candidates[element_index - 1]
            rf_locator = self.platform.build_locator_from_element(element)
            logger.info(f"Built locator: {rf_locator} from element: {element}")
            self._run_rf_keyword("Click Element", rf_locator)
            
        elif function_name == "input_text":
            element_index = arguments.get("element_index")
            text = arguments.get("text")
            
            if element_index is None or element_index < 1 or element_index > len(ui_candidates):
                raise AssertionError(f"Invalid element_index: {element_index}. Must be 1-{len(ui_candidates)}")
            
            if not text:
                raise AssertionError("'input_text' requires text argument")
            
            element = ui_candidates[element_index - 1]
            rf_locator = self.platform.build_locator_from_element(element)
            logger.info(f"Built locator: {rf_locator} from element: {element}")
            self._run_rf_keyword("Clear Text", rf_locator)
            self._run_rf_keyword("Input Text", rf_locator, text)
            
        elif function_name == "scroll_down":
            logger.info("Scrolling down...")
            self._run_rf_keyword("Swipe", "50", "80", "50", "20", "500")
            
        else:
            raise AssertionError(f"Unknown tool call: {function_name}")

    def _execute_visual_check_from_tool_calls(self, result: Dict[str, Any]) -> None:
        """Execute visual check from tool calls returned by the LLM."""
        tool_calls = result.get("tool_calls", [])
        
        if not tool_calls:
            logger.error("No tool calls in visual check response")
            raise AssertionError("AI did not return any tool calls for visual verification")
        
        # Extract the first tool call (should be verify_visual_match)
        tool_call = tool_calls[0]
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        
        if function_name != "verify_visual_match":
            raise AssertionError(f"Unexpected tool call for visual check: {function_name}")
        
        logger.info(f"Visual verification results: {arguments}")
        
        # Extract arguments
        verification_result = arguments.get("verification_result")
        confidence_score = arguments.get("confidence_score")
        analysis = arguments.get("analysis")
        found_elements = arguments.get("found_elements", [])
        issues = arguments.get("issues", [])

        # Log to Robot Framework with detailed AI response
        logger.debug("=" * 80)
        logger.debug("AI VISUAL VERIFICATION RESPONSE")
        logger.debug("=" * 80)
        logger.debug(f"Verification Result: {'PASS' if verification_result else 'FAIL'}")
        logger.debug(f"Confidence Score: {confidence_score:.2f}")
        logger.debug(f"Analysis: {analysis}")
        
        if found_elements:
            logger.debug(f"Found Elements ({len(found_elements)} total):")
            for i, element in enumerate(found_elements[:10], 1):  # Show first 10 elements
                element_type = element.get("element_type", "unknown")
                description = element.get("description", "no description")
                location = element.get("location", "unknown location")
                confidence = element.get("confidence", 0.0)
                logger.debug(f"  {i}. {element_type}: {description}")
                logger.debug(f"     Location: {location}")
                logger.debug(f"     Confidence: {confidence:.2f}")
        
        if issues:
            logger.debug(f"Issues Found ({len(issues)} total):")
            for i, issue in enumerate(issues, 1):
                logger.debug(f"  {i}. {issue}")
        
        logger.debug("=" * 80)

        # Also log to custom logger for consistency
        logger.debug(f"🔍 Verification result: {verification_result}")
        logger.debug(f"📊 Confidence score: {confidence_score}")
        logger.debug(f"📝 Analysis: {analysis}")
        
        if found_elements:
            logger.debug(f"🎯 Found elements: {len(found_elements)} elements detected")
            for i, element in enumerate(found_elements[:5], 1):  # Show first 5 elements
                element_type = element.get("element_type", "unknown")
                description = element.get("description", "no description")
                confidence = element.get("confidence", 0.0)
                logger.debug(f"  {i}. {element_type}: {description} (confidence: {confidence:.2f})")
        
        if issues:
            logger.debug(f"⚠️ Issues found: {len(issues)} issues detected")
            for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues
                logger.debug(f"  {i}. {issue}")

        # Assert based on verification result
        if verification_result:
            logger.info("✅ Visual verification passed")
        else:
            error_msg = f"Visual verification failed. Analysis: {analysis}"
            if issues:
                error_msg += f" Issues: {', '.join(issues[:3])}"
            raise AssertionError(error_msg)