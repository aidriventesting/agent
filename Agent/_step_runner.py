from typing import Any, Dict, List, Optional

from Agent.utilities._logger import RobotCustomLogger
from Agent.platforms import DeviceConnector
from Agent.agent._aiconnector import AiConnector
from Agent.utilities.imguploader.imghandler import ImageUploader

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
        self.logger = RobotCustomLogger()
        # Platform
        self.platform: DeviceConnector = platform or DeviceConnector()
        # Agent component
        self.agent = AiConnector(provider=llm_client, model=llm_model)
        self.image_uploader = ImageUploader(service="auto")
        
    # ----------------------- Public API -----------------------
    def do(self, instruction: str) -> None:
        self.logger.info(f"🚀 Starting Agent.Do with instruction: '{instruction}'")

        ui_candidates = self.platform.collect_ui_candidates()
        # TODO: the final logic will be coded when we have the visual model
        # currently we only work on locators
        result = self.agent.ask_ai_do(
            instruction=instruction,
            ui_elements=ui_candidates,
            temperature=0,
        )

        self.logger.info("⚡ Executing action...")
        self._execute_do(result, instruction)
        self.logger.success("✅ Agent.Do completed successfully")

    def visual_check(self, instruction: str) -> None:
        self.logger.info(f"👁️ Starting Agent.VisualCheck with instruction: '{instruction}'")

        # Log to Robot Framework
        from robot.api import logger as rf_logger
        rf_logger.info("=" * 80)
        rf_logger.info("AGENT VISUAL CHECK STARTED")
        rf_logger.info("=" * 80)
        rf_logger.info(f"Instruction: {instruction}")
        rf_logger.info("Capturing screenshot for AI analysis...")

        # Capture screenshot
        self.logger.info("📸 Capturing screenshot...")
        screenshot_base64 = self.platform.get_screenshot_base64()
        
        # Embed screenshot to Robot Framework log
        self.platform.embed_image_to_log(screenshot_base64, message="Visual Check Screenshot")
        rf_logger.info("Screenshot captured and sent to AI for analysis")
        image_url = self.image_uploader.upload_from_base64(screenshot_base64)

        result = self.agent.ask_ai_visual_check(
            instruction=instruction,
            image_base_or_url=image_url,
            temperature=0,
        )

        self.logger.info("⚡ Executing visual verification...")
        self._execute_visual_check(result)
        self.logger.success("✅ Agent.VisualCheck completed successfully")

    # ----------------------- Internals -----------------------
    def _run_rf_keyword(self, keyword_name: str, *args: Any) -> Any:
        from robot.api import logger as rf_logger
        from robot.libraries.BuiltIn import BuiltIn
        try:
            args_str = " ".join([str(a) for a in args]) if args else ""
            rf_logger.info(f"EXECUTING: {keyword_name} {args_str}".strip())
            self.logger.info(f"▶️ RF: {keyword_name} {args_str}")

            result = BuiltIn().run_keyword(keyword_name, *args)

            rf_logger.info(f"SUCCESS: {keyword_name} executed successfully")
            self.logger.success(f"✅ RF success: {keyword_name}")
            return result
        except Exception as exc:
            rf_logger.error(f"FAILED: {keyword_name} - {exc}")
            self.logger.error(f"❌ RF failed: {keyword_name} - {exc}")
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

    def _execute_do(self, result: Dict[str, Any], instruction: str) -> None:
        action = result.get("action")
        locator = result.get("locator", {})
        text = result.get("text")
        candidates = result.get("candidates", []) or []

        self.logger.info(f"🎬 Requested action: {action}")
        self.logger.info(f"📍 Provided locator: {locator}")
        self.logger.info(f"📝 Text to input: {text}")
        self.logger.info(f"🎯 Alternative candidates: {candidates}")

        if action == "open":
            self.logger.info("🚪 Executing: Open Application")
            self._run_rf_keyword("Open Application")
            self.logger.success("✅ Application opened successfully")
            return

        if not locator:
            raise AssertionError("No locator available for the action")

        rf_locator = self.platform.to_rf_locator(locator)
        self.logger.info(f"🎯 Converted Robot Framework locator: {rf_locator}")

        if action == "tap":
            self.logger.info(f"👆 Executing: Click Element with locator '{rf_locator}'")
            self._run_rf_keyword("Click Element", rf_locator)
            self.logger.success("✅ Element clicked successfully")
            return

        if action == "type":
            if text is None:
                text = self._extract_text_from_instruction(instruction)
                if text is None:
                    raise AssertionError("Agent.Do 'type' requires 'text'")
                self.logger.info(f"📝 Text automatically extracted from instruction: '{text}'")
            self.logger.info(f"⌨️ Executing: Input Text '{text}' into locator '{rf_locator}'")
            self._run_rf_keyword("Input Text", rf_locator, text)
            self.logger.success("✅ Text entered successfully")
            return

        if action == "clear":
            self.logger.info(f"🧹 Executing: Clear Text for locator '{rf_locator}'")
            self._run_rf_keyword("Clear Text", rf_locator)
            self.logger.success("✅ Text cleared successfully")
            return

        if action == "swipe":
            self.logger.error("🚫 Action 'swipe' not yet implemented")
            raise AssertionError("Swipe not yet implemented in Agent.Do")

        self.logger.error(f"🚫 Unsupported action: {action}")
        raise AssertionError(f"Unsupported action: {action}")

    def _execute_visual_check(self, result: Dict[str, Any]) -> None:
        verification_result = result.get("verification_result")
        confidence_score = result.get("confidence_score")
        analysis = result.get("analysis")
        found_elements = result.get("found_elements", [])
        issues = result.get("issues", [])

        # Log to Robot Framework with detailed AI response
        from robot.api import logger as rf_logger
        
        rf_logger.info("=" * 80)
        rf_logger.info("AI VISUAL VERIFICATION RESPONSE")
        rf_logger.info("=" * 80)
        rf_logger.info(f"Verification Result: {'PASS' if verification_result else 'FAIL'}")
        rf_logger.info(f"Confidence Score: {confidence_score:.2f}")
        rf_logger.info(f"Analysis: {analysis}")
        
        if found_elements:
            rf_logger.info(f"Found Elements ({len(found_elements)} total):")
            for i, element in enumerate(found_elements[:10], 1):  # Show first 10 elements
                element_type = element.get("element_type", "unknown")
                description = element.get("description", "no description")
                location = element.get("location", "unknown location")
                confidence = element.get("confidence", 0.0)
                rf_logger.info(f"  {i}. {element_type}: {description}")
                rf_logger.info(f"     Location: {location}")
                rf_logger.info(f"     Confidence: {confidence:.2f}")
        
        if issues:
            rf_logger.info(f"Issues Found ({len(issues)} total):")
            for i, issue in enumerate(issues, 1):
                rf_logger.info(f"  {i}. {issue}")
        
        rf_logger.info("=" * 80)

        # Also log to custom logger for consistency
        self.logger.info(f"🔍 Verification result: {verification_result}")
        self.logger.info(f"📊 Confidence score: {confidence_score}")
        self.logger.info(f"📝 Analysis: {analysis}")
        
        if found_elements:
            self.logger.info(f"🎯 Found elements: {len(found_elements)} elements detected")
            for i, element in enumerate(found_elements[:5], 1):  # Show first 5 elements
                element_type = element.get("element_type", "unknown")
                description = element.get("description", "no description")
                confidence = element.get("confidence", 0.0)
                self.logger.info(f"  {i}. {element_type}: {description} (confidence: {confidence:.2f})")
        
        if issues:
            self.logger.info(f"⚠️ Issues found: {len(issues)} issues detected")
            for i, issue in enumerate(issues[:3], 1):  # Show first 3 issues
                self.logger.info(f"  {i}. {issue}")

        # Assert based on verification result
        if verification_result:
            self.logger.success("✅ Visual verification passed")
            rf_logger.info("✅ VISUAL VERIFICATION PASSED")
        else:
            self.logger.error("❌ Visual verification failed")
            rf_logger.error("❌ VISUAL VERIFICATION FAILED")
            error_msg = f"Visual verification failed. Analysis: {analysis}"
            if issues:
                error_msg += f" Issues: {', '.join(issues[:3])}"
            raise AssertionError(error_msg)
