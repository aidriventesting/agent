from typing import Any, Dict, List, Optional

import xml.etree.ElementTree as ET
from robot.api import logger
from robot.libraries.BuiltIn import BuiltIn

from Agent.utilities._logger import RobotCustomLogger


class DeviceConnector:
    """Single-file Appium facade: UI XML, candidates, locators, screenshots."""

    def __init__(self) -> None:
        self.logger = RobotCustomLogger()

    # ---- Driver helpers ----
    def _get_driver(self) -> Any:
        appium_lib = BuiltIn().get_library_instance('AppiumLibrary')
        return appium_lib._current_application()

    # ---- UI & parsing ----
    def get_ui_xml(self) -> str:
        return self._get_driver().page_source

    def parse_ui(self, ui_xml: str, max_items: int = 20) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(ui_xml)

            def walk(node: Any, depth: int = 0, max_depth: int = 12) -> None:
                if depth > max_depth:
                    return

                attrs: Dict[str, Any] = {
                    'text': node.get('text', ''),
                    'resource_id': node.get('resource-id', ''),
                    'class_name': node.get('class', ''),
                    'content_desc': node.get('content-desc', ''),
                    'package': node.get('package', ''),
                    'clickable': node.get('clickable', 'false').lower() == 'true',
                    'enabled': node.get('enabled', 'false').lower() == 'true',
                    'bounds': node.get('bounds', ''),
                    'index': node.get('index', ''),
                }

                if attrs['clickable'] and attrs['enabled']:
                    candidates.append({**attrs})

                for child in list(node):
                    walk(child, depth + 1, max_depth)

            walk(root)

            def sort_key(item: Dict[str, Any]) -> int:
                score = 0
                if item.get('text'): score += 3
                if item.get('content_desc'): score += 2
                if item.get('resource_id'): score += 1
                return score

            candidates.sort(key=sort_key, reverse=True)
            return candidates[:max_items]
        except ET.ParseError as e:
            self.logger.warning(f"⚠️ Erreur parsing XML: {e}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du parsing UI: {e}")
            return []

    def to_rf_locator(self, locator: Dict[str, Any]) -> str:
        strategy = locator.get("strategy")
        value = locator.get("value")

        if not strategy or not value:
            raise AssertionError("Locator doit inclure 'strategy' et 'value'")

        if strategy == "id":
            return f"id={value}"
        if strategy == "accessibility_id":
            return f"accessibility_id={value}"
        if strategy == "xpath":
            return value
        if strategy == "class_name":
            return f"class={value}"
        if strategy == "android_uiautomator":
            return f"android=uiautomator={value}"
        if strategy == "ios_predicate":
            return f"-ios predicate string:{value}"

        self.logger.warning(f"Unknown strategy '{strategy}', returning raw value")
        return value

    def get_locator_strategies(self) -> List[str]:
        return [
            "id",
            "accessibility_id",
            "xpath",
            "class_name",
            "android_uiautomator",
            "ios_predicate",
        ]

    def collect_ui_candidates(self, max_items: int = 20) -> List[Dict[str, Any]]:
        self.logger.info("🔍 Extracting UI context...")
        xml = self.get_ui_xml()
        xml_length = len(xml)
        xml_preview = xml[:1000] + "..." if xml_length > 1000 else xml
        self.logger.info(f"📱 UI XML retrieved ({xml_length} characters)")
        self.logger.debug(f"📋 XML preview (truncated): {xml_preview}")
        candidates = self.parse_ui(xml, max_items=max_items)
        self.logger.info(f"🎯 Number of UI candidates extracted: {len(candidates)}")
        for i, candidate in enumerate(candidates[:5]):
            self.logger.debug(f"  Candidate {i+1}: {candidate}")
        return candidates

    # ---- Screenshot helpers ----
    def get_screenshot_base64(self) -> str:
        return self._get_driver().get_screenshot_as_base64()

    def embed_image_to_log(self, base64_screenshot: str, width: int = 400, message: Optional[str] = None) -> None:
        msg = f"{message if message else ''}</td></tr><tr><td colspan=\"3\"><img src=\"data:image/png;base64, {base64_screenshot}\" width=\"{width}\"></td></tr>"
        logger.info(msg, html=True, also_console=False)

    