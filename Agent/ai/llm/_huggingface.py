import base64
import os
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import io
from gradio_client import Client, handle_file
from robot.libraries.BuiltIn import BuiltIn
from Agent.utilities._logger import RobotCustomLogger


class OmniParser:
    """Client for Microsoft's OmniParser v2 model on Hugging Face."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = Client("microsoft/OmniParser-v2", hf_token=api_key)
        self.logger = RobotCustomLogger()

    def parse_screenshot(
        self,
        image: Union[str, Image.Image, bytes],
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        use_paddleocr: bool = True,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        if isinstance(image, str):
            image_input = handle_file(image)
        elif isinstance(image, Image.Image):
            temp_path = "temp_image.png"
            image.save(temp_path)
            image_input = handle_file(temp_path)
            os.remove(temp_path)
        elif isinstance(image, bytes):
            temp_path = "temp_image.png"
            with open(temp_path, "wb") as f:
                f.write(image)
            image_input = handle_file(temp_path)
            os.remove(temp_path)
        else:
            raise ValueError("Image must be a file path, PIL Image, or bytes")

        try:
            _, result_text = self.client.predict(
                image_input=image_input,
                box_threshold=box_threshold,
                iou_threshold=iou_threshold,
                use_paddleocr=use_paddleocr,
                imgsz=imgsz,
                api_name="/process",
            )
        except Exception as e:
            print(f"Erreur lors de l'appel à OmniParser: {e}")
            return []

        parsed_elements = self._parse_response(result_text)
        return parsed_elements

    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        elements = []
        lines = response_text.strip().split("\n")
        for line in lines:
            if not line.startswith("icon "):
                continue
            try:
                icon_idx, content = line.split(":", 1)
                icon_idx = int(icon_idx.replace("icon ", ""))
                element_dict = eval(content.strip())
                element_dict["id"] = icon_idx
                elements.append(element_dict)
            except Exception as e:
                print(f"Error parsing line: {line}, Error: {e}")
                continue
        return elements

    def analyze_screenshot_with_omniparser(self, screenshot_base64=None, embed_to_log=True):
        built_in = BuiltIn()
        driver = built_in.get_library_instance("AppiumLibrary")._current_application()
        if screenshot_base64 is None:
            screenshot_base64 = self._capture_page_screenshot(embed_to_log=embed_to_log)
        screenshot_bytes = base64.b64decode(screenshot_base64)
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        parser = OmniParser(api_key=api_key)
        try:
            elements = parser.parse_screenshot(image=screenshot_bytes, box_threshold=0.05, iou_threshold=0.1)
            self.logger.info(f"Detected {len(elements)} UI elements on screen")
            return elements
        except Exception as e:
            self.logger.error(f"Error analyzing screen with OmniParser: {e}")
            return []


