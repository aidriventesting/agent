from typing import Any, Dict, List, Optional, Tuple
from robot.api import logger
from PIL import Image

from Agent.ai.vlm.omniparser_client import OmniParserClient
from Agent.ai.vlm.omniparser_parser import OmniParserResultProcessor
from Agent.ai.vlm.omniparser_selector import OmniParserElementSelector


class OmniParserOrchestrator:
    """
    Main orchestrator for GUI element selection via OmniParser + LLM.
    
    This class coordinates:
    1. OmniParserClient - Analyzes the image via Hugging Face
    2. OmniParserResultProcessor - Parses and filters elements
    3. OmniParserElementSelector - Selects the element via LLM
    """

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        omniparser_space_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Initializes the orchestrator with all necessary components.
        
        Args:
            llm_provider: LLM provider (openai, anthropic, etc.)
            llm_model: Model to use
            omniparser_space_id: OmniParser Hugging Face space ID (optional)
            hf_token: Hugging Face token (optional)
        """
        self.client = OmniParserClient(
            space_id=omniparser_space_id,
            hf_token=hf_token
        )
        self.selector = OmniParserElementSelector(
            provider=llm_provider,
            model=llm_model
        )
        logger.debug("OmniParserOrchestrator initialized successfully")

    def find_element(
        self,
        element_description: str,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_name: Optional[str] = None,
        element_type: str = "interactive",
        box_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_paddleocr: Optional[bool] = None,
        imgsz: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Finds the GUI element that matches the description.
        
        Complete workflow:
        1. Sends the image to OmniParser to detect elements
        2. Parses and filters elements by type
        3. Uses the LLM to select the matching element
        
        Args:
            element_description: Description of the element being searched for (e.g., "YouTube app")
            image_path: Local path to the image
            image_url: Image URL
            image_base64: Base64 encoded image
            image_name: Image name (to infer extension if base64)
            element_type: Type of elements to filter:
                - "interactive" (default): interactive elements only
                - "icon": icons only
                - "text": texts only
                - "all": all elements
            box_threshold: OmniParser detection threshold
            iou_threshold: OmniParser IOU threshold
            use_paddleocr: Use PaddleOCR
            imgsz: Image size for OmniParser
            temperature: Temperature for the LLM
            
        Returns:
            Dictionary with:
            - element_key: Element key (e.g., 'icon3')
            - element_data: Complete element data (bbox, content, etc.)
            - confidence: LLM confidence level
            - reason: Reason for the choice
            - image_temp_path: Path to the annotated temporary image (optional)
            
            Returns None if no element is found.
        """
        logger.debug(f"🔍 Searching for element: '{element_description}'")
        
        # Step 1: Analyze image with OmniParser
        logger.debug("📸 Step 1/3: Analyzing image with OmniParser...")
        image_temp_path, parsed_text = self.client.parse_image(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_name=image_name,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        
        if not parsed_text:
            logger.error("❌ OmniParser detected no elements")
            return None
        
        # Step 2: Parse and filter elements by type
        logger.debug(f"🔧 Step 2/3: Parsing and filtering elements (type={element_type})...")
        processor = OmniParserResultProcessor(
            response_text=parsed_text,
            image_temp_path=image_temp_path,
        )
        elements_data = processor.get_parsed_ui_elements(element_type=element_type)
        
        if not elements_data:
            logger.error(f"❌ No elements of type '{element_type}' found")
            return None
         
        logger.debug(f"✓ {len(elements_data)} filtered elements")
        
        # Step 3: Select element via LLM
        logger.debug("🤖 Step 3/3: Selecting element via LLM...")
        result = self.selector.select_element(
            elements_data=elements_data,
            element_description=element_description,
            temperature=temperature,
        )
        
        if not result:
            logger.error("❌ The LLM found no matching element")
            return None
        
        # Add temporary image to result
        result["image_temp_path"] = image_temp_path
        
        logger.info(
            f"✅ Element found: {result['element_key']} "
            f"(confidence={result.get('confidence', 'unknown')})"
        )
        
        return result

    @staticmethod
    def bbox_to_pixels(
        bbox_normalized: List[float],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Converts normalized bbox coordinates to real pixel coordinates.
        
        Args:
            bbox_normalized: List [x1, y1, x2, y2] with values between 0 and 1
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple (x1, y1, x2, y2) in integer pixel coordinates
            
        Example:
            >>> bbox = [0.419, 0.170, 0.574, 0.266]
            >>> pixels = OmniParserOrchestrator.bbox_to_pixels(bbox, 1080, 1920)
            >>> print(pixels)  # (452, 326, 620, 510)
        """
        if len(bbox_normalized) != 4:
            raise ValueError(f"bbox must contain 4 values, received {len(bbox_normalized)}")
        
        x1_norm, y1_norm, x2_norm, y2_norm = bbox_normalized
        
        # Convert to pixels
        x1 = int(x1_norm * image_width)
        y1 = int(y1_norm * image_height)
        x2 = int(x2_norm * image_width)
        y2 = int(y2_norm * image_height)
        
        logger.debug(
            f"Bbox conversion: [{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}] "
            f"-> [{x1}, {y1}, {x2}, {y2}] (image: {image_width}x{image_height})"
        )
        
        return (x1, y1, x2, y2)

    @staticmethod
    def bbox_to_pixels_from_image(
        bbox_normalized: List[float],
        image_path: str,
    ) -> Tuple[int, int, int, int]:
        """
        Converts normalized bbox to pixels by automatically reading dimensions.
        
        Args:
            bbox_normalized: List [x1, y1, x2, y2] with values between 0 and 1
            image_path: Path to the image to get dimensions
            
        Returns:
            Tuple (x1, y1, x2, y2) in integer pixel coordinates
            
        Example:
            >>> bbox = [0.419, 0.170, 0.574, 0.266]
            >>> pixels = OmniParserOrchestrator.bbox_to_pixels_from_image(
            ...     bbox, "screenshot.png"
            ... )
            >>> print(pixels)  # (452, 326, 620, 510)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            logger.debug(f"Image dimensions '{image_path}': {width}x{height}")
        except Exception as e:
            logger.error(f"Unable to open image '{image_path}': {e}")
            raise
        
        return OmniParserOrchestrator.bbox_to_pixels(
            bbox_normalized=bbox_normalized,
            image_width=width,
            image_height=height,
        )


# Quick test
if __name__ == "__main__":
    from PIL import ImageDraw
    
    orchestrator = OmniParserOrchestrator()
    
    image_path = "tests/_data/images/screenshots/screenshot-Google Pixel 5-11.0.png"
    result = orchestrator.find_element(
        element_description="YouTube icon",
        image_path=image_path,
        element_type="interactive",
    )
    
    print("=" * 80)
    if result:
        print(f"Element found: {result['element_key']}")
        print(f"Content: {result['element_data']['content']}")
        print(f"Normalized bbox: {result['element_data']['bbox']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reason: {result['reason']}")
        
        # Convert to pixel coordinates
        print("\n" + "=" * 80)
        print("Converting to pixel coordinates:")
        
        bbox_normalized = result['element_data']['bbox']
        bbox_pixels = orchestrator.bbox_to_pixels_from_image(bbox_normalized, image_path)
        print(f"Bbox pixels: {bbox_pixels}")
        
        # Calculate center
        x1, y1, x2, y2 = bbox_pixels
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        print(f"Center: ({center_x}, {center_y})")
        
        # Draw a point on the image
        print("\n" + "=" * 80)
        print("Creating image with marker...")
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Draw a rectangle around the element
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw a point at center
        point_radius = 10
        draw.ellipse(
            [center_x - point_radius, center_y - point_radius,
             center_x + point_radius, center_y + point_radius],
            fill="red",
            outline="white",
            width=2
        )
        
        # Save the image
        output_path = "test_output_with_marker.png"
        img.save(output_path)
        print(f"✅ Image saved: {output_path}")
        print(f"   - Red rectangle around element")
        print(f"   - Red point at center ({center_x}, {center_y})")
        
    else:
        print("No element found")

