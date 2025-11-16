from typing import Any, Dict, Optional
from robot.api import logger

from Agent.ai.llm.facade import UnifiedLLMFacade


class OmniParserElementSelector:
    """
    Selects a GUI element using ChatGPT.
    
    Takes a dictionary of elements and a description,
    then asks the AI to find the matching element.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini") -> None:
        """
        Initializes the selector with the AI model.
        
        Args:
            provider: The AI provider (openai, anthropic, etc.)
            model: The model to use
        """
        self.llm = UnifiedLLMFacade(provider=provider, model=model)
        logger.info(f"OmniParserElementSelector initialized with {provider}/{model}")

    def select_element(
        self,
        elements_data: Dict[str, Dict[str, Any]],
        element_description: str,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the GUI element that matches the description.
        
        Args:
            elements_data: Dictionary of elements (e.g., {'icon3': {'type': 'icon', ...}})
            element_description: Description of the element to find (e.g., "YouTube app")
            temperature: Temperature for generation (0.0 = deterministic)
            
        Returns:
            A dictionary with:
            - element_key: The key of the found element (e.g., 'icon3')
            - element_data: The element data
            - confidence: Confidence level (optional)
            - reason: Reason for the choice (optional)
            
            Returns None if no element is found.
        """
        logger.info(f"Searching for element: '{element_description}'")
        logger.debug(f"Number of elements to analyze: {len(elements_data)}")

        # Build the prompt
        messages = self._build_prompt(elements_data, element_description)
        
        # Send to AI
        try:
            response = self.llm.send_ai_request_and_return_response(
                messages=messages,
                temperature=temperature
            )
            
            # Parse the response
            result = self._parse_response(response, elements_data)
            
            if result:
                logger.info(f"✅ Element found: {result.get('element_key')}")
            else:
                logger.warn("❌ No matching element found")
                
            return result
            
        except Exception as e:
            logger.error(f"Error during selection: {str(e)}")
            return None

    def _build_prompt(
        self,
        elements_data: Dict[str, Dict[str, Any]],
        element_description: str,
    ) -> list:
        """
        Builds the prompt for the AI.
        
        Args:
            elements_data: The available UI elements
            element_description: The description of the element being searched for
            
        Returns:
            List of messages for the AI
        """
        # Format elements in a readable way
        elements_text = self._format_elements(elements_data)
        
        system_prompt = """You are an assistant specialized in GUI element selection.
Your task is to find the element that best matches the given description.

Analyze the available elements and return the one that matches best.
If no element matches, indicate 'element_key': null.

Respond ONLY in JSON with this structure:
{
    "element_key": "the element key (e.g., icon3) or null",
    "confidence": "high, medium or low",
    "reason": "brief explanation of your choice"
}"""

        user_prompt = f"""Available elements:
{elements_text}

Description of the element being searched for: "{element_description}"

Find the element that best matches this description."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_elements(self, elements_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Formats elements for the prompt.
        
        Args:
            elements_data: The elements to format
            
        Returns:
            Formatted string of elements
        """
        lines = []
        for key, data in elements_data.items():
            content = data.get("content", "")
            element_type = data.get("type", "unknown")
            interactive = data.get("interactivity", False)
            
            lines.append(
                f"- {key}: type={element_type}, content='{content}', "
                f"interactive={interactive}"
            )
        
        return "\n".join(lines)

    def _parse_response(
        self,
        response: Dict[str, Any],
        elements_data: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Parses the AI response.
        
        Args:
            response: The JSON response from the AI
            elements_data: The original elements
            
        Returns:
            Dictionary with the found element or None
        """
        element_key = response.get("element_key")
        
        # If no element found
        if not element_key or element_key == "null":
            return None
        
        # Check that the element exists
        if element_key not in elements_data:
            logger.warn(f"The returned element '{element_key}' does not exist in the data")
            return None
        
        # Build the result
        return {
            "element_key": element_key,
            "element_data": elements_data[element_key],
            "confidence": response.get("confidence", "unknown"),
            "reason": response.get("reason", ""),
        }


# Quick test
if __name__ == "__main__":
    # Example data
    test_data = {
        'icon3': {
            'type': 'icon',
            'bbox': [0.41938668489456177, 0.17028668522834778, 0.5745916366577148, 0.2660691440105438],
            'interactivity': True,
            'content': 'YouTube '
        },
        'icon9': {
            'type': 'icon',
            'bbox': [0.23282678425312042, 0.17132169008255005, 0.38811373710632324, 0.26554766297340393],
            'interactivity': True,
            'content': 'Gmail '
        },
        'icon22': {
            'type': 'icon',
            'bbox': [0.05158957466483116, 0.639639139175415, 0.2134605348110199, 0.7337194681167603],
            'interactivity': True,
            'content': 'Chrome '
        }
    }
    
    selector = OmniParserElementSelector()
    result = selector.select_element(test_data, "YouTube")
    
    print("=" * 80)
    print(f"Result: {result}")

