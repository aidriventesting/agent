import json
import re
from typing import Any, Dict


def extract_json_safely(response: str) -> Dict[str, Any]:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                raise ValueError("Extracted content is not valid JSON.")
        else:
            raise ValueError("No JSON content found in the response.")


