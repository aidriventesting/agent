from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union


class BaseLLMClient(ABC):
    @abstractmethod
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs
    ):
        pass

    @abstractmethod
    def format_response(self, response, include_tokens: bool = True, include_reason: bool = False):
        pass


