import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from typing import Optional, Dict, List, Union
from robot.api import logger
from Agent.ai.llm._baseclient import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
    ):
        self.api_key: str = api_key

        if not self.api_key:
            from Agent.config.config import Config
            config = Config()
            self.api_key = config.GEMINI_API_KEY

        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or in the environment variables.")

        self.default_model = model
        self.max_retries = max_retries

        genai.configure(api_key=self.api_key)
        model_name = model.replace("models/", "") if model.startswith("models/") else model
        self.client = genai.GenerativeModel(model_name=model_name)

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
    ) -> Optional[GenerateContentResponse]:
        try:
            self._validate_parameters(temperature, top_p)

            if model and model != self.default_model:
                client = genai.GenerativeModel(model_name=model)
            else:
                client = self.client

            gemini_messages = self._convert_messages_to_gemini_format(messages)

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )

            request_params = {
                "contents": gemini_messages,
                "generation_config": generation_config,
            }

            if tools:
                # Check format and convert if necessary
                if len(tools) > 0 and isinstance(tools[0], dict) and "type" in tools[0] and tools[0]["type"] == "function":
                    from Agent.ai.llm._converters import convert_to_gemini_tools
                    request_params["tools"] = convert_to_gemini_tools(tools)
                else:
                    request_params["tools"] = tools
            
            if tool_choice:
                from Agent.ai.llm._converters import convert_tool_choice_to_gemini_config
                request_params["tool_config"] = convert_tool_choice_to_gemini_config(tool_choice)

            response = client.generate_content(**request_params)

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens = response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
                logger.debug(f"Gemini API call successful. Tokens used: {total_tokens}", True)
            else:
                logger.debug(f"Gemini API call successful (no usage metadata available)", True)

            logger.debug(f"Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}", True)
            raise

    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        gemini_messages = []
        system_message = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
            elif role == "assistant":
                if isinstance(content, str):
                    gemini_messages.append({"role": "model", "parts": [content]})
                else:
                    gemini_messages.append({"role": "model", "parts": [str(content)]})
            elif role == "user":
                parts = []
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts = self._process_content_parts(content)
                else:
                    parts.append(str(content))

                if system_message and not any(m.get("role") == "user" for m in gemini_messages):
                    parts.insert(0, system_message)
                    system_message = None

                gemini_messages.append({"role": "user", "parts": parts})

        return gemini_messages

    def _process_content_parts(self, content_list: List[Dict]) -> List:
        parts = []
        for item in content_list:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
            elif item_type == "image_url":
                image_data = self._process_image_url(item.get("image_url", {}))
                if image_data:
                    parts.append(image_data)
            elif item_type == "image":
                if "inline_data" in item:
                    parts.append(item)
        return parts

    def _process_image_url(self, image_url_data: Dict) -> Optional[Dict]:
        if not isinstance(image_url_data, dict):
            return None
        url = image_url_data.get("url", "")
        if not url:
            return None
        try:
            if url.startswith("data:"):
                header, data = url.split(",", 1)
                media_type = header.split(";")[0].split(":")[1]
                return {"inline_data": {"mime_type": media_type, "data": data}}
        except Exception as e:
            logger.error(f"Error processing image URL: {e}", True)
            return None
        return None

    def _validate_parameters(self, temperature: float, top_p: float):
        if not (0 <= temperature <= 2):
            logger.error(f"Invalid temperature {temperature}. Must be between 0 and 2")
            raise ValueError(f"Invalid temperature {temperature}. Must be between 0 and 2")
        if not (0 <= top_p <= 1):
            logger.error(f"Invalid top_p {top_p}. Must be between 0 and 1")
            raise ValueError(f"Invalid top_p {top_p}. Must be between 0 and 1")

    def format_response(
        self,
        response: GenerateContentResponse,
        include_tokens: bool = True,
        include_reason: bool = False,
    ) -> Dict[str, Union[str, int]]:
        if not response or not response.candidates:
            logger.error(f"Invalid response or no candidates in the response", True)
            return {}

        finish_reason = response.candidates[0].finish_reason
        finish_reason_name = str(finish_reason)
        
        content_text = ""
        tool_calls = []
        
        try:
            content_text = response.text
        except Exception as e:
            logger.warn(f"Cannot extract text from response: {e}", True)
            if finish_reason == 2:
                content_text = "[Content blocked by safety filters]"
                logger.warn("Response blocked by Gemini safety filters", True)
            elif finish_reason == 3:
                content_text = "[Content blocked by recitation filter]"
                logger.warn("Response blocked by recitation filter", True)
            else:
                content_text = f"[No content available - finish_reason: {finish_reason_name}]"

        # Extract function calls
        if response.candidates and response.candidates[0].content.parts:
            import json
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    # Convert arguments to dict (they are Struct/Map)
                    args = {}
                    for key, value in fc.args.items():
                        args[key] = value
                    
                    tool_calls.append({
                        "id": "call_" + fc.name,  # Gemini doesn't provide ID
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(args)
                        }
                    })

        result = {"content": content_text}
        
        if tool_calls:
            result["tool_calls"] = tool_calls
            
        if include_tokens and hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            total_tokens = prompt_tokens + completion_tokens
            logger.debug(f"Tokens used: input={prompt_tokens}, output={completion_tokens}")
            result.update(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
        if include_reason and response.candidates:
            logger.debug(f"Finish reason: {finish_reason_name}")
            result["finish_reason"] = finish_reason_name
        return result


