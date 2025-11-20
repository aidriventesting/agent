from typing import List, Dict, Any, Union

def convert_to_anthropic_tools(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style tool definitions to Anthropic format.
    """
    anthropic_tools = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            # Pass through non-function tools or already formatted tools
            anthropic_tools.append(tool)
            continue
            
        fn = tool.get("function", {})
        anthropic_tool = {
            "name": fn.get("name"),
            "description": fn.get("description"),
            "input_schema": fn.get("parameters", {})
        }
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools

def convert_to_gemini_tools(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style tool definitions to Gemini format.
    Gemini expects a list of tool objects, where each tool object contains 'function_declarations'.
    """
    function_declarations = []
    
    for tool in openai_tools:
        if tool.get("type") != "function":
            continue
            
        fn = tool.get("function", {})
        
        func_decl = {
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters")
        }
        function_declarations.append(func_decl)

    if not function_declarations:
        return openai_tools

    return [{"function_declarations": function_declarations}]

def convert_tool_choice_to_anthropic(tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert OpenAI-style tool_choice to Anthropic format.
    """
    if not tool_choice:
        return {"type": "auto"}

    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            # Anthropic doesn't have "none", so we return auto and let the model decide
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
        else:
            # Assume it's a function name passed as string
            return {"type": "tool", "name": tool_choice}
            
    if isinstance(tool_choice, dict):
        # Already in Anthropic format
        if "type" in tool_choice:
            return tool_choice
        # OpenAI format: {"type": "function", "function": {"name": "..."}}
        if tool_choice.get("type") == "function":
            fn_name = tool_choice.get("function", {}).get("name")
            if fn_name:
                return {"type": "tool", "name": fn_name}
                
    # Fallback
    return {"type": "auto"}

def convert_tool_choice_to_gemini_config(tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert OpenAI-style tool_choice to Gemini tool_config.
    """
    if not tool_choice:
        return {}

    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"function_calling_config": {"mode": "AUTO"}}
        elif tool_choice == "none":
            return {"function_calling_config": {"mode": "NONE"}}
        elif tool_choice == "required":
            return {"function_calling_config": {"mode": "ANY"}}
        else:
            # Assume it's a function name passed as string (non-standard but possible)
            return {"function_calling_config": {"mode": "ANY", "allowed_function_names": [tool_choice]}}
            
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn_name = tool_choice.get("function", {}).get("name")
            if fn_name:
                return {"function_calling_config": {"mode": "ANY", "allowed_function_names": [fn_name]}}
                
    # Fallback
    return {}
