"""
Gemini native function calling implementation
Converts MCP tools to Gemini function definitions and handles function calling
"""

import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool, ToolConfig

logger = logging.getLogger(__name__)


class MCPToGeminiFunctionConverter:
    """
    Converts MCP (Model Context Protocol) tools to Gemini function definitions
    """
    
    @staticmethod
    def convert_mcp_tool_to_gemini_function(mcp_tool: Dict[str, Any]) -> FunctionDeclaration:
        """
        Convert a single MCP tool to a Gemini function declaration
        
        Args:
            mcp_tool: MCP tool definition
            
        Returns:
            FunctionDeclaration for Gemini
        """
        try:
            # Extract MCP tool information
            name = mcp_tool.get("name", "unknown_tool")
            description = mcp_tool.get("description", "")
            input_schema = mcp_tool.get("inputSchema", {})
            
            # Convert MCP schema to Gemini format
            parameters = MCPToGeminiFunctionConverter._convert_schema_properties(input_schema)
            
            return FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters
            )
            
        except Exception as e:
            logger.error(f"Error converting MCP tool {mcp_tool.get('name', 'unknown')}: {e}")
            # Return a basic function declaration as fallback
            return FunctionDeclaration(
                name=mcp_tool.get("name", "fallback_tool"),
                description="Tool conversion failed",
                parameters={"type": "object", "properties": {}}
            )
    
    @staticmethod
    def _convert_schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MCP JSON Schema to Gemini function parameters format
        
        Args:
            schema: MCP JSON Schema
            
        Returns:
            Dict in Gemini function parameters format
        """
        if not schema or schema.get("type") != "object":
            return {"type": "object", "properties": {}}
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Convert each property
        gemini_properties = {}
        for prop_name, prop_def in properties.items():
            gemini_properties[prop_name] = MCPToGeminiFunctionConverter._convert_property(prop_def)
        
        result = {
            "type": "object",
            "properties": gemini_properties
        }
        
        if required:
            result["required"] = required
        
        return result
    
    @staticmethod
    def _convert_property(prop_def: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single property definition from MCP to Gemini format
        
        Args:
            prop_def: Property definition from MCP schema
            
        Returns:
            Dict in Gemini format
        """
        prop_type = prop_def.get("type", "string")
        description = prop_def.get("description", "")
        
        gemini_prop = {
            "type": prop_type,
            "description": description
        }
        
        # Handle additional constraints
        if "default" in prop_def:
            gemini_prop["default"] = prop_def["default"]
        
        if "enum" in prop_def:
            gemini_prop["enum"] = prop_def["enum"]
        
        if prop_type == "array" and "items" in prop_def:
            gemini_prop["items"] = MCPToGeminiFunctionConverter._convert_property(prop_def["items"])
        
        if prop_type == "integer":
            if "minimum" in prop_def:
                gemini_prop["minimum"] = prop_def["minimum"]
            if "maximum" in prop_def:
                gemini_prop["maximum"] = prop_def["maximum"]
        
        if prop_type == "string":
            if "minLength" in prop_def:
                gemini_prop["minLength"] = prop_def["minLength"]
            if "maxLength" in prop_def:
                gemini_prop["maxLength"] = prop_def["maxLength"]
            if "pattern" in prop_def:
                gemini_prop["pattern"] = prop_def["pattern"]
        
        return gemini_prop
    
    @staticmethod
    def convert_mcp_tools_to_gemini_tools(mcp_tools: List[Dict[str, Any]]) -> List[Tool]:
        """
        Convert a list of MCP tools to Gemini Tools
        
        Args:
            mcp_tools: List of MCP tool definitions
            
        Returns:
            List of Gemini Tool objects
        """
        function_declarations = []
        
        for mcp_tool in mcp_tools:
            try:
                func_decl = MCPToGeminiFunctionConverter.convert_mcp_tool_to_gemini_function(mcp_tool)
                function_declarations.append(func_decl)
                logger.debug(f"Converted MCP tool '{mcp_tool.get('name')}' to Gemini function")
            except Exception as e:
                logger.error(f"Failed to convert MCP tool {mcp_tool.get('name', 'unknown')}: {e}")
        
        if not function_declarations:
            logger.warning("No MCP tools were successfully converted to Gemini functions")
            return []
        
        # Create Gemini Tool with all function declarations
        tool = Tool(function_declarations=function_declarations)
        logger.info(f"Created Gemini tool with {len(function_declarations)} functions")
        
        return [tool]


class GeminiFunctionCallHandler:
    """
    Handles function calling with Gemini models using MCP tools
    """
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.converter = MCPToGeminiFunctionConverter()
    
    async def get_gemini_tools(self) -> List[Tool]:
        """
        Get Gemini tools from MCP server
        
        Returns:
            List of Gemini Tool objects
        """
        try:
            # Get MCP tools
            mcp_tools = await self.mcp_client.get_tools()
            
            if not mcp_tools:
                logger.warning("No MCP tools available")
                return []
            
            # Convert to Gemini format
            gemini_tools = self.converter.convert_mcp_tools_to_gemini_tools(mcp_tools)
            
            logger.info(f"Converted {len(mcp_tools)} MCP tools to {len(gemini_tools)} Gemini tools")
            return gemini_tools
            
        except Exception as e:
            logger.error(f"Error getting Gemini tools: {e}")
            return []
    
    async def handle_function_calls(self, function_calls: List[genai.types.FunctionCall]) -> List[genai.types.FunctionResponse]:
        """
        Execute function calls using MCP client and return responses
        
        Args:
            function_calls: List of function calls from Gemini
            
        Returns:
            List of function responses
        """
        responses = []
        
        for func_call in function_calls:
            try:
                logger.info(f"Executing function call: {func_call.name}")
                
                # Execute via MCP client
                result = await self.mcp_client.call_tool(func_call.name, dict(func_call.args))
                
                # Format response for Gemini
                if isinstance(result, dict) and "error" in result:
                    # Handle error case
                    response = genai.types.FunctionResponse(
                        name=func_call.name,
                        response={"error": result["error"], "success": False}
                    )
                else:
                    # Handle success case
                    response = genai.types.FunctionResponse(
                        name=func_call.name,
                        response={"result": result, "success": True}
                    )
                
                responses.append(response)
                logger.info(f"Function call {func_call.name} completed successfully")
                
            except Exception as e:
                logger.error(f"Error executing function {func_call.name}: {e}")
                # Return error response
                error_response = genai.types.FunctionResponse(
                    name=func_call.name,
                    response={"error": str(e), "success": False}
                )
                responses.append(error_response)
        
        return responses
    
    async def chat_with_functions(
        self,
        model: genai.GenerativeModel,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation with function calling enabled
        
        Args:
            model: Gemini model instance
            messages: Conversation messages
            system_instruction: Optional system instruction
            
        Returns:
            Dict containing response and function call information
        """
        try:
            # Get available tools
            tools = await self.get_gemini_tools()
            
            if not tools:
                logger.warning("No tools available for function calling")
                # Fall back to regular chat without tools
                response = model.generate_content(messages[-1]["content"] if messages else "")
                return {
                    "response": response.text,
                    "function_calls_made": [],
                    "tools_available": False
                }
            
            # Configure model with tools
            if system_instruction:
                model._system_instruction = system_instruction
            
            # Start chat with tools
            chat = model.start_chat(
                tools=tools,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                    )
                )
            )
            
            # Send the latest message
            latest_message = messages[-1]["content"] if messages else ""
            response = chat.send_message(latest_message)
            
            function_calls_made = []
            
            # Handle function calls if any
            while response.candidates[0].content.parts:
                function_calls = []
                
                # Extract function calls from response
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
                
                if not function_calls:
                    break
                
                # Execute function calls
                function_responses = await self.handle_function_calls(function_calls)
                function_calls_made.extend([fc.name for fc in function_calls])
                
                # Send function responses back to model
                response = chat.send_message(function_responses)
            
            return {
                "response": response.text,
                "function_calls_made": function_calls_made,
                "tools_available": True,
                "chat_history": [msg.to_dict() for msg in chat.history] if hasattr(chat, 'history') else []
            }
            
        except Exception as e:
            logger.error(f"Error in chat with functions: {e}")
            return {
                "response": f"Error in function calling chat: {str(e)}",
                "function_calls_made": [],
                "tools_available": False,
                "error": str(e)
            }


def create_gemini_model_with_tools(
    model_name: str = "gemini-2.5-flash",
    generation_config: Optional[genai.types.GenerationConfig] = None,
    system_instruction: Optional[str] = None
) -> genai.GenerativeModel:
    """
    Create a Gemini model configured for function calling
    
    Args:
        model_name: Name of the Gemini model to use
        generation_config: Optional generation configuration
        system_instruction: Optional system instruction
        
    Returns:
        Configured GenerativeModel instance
    """
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction
    )