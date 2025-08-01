"""
Function calling management for chat functionality
Integrates with Gemini native function calling and MCP tools
"""

import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class FunctionCallManager:
    """
    Manages function calling integration between Gemini and MCP tools
    """
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
    
    async def create_gemini_model_with_tools(
        self,
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
    
    async def chat_with_native_functions(
        self,
        model: genai.GenerativeModel,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conduct a chat conversation with native Gemini function calling
        
        Args:
            model: Gemini model instance
            messages: Conversation messages
            system_instruction: Optional system instruction
            
        Returns:
            Dict containing response and function call information
        """
        try:
            # Import the function calling module
            from gemini_function_calling import GeminiFunctionCallHandler
            
            # Create function call handler
            function_handler = GeminiFunctionCallHandler(self.mcp_client)
            
            # Use native function calling
            result = await function_handler.chat_with_functions(
                model=model,
                messages=messages,
                system_instruction=system_instruction
            )
            
            return result
            
        except ImportError:
            logger.warning("Native function calling not available")
            return await self._fallback_keyword_matching(messages, model)
        except Exception as e:
            logger.error(f"Error in native function calling: {e}")
            return {
                "response": f"Error in function calling: {str(e)}",
                "function_calls_made": [],
                "tools_available": False,
                "error": str(e)
            }
    
    async def _fallback_keyword_matching(
        self,
        messages: List[Dict[str, str]],
        model: genai.GenerativeModel
    ) -> Dict[str, Any]:
        """
        Fallback to legacy keyword-based tool detection
        
        Args:
            messages: Conversation messages
            model: Gemini model instance
            
        Returns:
            Dict containing response information
        """
        try:
            # Get available tools
            available_tools = await self.mcp_client.get_tools()
            
            # Get the latest message
            latest_message = messages[-1]["content"] if messages else ""
            
            # Simple tool detection
            needs_tools = any(keyword in latest_message.lower() for keyword in [
                "table", "record", "create", "update", "delete", "search", "metadata", "airtable"
            ])
            
            tools_used = []
            
            if needs_tools and available_tools:
                response_text = await self._handle_tool_request_legacy(
                    latest_message, available_tools
                )
            else:
                # Regular chat without tools
                response = model.generate_content(latest_message)
                response_text = response.text
            
            return {
                "response": response_text,
                "function_calls_made": tools_used,
                "tools_available": len(available_tools) > 0,
                "fallback_mode": "keyword_matching"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback keyword matching: {e}")
            return {
                "response": f"Error in chat processing: {str(e)}",
                "function_calls_made": [],
                "tools_available": False,
                "error": str(e)
            }
    
    async def _handle_tool_request_legacy(
        self,
        message: str,
        available_tools: List[Dict[str, Any]],
        base_id: Optional[str] = None
    ) -> str:
        """
        Handle requests that need tool calling using legacy keyword matching
        
        Args:
            message: User message
            available_tools: List of available MCP tools
            base_id: Optional Airtable base ID
            
        Returns:
            Response text
        """
        message_lower = message.lower()
        
        if not base_id:
            return "I need your Airtable base ID to help you. Please provide your base ID (it looks like 'appXXXXXXXXXXXXXX')."
        
        # Simple pattern matching for demo
        if "metadata table" in message_lower or "describe all tables" in message_lower:
            # Use create_metadata_table tool
            result = await self.mcp_client.call_tool("create_metadata_table", {"base_id": base_id})
            if "error" in result:
                return f"Error creating metadata table: {result['error']}"
            
            return f"I've analyzed your Airtable base and created metadata! Here's what I found:\n\n{result}"
        
        elif "list tables" in message_lower:
            # Use list_tables tool
            result = await self.mcp_client.call_tool("list_tables", {"base_id": base_id})
            if "error" in result:
                return f"Error listing tables: {result['error']}"
            
            return f"Here are the tables in your base:\n\n{result}"
        
        else:
            return (
                f"I understand you want to work with Airtable data, but I need more specific instructions. "
                f"I can help you:\n\n"
                f"- List tables in your base\n"
                f"- Create a metadata table describing all tables\n"
                f"- Get records from specific tables\n"
                f"- Create, update, or delete records\n\n"
                f"What would you like to do?"
            )