"""
Chat endpoint handler for LLM Orchestrator
Handles chat requests with function calling and cost tracking
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel

import google.generativeai as genai
from fastapi import HTTPException

from .function_calling import FunctionCallManager

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    base_id: Optional[str] = None
    thinking_budget: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    thinking_process: Optional[str] = None
    tools_used: List[str] = []
    session_id: str
    timestamp: str
    cost_info: Optional[Dict[str, Any]] = None


class ChatHandler:
    """
    Handles chat endpoint functionality with cost tracking and session management
    """
    
    def __init__(
        self,
        session_manager,
        mcp_client,
        cost_tracker=None,
        default_model: str = "gemini-2.5-flash",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        thinking_budget: int = 5
    ):
        self.session_manager = session_manager
        self.mcp_client = mcp_client
        self.cost_tracker = cost_tracker
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        
        # Initialize function call manager
        self.function_manager = FunctionCallManager(mcp_client)
    
    async def handle_chat_request(self, request: ChatRequest) -> ChatResponse:
        """
        Handle a chat request with function calling and cost tracking
        
        Args:
            request: Chat request data
            
        Returns:
            Chat response with function call results and cost information
        """
        # Clean up expired sessions
        await self.session_manager.cleanup_expired_sessions()
        session = await self.session_manager.get_session(request.session_id)
        
        # Pre-request budget checking
        if self.cost_tracker:
            await self._check_budget_before_request(request, session)
        
        try:
            # Configure generation with thinking budget
            thinking_budget = request.thinking_budget or self.thinking_budget
            generation_config = genai.types.GenerationConfig(
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
                max_output_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Prepare system instruction
            system_instruction = self._build_system_instruction(request)
            
            # Create Gemini model with function calling
            model = await self.function_manager.create_gemini_model_with_tools(
                model_name=self.default_model,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
            
            # Build conversation history for context
            messages = await self._build_conversation_messages(session, request)
            
            # Use function calling
            start_time = datetime.now()
            result = await self.function_manager.chat_with_native_functions(
                model=model,
                messages=messages,
                system_instruction=system_instruction
            )
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            response_text = result.get("response", "I apologize, but I couldn't process your request.")
            tools_used = result.get("function_calls_made", [])
            thinking_process = f"Used {'native' if not result.get('fallback_mode') else 'legacy'} function calling with {len(tools_used)} tool calls. Budget: {thinking_budget}"
            
            # If no tools were available, add a note
            if not result.get("tools_available", False):
                thinking_process += " (No tools available - using direct chat)"
            
            # Track cost and usage
            cost_info = await self._track_api_call(
                request, session, response_text, thinking_process, response_time_ms
            )
            
            # Update session history
            await self._update_session_history(
                request, response_text, tools_used, cost_info, thinking_process, response_time_ms
            )
            
            # Create response
            chat_response = ChatResponse(
                response=response_text,
                thinking_process=thinking_process,
                tools_used=tools_used,
                session_id=request.session_id,
                timestamp=datetime.now().isoformat()
            )
            
            # Add cost information to response
            if self.cost_tracker and cost_info.get("cost_tracking") in ["success", "budget_exceeded"]:
                chat_response.cost_info = self._build_cost_info_response(cost_info, response_time_ms)
            elif self.cost_tracker and cost_info.get("cost_tracking") == "error":
                chat_response.cost_info = {
                    "tracking_status": "error",
                    "error": cost_info.get("error", "Unknown cost tracking error"),
                    "total_cost": "0.00",
                    "model": self.default_model,
                    "response_time_ms": response_time_ms
                }
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _check_budget_before_request(self, request: ChatRequest, session: Dict[str, Any]):
        """Check budget limits before processing request"""
        try:
            budget_check = await self.cost_tracker.check_budget_before_request(
                session_id=request.session_id,
                user_id=session.get("user_id"),
                model_name=self.default_model,
                input_text=request.message,
                estimated_output_tokens=1500  # Conservative estimate
            )
            
            if not budget_check.get("allowed", True):
                budget_info = budget_check.get("budget_check", {})
                limits_exceeded = budget_info.get("limits_exceeded", [])
                
                error_msg = "Request blocked - budget limit would be exceeded. "
                for limit in limits_exceeded:
                    error_msg += f"{limit['type'].title()} budget: estimated ${budget_check.get('estimated_cost', '0.00')} would exceed ${limit['limit']}. "
                
                logger.warning(f"Blocked expensive request for session {request.session_id}: {error_msg}")
                
                raise HTTPException(
                    status_code=429, 
                    detail=error_msg + f"Estimated cost: ${budget_check.get('estimated_cost', '0.00')}"
                )
            
            # Log warnings if any
            warnings = budget_check.get("budget_check", {}).get("warnings", [])
            if warnings:
                for warning in warnings:
                    logger.warning(f"Budget warning for session {request.session_id}: {warning}")
                    
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Pre-request budget check error: {e}")
            # Continue with request on error (fail open for availability)
    
    def _build_system_instruction(self, request: ChatRequest) -> str:
        """Build system instruction for the chat"""
        return f"""You are an AI assistant that can help users interact with their Airtable data. 
You have access to Airtable tools for various operations like listing tables, getting records, creating/updating/deleting records, and searching.

When users ask about Airtable data, use the appropriate tools to help them. Always provide clear explanations of what you're doing and what the results mean.

Current conversation context:
- Session ID: {request.session_id}
- Base ID: {request.base_id or 'Not specified - ask user for their Airtable base ID if needed'}

If the user hasn't provided a base ID and you need one for Airtable operations, ask them to provide it.
"""
    
    async def _build_conversation_messages(self, session: Dict[str, Any], request: ChatRequest) -> List[Dict[str, str]]:
        """Build conversation messages with recent history"""
        # Build conversation history for context
        recent_history = session.get('history', [])[-5:]  # Last 5 messages for context
        messages = []
        
        # Add recent history
        for msg in recent_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("message", "")
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        return messages
    
    async def _track_api_call(
        self,
        request: ChatRequest,
        session: Dict[str, Any],
        response_text: str,
        thinking_process: str,
        response_time_ms: int
    ) -> Dict[str, Any]:
        """Track API call cost if cost tracking is available"""
        cost_info = {"cost_tracking": "disabled", "cost": "0.00"}
        
        if self.cost_tracker:
            try:
                cost_info = await self.cost_tracker.track_api_call(
                    session_id=request.session_id,
                    user_id=session.get("user_id"),
                    model_name=self.default_model,
                    input_text=request.message,
                    output_text=response_text,
                    thinking_text=thinking_process,
                    response_time_ms=response_time_ms,
                    success=True
                )
                
                # Check for budget warnings
                if cost_info.get("budget_warnings"):
                    for warning in cost_info["budget_warnings"]:
                        logger.warning(f"Budget warning for session {request.session_id}: {warning}")
                
                # Check if budget was exceeded
                if cost_info.get("cost_tracking") == "budget_exceeded":
                    budget_info = cost_info.get("budget_check", {})
                    limits_exceeded = budget_info.get("limits_exceeded", [])
                    
                    error_msg = "Budget limit exceeded. "
                    for limit in limits_exceeded:
                        error_msg += f"{limit['type'].title()} budget: ${limit['would_spend']} would exceed ${limit['limit']}. "
                    
                    raise HTTPException(status_code=429, detail=error_msg)
                    
            except HTTPException:
                raise  # Re-raise HTTP exceptions
            except Exception as e:
                logger.error(f"Cost tracking error: {e}")
                cost_info = {"cost_tracking": "error", "cost": "0.00", "error": str(e)}
        
        return cost_info
    
    async def _update_session_history(
        self,
        request: ChatRequest,
        response_text: str,
        tools_used: List[str],
        cost_info: Dict[str, Any],
        thinking_process: str,
        response_time_ms: int
    ):
        """Update session history with cost information"""
        # Update session history with cost information
        await self.session_manager.add_message(
            request.session_id, 
            "user", 
            request.message,
            token_count=int(cost_info.get("cost_data", {}).get("input_tokens", len(request.message.split()) * 1.3)),
            cost=cost_info.get("cost_data", {}).get("input_cost", "0.00")
        )
        await self.session_manager.add_message(
            request.session_id, 
            "assistant", 
            response_text, 
            tools_used=tools_used,
            token_count=int(cost_info.get("cost_data", {}).get("output_tokens", len(response_text.split()) * 1.3)),
            cost=cost_info.get("cost_data", {}).get("total_cost", "0.00"),
            model_used=self.default_model,
            thinking_process=thinking_process,
            response_time_ms=response_time_ms
        )
    
    def _build_cost_info_response(self, cost_info: Dict[str, Any], response_time_ms: int) -> Dict[str, Any]:
        """Build cost information for response"""
        cost_data = cost_info.get("cost_data", {})
        return {
            "tracking_status": cost_info.get("cost_tracking"),
            "total_cost": cost_info["cost"],
            "token_details": {
                "input_tokens": cost_data.get("input_tokens", 0),
                "output_tokens": cost_data.get("output_tokens", 0),
                "thinking_tokens": cost_data.get("thinking_tokens", 0),
                "total_tokens": cost_data.get("total_tokens", 0),
                "counting_method": cost_data.get("token_counting_method", "unknown")
            },
            "cost_breakdown": {
                "input_cost": cost_data.get("input_cost", "0.00"),
                "output_cost": cost_data.get("output_cost", "0.00"),
                "thinking_cost": cost_data.get("thinking_cost", "0.00")
            },
            "model": self.default_model,
            "response_time_ms": response_time_ms,
            "budget_warnings": cost_info.get("budget_warnings", []),
            "budget_status": cost_info.get("budget_status")
        }