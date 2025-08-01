"""
Cost tracking and budget management for LLM API calls
Provides cost calculation, budget enforcement, and usage analytics
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
from decimal import Decimal

logger = logging.getLogger(__name__)


class CostTrackingManager:
    """
    Manages cost tracking for Gemini API calls with budget enforcement
    """
    
    def __init__(self):
        self.enabled = False
        self.cost_calculator = None
        self.budget_manager = None
        
        # Try to import cost tracking dependencies
        try:
            # Add path to pyairtable-common
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../pyairtable-common'))
            from pyairtable_common.cost_tracking import cost_calculator, budget_manager
            from pyairtable_common.database import get_async_session
            from pyairtable_common.database.models import ApiUsageLog
            
            self.cost_calculator = cost_calculator
            self.budget_manager = budget_manager
            self.get_async_session = get_async_session
            self.ApiUsageLog = ApiUsageLog
            self.enabled = True
            logger.info("✅ Cost tracking initialized")
            
        except ImportError as e:
            logger.warning(f"⚠️ Cost tracking dependencies not available: {e}")
    
    async def track_api_call(
        self,
        session_id: str,
        user_id: Optional[str],
        model_name: str,
        input_text: str,
        output_text: str,
        thinking_text: str = "",
        correlation_id: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track a Gemini API call and calculate costs
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            model_name: Gemini model used
            input_text: Input text content
            output_text: Output text content
            thinking_text: Thinking process text
            correlation_id: Request correlation ID
            response_time_ms: Response time in milliseconds
            success: Whether the call was successful
            error_message: Error message if failed
            
        Returns:
            Dict with cost information and tracking status
        """
        if not self.enabled:
            return {"cost_tracking": "disabled", "cost": "0.00"}
        
        try:
            # Calculate cost using real token counting
            cost_data = await self.cost_calculator.calculate_cost_from_text_real(
                model_name=model_name,
                input_text=input_text,
                output_text=output_text,
                thinking_text=thinking_text
            )
            
            cost = Decimal(cost_data["total_cost"])
            
            # Check budget limits before recording
            if success:  # Only check budgets for successful calls
                budget_check = await self.budget_manager.check_budget_limits(session_id, user_id, cost)
                
                if not budget_check["allowed"]:
                    logger.warning(f"Budget limit would be exceeded: {budget_check['limits_exceeded']}")
                    return {
                        "cost_tracking": "budget_exceeded",
                        "cost": str(cost),
                        "budget_check": budget_check,
                        "cost_data": cost_data
                    }
                
                # Record usage against budgets
                usage_result = await self.budget_manager.record_usage(session_id, user_id, cost)
                if usage_result.get("alerts"):
                    logger.warning(f"Budget alerts triggered: {usage_result['alerts']}")
            
            # Log to database
            await self._log_to_database(
                session_id=session_id,
                user_id=user_id,
                correlation_id=correlation_id,
                model_name=model_name,
                cost_data=cost_data,
                response_time_ms=response_time_ms,
                success=success,
                error_message=error_message
            )
            
            result = {
                "cost_tracking": "success",
                "cost": str(cost),
                "cost_data": cost_data,
                "budget_warnings": budget_check.get("warnings", []) if success else []
            }
            
            # Add budget status if there were warnings
            if success and budget_check.get("warnings"):
                result["budget_status"] = await self.budget_manager.get_budget_status(session_id, user_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error tracking API call cost: {e}")
            return {
                "cost_tracking": "error",
                "cost": "0.00",
                "error": str(e)
            }
    
    async def _log_to_database(
        self,
        session_id: str,
        user_id: Optional[str],
        correlation_id: Optional[str],
        model_name: str,
        cost_data: Dict[str, Any],
        response_time_ms: Optional[int],
        success: bool,
        error_message: Optional[str]
    ):
        """Log API usage to database"""
        try:
            async with self.get_async_session() as db:
                usage_log = self.ApiUsageLog(
                    session_id=session_id,
                    user_id=user_id,
                    correlation_id=correlation_id,
                    service_name="gemini",
                    endpoint="generate_content",
                    method="POST",
                    input_tokens=cost_data["input_tokens"],
                    output_tokens=cost_data["output_tokens"],
                    total_tokens=cost_data["total_tokens"],
                    cost=cost_data["total_cost"],
                    response_time_ms=response_time_ms,
                    status_code=200 if success else 500,
                    success=success,
                    model_used=model_name,
                    error_message=error_message,
                    metadata={
                        "thinking_tokens": cost_data["thinking_tokens"],
                        "pricing_used": cost_data["pricing_used"],
                        "estimated": cost_data.get("estimated", False)
                    }
                )
                
                db.add(usage_log)
                await db.commit()
                
                logger.debug(f"Logged API usage: {session_id} - ${cost_data['total_cost']}")
                
        except Exception as e:
            logger.error(f"Error logging to database: {e}")
    
    async def get_session_cost_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get cost summary for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with cost summary
        """
        if not self.enabled:
            return {"cost_tracking": "disabled"}
        
        try:
            async with self.get_async_session() as db:
                from sqlalchemy import select, func
                
                # Get usage stats for session
                result = await db.execute(
                    select(
                        func.count(self.ApiUsageLog.id).label("total_calls"),
                        func.sum(self.ApiUsageLog.total_tokens).label("total_tokens"),
                        func.sum(func.cast(self.ApiUsageLog.cost, db.bind.dialect.NUMERIC)).label("total_cost"),
                        func.avg(self.ApiUsageLog.response_time_ms).label("avg_response_time"),
                        func.count(self.ApiUsageLog.id).filter(self.ApiUsageLog.success == True).label("successful_calls")
                    ).where(self.ApiUsageLog.session_id == session_id)
                )
                
                stats = result.first()
                
                if not stats or not stats.total_calls:
                    return {
                        "session_id": session_id,
                        "total_calls": 0,
                        "total_cost": "0.00",
                        "cost_tracking": "no_data"
                    }
                
                # Get budget status
                budget_status = self.budget_manager.get_budget_status(session_id)
                
                return {
                    "session_id": session_id,
                    "total_calls": int(stats.total_calls),
                    "successful_calls": int(stats.successful_calls or 0),
                    "total_tokens": int(stats.total_tokens or 0),
                    "total_cost": str(stats.total_cost or Decimal("0.00")),
                    "avg_response_time_ms": float(stats.avg_response_time or 0),
                    "budget_status": budget_status,
                    "cost_tracking": "success"
                }
                
        except Exception as e:
            logger.error(f"Error getting session cost summary: {e}")
            return {
                "session_id": session_id,
                "cost_tracking": "error",
                "error": str(e)
            }
    
    async def set_session_budget(self, session_id: str, budget_limit: float):
        """Set budget limit for a session"""
        if self.enabled and self.budget_manager:
            return await self.budget_manager.set_session_budget(session_id, Decimal(str(budget_limit)))
    
    async def set_user_budget(self, user_id: str, budget_limit: float, reset_period: str = "monthly"):
        """Set budget limit for a user"""
        if self.enabled and self.budget_manager:
            return await self.budget_manager.set_user_budget(user_id, Decimal(str(budget_limit)), reset_period)
    
    async def get_budget_status(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get budget status"""
        if not self.enabled or not self.budget_manager:
            return {"cost_tracking": "disabled"}
        
        return await self.budget_manager.get_budget_status(session_id, user_id)
    
    async def check_budget_before_request(
        self,
        session_id: str,
        user_id: Optional[str],
        model_name: str,
        input_text: str,
        estimated_output_tokens: int = 1000  # Conservative estimate
    ) -> Dict[str, Any]:
        """
        Check budget limits before making API request to prevent overspend
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            model_name: Gemini model name
            input_text: Input text to estimate cost
            estimated_output_tokens: Estimated output tokens (conservative)
            
        Returns:
            Dict with budget check result and estimated cost
        """
        if not self.enabled:
            return {"allowed": True, "cost_tracking": "disabled"}
        
        try:
            # Count actual input tokens
            input_token_count = await self.cost_calculator.count_tokens_real(
                model_name=model_name,
                input_text=input_text,
                output_text="",
                thinking_text=""
            )
            
            input_tokens = input_token_count["input_tokens"]
            
            # Estimate cost with real input tokens + estimated output
            estimated_cost_data = self.cost_calculator.calculate_cost(
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=estimated_output_tokens,
                thinking_tokens=int(estimated_output_tokens * 0.1)  # 10% thinking estimate
            )
            
            estimated_cost = Decimal(estimated_cost_data["total_cost"])
            
            # Check budget limits
            budget_check = await self.budget_manager.check_budget_limits(
                session_id, user_id, estimated_cost
            )
            
            return {
                "allowed": budget_check["allowed"],
                "estimated_cost": str(estimated_cost),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "budget_check": budget_check,
                "cost_tracking": "success"
            }
            
        except Exception as e:
            logger.error(f"Error checking budget before request: {e}")
            # Allow request on error (fail open)
            return {
                "allowed": True,
                "cost_tracking": "error",
                "error": str(e)
            }


def cost_tracking_decorator(func: Callable) -> Callable:
    """
    Decorator to add cost tracking to Gemini API calls
    
    Usage:
        @cost_tracking_decorator
        async def my_gemini_call(session_id, message, ...):
            # Your Gemini API call here
            return response
    """
    async def wrapper(*args, **kwargs):
        # Extract session info from kwargs
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        
        if not session_id:
            # Try to find session_id in args (if it's a positional argument)
            for arg in args:
                if isinstance(arg, str) and arg.startswith(("session-", "user-", "sess_")):
                    session_id = arg
                    break
        
        start_time = datetime.now()
        
        try:
            result = await func(*args, **kwargs)
            
            # Calculate response time
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Track the call if we have the necessary info
            if session_id and hasattr(result, 'text'):
                input_text = kwargs.get("message", "")
                output_text = result.text
                thinking_text = getattr(result, 'thinking', "") or ""
                model_name = kwargs.get("model_name", "gemini-2.5-flash")
                
                # Create global tracker if needed
                global_tracker = CostTrackingManager()
                
                cost_info = await global_tracker.track_api_call(
                    session_id=session_id,
                    user_id=user_id,
                    model_name=model_name,
                    input_text=input_text,
                    output_text=output_text,
                    thinking_text=thinking_text,
                    response_time_ms=response_time_ms,
                    success=True
                )
                
                # Attach cost info to result if possible
                if hasattr(result, '__dict__'):
                    result.cost_info = cost_info
            
            return result
            
        except Exception as e:
            # Track failed call
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            if session_id:
                global_tracker = CostTrackingManager()
                await global_tracker.track_api_call(
                    session_id=session_id,
                    user_id=user_id,
                    model_name=kwargs.get("model_name", "gemini-2.5-flash"),
                    input_text=kwargs.get("message", ""),
                    output_text="",
                    response_time_ms=response_time_ms,
                    success=False,
                    error_message=str(e)
                )
            
            raise
    
    return wrapper