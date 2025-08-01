"""
Budget management API endpoints for LLM Orchestrator
"""

import logging
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

# Import cost tracking
try:
    from cost_tracking_middleware import cost_tracker
    COST_TRACKING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Cost tracking not available: {e}")
    COST_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router for budget management endpoints
budget_router = APIRouter(prefix="/budgets", tags=["Budget Management"])


# Request/Response Models
class SessionBudgetRequest(BaseModel):
    """Request model for creating/updating session budgets"""
    budget_limit: float = Field(..., gt=0, description="Budget limit in USD")
    alert_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Alert threshold (0.0 to 1.0)")
    reset_period: str = Field("session", description="Reset period: session, daily, weekly, monthly")
    
    @validator('reset_period')
    def validate_reset_period(cls, v):
        allowed = ["session", "daily", "weekly", "monthly"]
        if v not in allowed:
            raise ValueError(f"reset_period must be one of: {allowed}")
        return v


class UserBudgetRequest(BaseModel):
    """Request model for creating/updating user budgets"""
    budget_limit: float = Field(..., gt=0, description="Budget limit in USD")
    alert_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Alert threshold (0.0 to 1.0)")
    reset_period: str = Field("monthly", description="Reset period: daily, weekly, monthly")
    
    @validator('reset_period')
    def validate_reset_period(cls, v):
        allowed = ["daily", "weekly", "monthly"]
        if v not in allowed:
            raise ValueError(f"reset_period must be one of: {allowed}")
        return v


class BudgetResponse(BaseModel):
    """Response model for budget operations"""
    success: bool
    message: str
    budget: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BudgetStatusResponse(BaseModel):
    """Response model for budget status"""
    session_id: Optional[str]
    user_id: Optional[str]
    budgets: Dict[str, Any]
    total_spending: Optional[str] = None
    warnings: Optional[list] = None


# Session Budget Endpoints
@budget_router.post("/session/{session_id}", response_model=BudgetResponse)
async def create_session_budget(session_id: str, request: SessionBudgetRequest):
    """
    Create or update a session budget
    
    Args:
        session_id: Session identifier
        request: Budget configuration
        
    Returns:
        Budget creation result
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        budget_limit = Decimal(str(request.budget_limit))
        
        if cost_tracker.budget_manager:
            budget = await cost_tracker.budget_manager.set_session_budget(
                session_id=session_id,
                budget_limit=budget_limit
            )
            
            return BudgetResponse(
                success=True,
                message=f"Session budget set: ${budget_limit}",
                budget=budget
            )
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error creating session budget: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session budget: {str(e)}")


@budget_router.get("/session/{session_id}", response_model=BudgetStatusResponse)
async def get_session_budget(session_id: str):
    """
    Get session budget status
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session budget status
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        if cost_tracker.budget_manager:
            status = await cost_tracker.budget_manager.get_budget_status(session_id)
            
            return BudgetStatusResponse(
                session_id=session_id,
                user_id=None,
                budgets=status.get("budgets", {}),
                total_spending=status.get("budgets", {}).get("session", {}).get("spent")
            )
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error getting session budget: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session budget: {str(e)}")


@budget_router.delete("/session/{session_id}")
async def reset_session_budget(session_id: str):
    """
    Reset session budget (clear spending)
    
    Args:
        session_id: Session identifier
        
    Returns:
        Reset result
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        if cost_tracker.budget_manager:
            success = await cost_tracker.budget_manager.reset_session_budget(session_id)
            
            if success:
                return {"success": True, "message": f"Session budget reset: {session_id}"}
            else:
                raise HTTPException(status_code=404, detail="Session budget not found")
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error resetting session budget: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting session budget: {str(e)}")


# User Budget Endpoints
@budget_router.post("/user/{user_id}", response_model=BudgetResponse)
async def create_user_budget(user_id: str, request: UserBudgetRequest):
    """
    Create or update a user budget
    
    Args:
        user_id: User identifier
        request: Budget configuration
        
    Returns:
        Budget creation result
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        budget_limit = Decimal(str(request.budget_limit))
        
        if cost_tracker.budget_manager:
            budget = await cost_tracker.budget_manager.set_user_budget(
                user_id=user_id,
                budget_limit=budget_limit,
                reset_period=request.reset_period
            )
            
            return BudgetResponse(
                success=True,
                message=f"User budget set: ${budget_limit} ({request.reset_period})",
                budget=budget
            )
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error creating user budget: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating user budget: {str(e)}")


@budget_router.get("/user/{user_id}", response_model=BudgetStatusResponse)
async def get_user_budget(
    user_id: str,
    reset_period: str = Query("monthly", description="Budget period to check")
):
    """
    Get user budget status
    
    Args:
        user_id: User identifier
        reset_period: Budget period to check
        
    Returns:
        User budget status
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        if cost_tracker.budget_manager:
            status = await cost_tracker.budget_manager.get_budget_status(None, user_id)
            
            return BudgetStatusResponse(
                session_id=None,
                user_id=user_id,
                budgets=status.get("budgets", {}),
                total_spending=status.get("budgets", {}).get("user", {}).get("spent")
            )
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error getting user budget: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting user budget: {str(e)}")


# Combined Status Endpoints
@budget_router.get("/status/{session_id}")
async def get_budget_status(
    session_id: str,
    user_id: Optional[str] = Query(None, description="Optional user ID")
):
    """
    Get comprehensive budget status for session and optionally user
    
    Args:
        session_id: Session identifier
        user_id: Optional user identifier
        
    Returns:
        Combined budget status
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        if cost_tracker.budget_manager:
            status = await cost_tracker.budget_manager.get_budget_status(session_id, user_id)
            
            # Calculate total spending
            session_spent = Decimal("0.00")
            user_spent = Decimal("0.00")
            warnings = []
            
            if "session" in status.get("budgets", {}):
                session_spent = Decimal(status["budgets"]["session"]["spent"])
                usage_percent = status["budgets"]["session"]["usage_percent"]
                if usage_percent >= 80:
                    warnings.append(f"Session budget {usage_percent:.1f}% used")
            
            if "user" in status.get("budgets", {}):
                user_spent = Decimal(status["budgets"]["user"]["spent"])
                usage_percent = status["budgets"]["user"]["usage_percent"]
                if usage_percent >= 80:
                    warnings.append(f"User budget {usage_percent:.1f}% used")
            
            total_spending = str(session_spent + user_spent)
            
            return BudgetStatusResponse(
                session_id=session_id,
                user_id=user_id,
                budgets=status.get("budgets", {}),
                total_spending=total_spending,
                warnings=warnings if warnings else None
            )
        else:
            raise HTTPException(status_code=503, detail="Budget manager not available")
            
    except Exception as e:
        logger.error(f"Error getting budget status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting budget status: {str(e)}")


# Cost Estimation Endpoint
@budget_router.post("/estimate-cost")
async def estimate_request_cost(
    session_id: str = Query(..., description="Session ID"),
    user_id: Optional[str] = Query(None, description="Optional user ID"),
    input_text: str = Query(..., description="Input text to estimate cost for"),
    model_name: str = Query("gemini-2.5-flash", description="Model to use"),
    estimated_output_tokens: int = Query(1000, description="Estimated output tokens")
):
    """
    Estimate cost for a request and check budget limits
    
    Args:
        session_id: Session identifier
        user_id: Optional user identifier
        input_text: Input text to estimate
        model_name: Model name
        estimated_output_tokens: Estimated output tokens
        
    Returns:
        Cost estimation and budget check
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        budget_check = await cost_tracker.check_budget_before_request(
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            input_text=input_text,
            estimated_output_tokens=estimated_output_tokens
        )
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "model_name": model_name,
            "estimated_cost": budget_check.get("estimated_cost"),
            "estimated_input_tokens": budget_check.get("estimated_input_tokens"),
            "estimated_output_tokens": budget_check.get("estimated_output_tokens"),
            "allowed": budget_check.get("allowed"),
            "budget_check": budget_check.get("budget_check", {}),
            "warnings": budget_check.get("budget_check", {}).get("warnings", [])
        }
        
    except Exception as e:
        logger.error(f"Error estimating request cost: {e}")
        raise HTTPException(status_code=500, detail=f"Error estimating request cost: {str(e)}")


# Usage Statistics Endpoint
@budget_router.get("/usage/{session_id}")
async def get_session_usage(session_id: str):
    """
    Get detailed usage statistics for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session usage statistics
    """
    if not COST_TRACKING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Cost tracking service unavailable")
    
    try:
        # Get cost summary from cost tracker
        summary = await cost_tracker.get_session_cost_summary(session_id)
        
        # Get budget status
        budget_status = await cost_tracker.get_budget_status(session_id)
        
        return {
            "session_id": session_id,
            "usage_summary": summary,
            "budget_status": budget_status,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting session usage: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session usage: {str(e)}")


# Health Check Endpoint
@budget_router.get("/health")
async def budget_health_check():
    """Check budget management system health"""
    try:
        status = {
            "cost_tracking_available": COST_TRACKING_AVAILABLE,
            "budget_manager_available": False,
            "database_backend": False,
            "timestamp": datetime.now().isoformat()
        }
        
        if COST_TRACKING_AVAILABLE and cost_tracker.budget_manager:
            status["budget_manager_available"] = True
            status["database_backend"] = cost_tracker.budget_manager.use_database
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking budget health: {e}")
        return {
            "cost_tracking_available": False,
            "budget_manager_available": False,
            "database_backend": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }