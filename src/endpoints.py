"""
Additional API endpoints for LLM Orchestrator
Contains health checks, function calling status, and analytics endpoints
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def function_calling_status_handler(mcp_client):
    """Get status of function calling implementation"""
    try:
        from .gemini_function_calling import GeminiFunctionCallHandler
        
        # Test function calling setup
        handler = GeminiFunctionCallHandler(mcp_client)
        tools = await handler.get_gemini_tools()
        
        return {
            "native_function_calling": True,
            "tools_available": len(tools),
            "tools": [tool.function_declarations[0].name for tool in tools if tool.function_declarations] if tools else [],
            "mcp_client_type": type(mcp_client).__name__,
            "status": "operational"
        }
        
    except ImportError:
        return {
            "native_function_calling": False,
            "tools_available": 0,
            "fallback_mode": "keyword_matching",
            "status": "fallback"
        }
    except Exception as e:
        return {
            "native_function_calling": False,
            "error": str(e),
            "status": "error"
        }


async def cost_tracking_status_handler(cost_tracker):
    """Get cost tracking system status"""
    if not cost_tracker or not cost_tracker.enabled:
        return {
            "cost_tracking": False,
            "status": "disabled",
            "reason": "Cost tracking dependencies not available"
        }
    
    return {
        "cost_tracking": True,
        "status": "enabled",
        "features": {
            "budget_management": True,
            "usage_tracking": True,
            "database_logging": True,
            "real_time_monitoring": True
        }
    }


async def health_check_handler(session_manager, cost_tracker):
    """Health check endpoint with session manager status"""
    base_status = {"status": "healthy", "service": "llm-orchestrator", "version": "2.0.0"}
    
    # Add session manager health info
    if hasattr(session_manager, 'health_check'):
        try:
            session_health = await session_manager.health_check()
            base_status["session_manager"] = session_health
        except Exception as e:
            base_status["session_manager"] = {"error": str(e)}
    else:
        base_status["session_manager"] = {"type": "legacy", "status": "unknown"}
    
    # Add cost tracking status
    base_status["cost_tracking"] = {
        "enabled": cost_tracker.enabled if cost_tracker else False,
        "status": "operational" if cost_tracker and cost_tracker.enabled else "disabled"
    }
    
    return base_status


async def cost_analytics_handler(days: int = 7):
    """Get cost analytics for the specified number of days"""
    try:
        from pyairtable_common.database import get_async_session
        from pyairtable_common.database.models import ApiUsageLog
        from sqlalchemy import select, func
        
        async with get_async_session() as db:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Get total usage stats
            result = await db.execute(
                select(
                    func.count(ApiUsageLog.id).label("total_calls"),
                    func.sum(ApiUsageLog.total_tokens).label("total_tokens"),
                    func.sum(func.cast(ApiUsageLog.cost, db.bind.dialect.NUMERIC)).label("total_cost"),
                    func.avg(ApiUsageLog.response_time_ms).label("avg_response_time"),
                    func.count(ApiUsageLog.id).filter(ApiUsageLog.success == True).label("successful_calls"),
                    func.count(ApiUsageLog.id).filter(ApiUsageLog.success == False).label("failed_calls")
                ).where(
                    ApiUsageLog.timestamp >= cutoff_date,
                    ApiUsageLog.service_name == "gemini"
                )
            )
            
            stats = result.first()
            
            # Get daily breakdown
            daily_result = await db.execute(
                select(
                    func.date(ApiUsageLog.timestamp).label("date"),
                    func.count(ApiUsageLog.id).label("calls"),
                    func.sum(func.cast(ApiUsageLog.cost, db.bind.dialect.NUMERIC)).label("cost"),
                    func.sum(ApiUsageLog.total_tokens).label("tokens")
                ).where(
                    ApiUsageLog.timestamp >= cutoff_date,
                    ApiUsageLog.service_name == "gemini"
                ).group_by(func.date(ApiUsageLog.timestamp))
                .order_by(func.date(ApiUsageLog.timestamp))
            )
            
            daily_stats = [{
                "date": str(row.date),
                "calls": int(row.calls),
                "cost": str(row.cost or 0),
                "tokens": int(row.tokens or 0)
            } for row in daily_result]
            
            return {
                "period_days": days,
                "total_stats": {
                    "total_calls": int(stats.total_calls or 0),
                    "successful_calls": int(stats.successful_calls or 0),
                    "failed_calls": int(stats.failed_calls or 0),
                    "total_tokens": int(stats.total_tokens or 0),
                    "total_cost": str(stats.total_cost or 0),
                    "avg_response_time_ms": float(stats.avg_response_time or 0),
                    "success_rate": float((stats.successful_calls or 0) / max(stats.total_calls or 1, 1) * 100)
                },
                "daily_breakdown": daily_stats,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error generating cost analytics: {e}")
        return {
            "cost_tracking": "error",
            "error": str(e)
        }


async def circuit_breaker_status_handler():
    """Get status of all circuit breakers"""
    try:
        from pyairtable_common.resilience import circuit_breaker_registry
        stats = await circuit_breaker_registry.get_all_stats()
        return stats
    except ImportError:
        return {
            "circuit_breakers": {},
            "total_breakers": 0,
            "message": "Circuit breaker monitoring not available",
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving circuit breaker status")


async def services_health_handler():
    """Get health status of all dependent services"""
    try:
        from pyairtable_common.http import service_registry
        health_checks = await service_registry.health_check_all()
        return health_checks
    except ImportError:
        return {
            "overall_status": "unknown",
            "services": {},
            "total_services": 0,
            "message": "Service health monitoring not available",
            "checked_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking services health: {e}")
        raise HTTPException(status_code=500, detail="Error checking services health")