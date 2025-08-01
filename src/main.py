"""
LLM Orchestrator Service - Modular Architecture
Integrates Gemini 2.5 Flash with MCP tools for Airtable operations
"""

import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Import modular components
from .chat import ChatRequest, ChatResponse
from .config import load_configuration, setup_middleware, cleanup_configuration
from .app_factory import initialize_components, cleanup_components
from .endpoints import (
    function_calling_status_handler, 
    cost_tracking_status_handler,
    health_check_handler,
    cost_analytics_handler,
    circuit_breaker_status_handler,
    services_health_handler
)

# Global configuration and components
app_config, config_manager = load_configuration()
session_manager = None
mcp_client = None
chat_handler = None
cost_tracker = None


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global session_manager, mcp_client, chat_handler, cost_tracker
    
    # Startup
    session_manager, mcp_client, chat_handler, cost_tracker = await initialize_components(app_config)
    
    yield
    
    # Shutdown
    await cleanup_components(session_manager, mcp_client, config_manager)
    await cleanup_configuration(config_manager)


# Initialize FastAPI app
app = FastAPI(
    title="LLM Orchestrator",
    description="Gemini 2.5 Flash integration with MCP tools - Modular Architecture",
    version="2.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app, app_config)

# Main API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with native Gemini function calling and budget enforcement"""
    return await chat_handler.handle_chat_request(request)

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    tools = await mcp_client.get_tools()
    return {"tools": tools}

@app.post("/execute-tool")
async def execute_tool(request: ToolExecutionRequest):
    """Execute a specific MCP tool directly"""
    result = await mcp_client.call_tool(request.tool_name, request.arguments)
    return {"result": result, "tool": request.tool_name}

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = await session_manager.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving session history")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    try:
        cleared = await session_manager.clear_session(session_id)
        if cleared:
            return {"message": f"Session {session_id} cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Error clearing session")

# Health and Status Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint with session manager status"""
    return await health_check_handler(session_manager, cost_tracker)

@app.get("/function-calling/status")
async def function_calling_status():
    """Get status of function calling implementation"""
    return await function_calling_status_handler(mcp_client)

@app.get("/cost-tracking/status")
async def cost_tracking_status():
    """Get cost tracking system status"""
    return await cost_tracking_status_handler(cost_tracker)

@app.get("/cost-tracking/analytics")
async def get_cost_analytics(days: int = 7):
    """Get cost analytics for the specified number of days"""
    return await cost_analytics_handler(days)

@app.get("/sessions/{session_id}/cost-summary")
async def get_session_cost_summary(session_id: str):
    """Get cost summary for a specific session"""
    if not cost_tracker or not cost_tracker.enabled:
        return {"cost_tracking": "disabled"}
    return await cost_tracker.get_session_cost_summary(session_id)

@app.get("/health/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    return await circuit_breaker_status_handler()

@app.get("/health/services")
async def get_services_health():
    """Get health status of all dependent services"""
    return await services_health_handler()

# Include budget management API
try:
    from .budget_api import budget_router
    app.include_router(budget_router)
    logger.info("✅ Budget management API endpoints enabled")
except ImportError as e:
    logger.warning(f"⚠️ Budget API not available: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)))