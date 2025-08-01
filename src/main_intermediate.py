"""
LLM Orchestrator Service - Modular Architecture
Integrates Gemini 2.5 Flash with MCP tools for Airtable operations
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Import modular components
from chat import ChatHandler, ChatRequest, ChatResponse
from session.manager import HybridSessionManager
from mcp.http_client import MCPHttpClient
from mcp.stdio_client import MCPStdioClient
from cost.tracking import CostTrackingManager

# Secure configuration imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

try:
    from pyairtable_common.config import initialize_secrets, get_secret, close_secrets
    from pyairtable_common.middleware import setup_security_middleware
    SECURE_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Secure configuration not available: {e}")
    SECURE_CONFIG_AVAILABLE = False

# Resilient HTTP client
try:
    from pyairtable_common.middleware import add_circuit_breaker_middleware, SERVICE_CONFIGS
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Circuit breaker middleware not available: {e}")
    CIRCUIT_BREAKER_AVAILABLE = False

# Configuration Management
def load_configuration():
    """Load and validate configuration from secure sources"""
    config_manager = None
    
    if SECURE_CONFIG_AVAILABLE:
        try:
            config_manager = initialize_secrets()
            gemini_api_key = get_secret("GEMINI_API_KEY")
            redis_password = get_secret("REDIS_PASSWORD")
            logger.info("✅ Secure configuration manager initialized")
        except Exception as e:
            logger.error(f"💥 Failed to initialize secure configuration: {e}")
            raise
    else:
        # Fallback to environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        redis_password = os.getenv("REDIS_PASSWORD")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Other configuration (non-secret)
    config = {
        "gemini_api_key": gemini_api_key,
        "redis_password": redis_password,
        "mcp_server_stdio_command": os.getenv("MCP_SERVER_STDIO_COMMAND", "python -m src.server"),
        "mcp_server_working_dir": os.getenv("MCP_SERVER_WORKING_DIR", "/Users/kg/IdeaProjects/mcp-server-py"),
        "mcp_server_http_url": os.getenv("MCP_SERVER_HTTP_URL", "http://localhost:8001"),
        "use_http_mcp": os.getenv("USE_HTTP_MCP", "true").lower() == "true",
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        "use_redis_sessions": os.getenv("USE_REDIS_SESSIONS", "true").lower() == "true",
        "thinking_budget": int(os.getenv("THINKING_BUDGET", "5")),
        "default_model": os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
        "max_tokens": int(os.getenv("MAX_TOKENS", "4000")),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600")),
        "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    }
    
    return config, config_manager

# Global components
app_config, config_manager = load_configuration()
session_manager = None
mcp_client = None
chat_handler = None
cost_tracker = None


# Request/Response Models
class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting LLM Orchestrator Service...")
    logger.info(f"Using Gemini model: {app_config['default_model']}")
    
    global session_manager, mcp_client, chat_handler, cost_tracker
    
    # Initialize cost tracking
    cost_tracker = CostTrackingManager()
    logger.info(f"Cost tracking: {'enabled' if cost_tracker.enabled else 'disabled'}")
    
    # Initialize MCP client
    if app_config["use_http_mcp"]:
        logger.info(f"HTTP MCP Server URL: {app_config['mcp_server_http_url']}")
        mcp_client = MCPHttpClient(app_config["mcp_server_http_url"])
    else:
        logger.info(f"Stdio MCP Server command: {app_config['mcp_server_stdio_command']}")
        mcp_client = MCPStdioClient(
            app_config["mcp_server_stdio_command"], 
            app_config["mcp_server_working_dir"]
        )
    
    # Initialize session manager
    session_manager = HybridSessionManager(
        redis_url=app_config["redis_url"],
        redis_password=app_config["redis_password"],
        session_timeout=app_config["session_timeout"],
        use_postgres=True,
        use_redis_fallback=app_config["use_redis_sessions"]
    )
    
    try:
        await session_manager.initialize()
        logger.info("✅ Session manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize session manager: {e}")
        raise
    
    # Initialize chat handler
    chat_handler = ChatHandler(
        session_manager=session_manager,
        mcp_client=mcp_client,
        cost_tracker=cost_tracker if cost_tracker.enabled else None,
        default_model=app_config["default_model"],
        max_tokens=app_config["max_tokens"],
        temperature=app_config["temperature"],
        thinking_budget=app_config["thinking_budget"]
    )
    
    logger.info("✅ LLM Orchestrator Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Orchestrator Service...")
    
    try:
        await session_manager.close()
        logger.info("Session manager closed")
    except Exception as e:
        logger.error(f"Error closing session manager: {e}")
    
    try:
        await mcp_client.close()
        logger.info("MCP client closed")
    except Exception as e:
        logger.error(f"Error closing MCP client: {e}")
    
    if config_manager:
        await close_secrets()
        logger.info("Closed secure configuration manager")


# Initialize FastAPI app
app = FastAPI(
    title="LLM Orchestrator",
    description="Gemini 2.5 Flash integration with MCP tools - Modular Architecture",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Add security middleware
if SECURE_CONFIG_AVAILABLE:
    setup_security_middleware(app, rate_limit_calls=500, rate_limit_period=60)

# Add circuit breaker middleware
if CIRCUIT_BREAKER_AVAILABLE:
    add_circuit_breaker_middleware(app, default_config=SERVICE_CONFIGS["llm_service"])
    logger.info("✅ Circuit breaker middleware enabled")


# API Endpoints

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


@app.get("/health")
async def health_check():
    """Health check endpoint with session manager status"""
    return await health_check_handler(session_manager, cost_tracker)


# Include budget management API
try:
    from budget_api import budget_router
    app.include_router(budget_router)
    logger.info("✅ Budget management API endpoints enabled")
except ImportError as e:
    logger.warning(f"⚠️ Budget API not available: {e}")

# Additional endpoints
from endpoints import (
    function_calling_status_handler, 
    cost_tracking_status_handler,
    health_check_handler,
    cost_analytics_handler,
    circuit_breaker_status_handler,
    services_health_handler
)

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

@app.get("/health/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    return await circuit_breaker_status_handler()

@app.get("/health/services")
async def get_services_health():
    """Get health status of all dependent services"""
    return await services_health_handler()


@app.get("/sessions/{session_id}/cost-summary")
async def get_session_cost_summary(session_id: str):
    """Get cost summary for a specific session"""
    if not cost_tracker or not cost_tracker.enabled:
        return {"cost_tracking": "disabled"}
    
    return await cost_tracker.get_session_cost_summary(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)))