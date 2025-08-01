"""
LLM Orchestrator Service
Integrates Gemini 2.5 Flash with MCP tools for Airtable operations
"""

import os
import sys
import json
import asyncio
import subprocess
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import redis.asyncio as redis
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Secure configuration imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

try:
    from pyairtable_common.config import initialize_secrets, get_secret, close_secrets, ConfigurationError
    from pyairtable_common.middleware import setup_security_middleware
    SECURE_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Secure configuration not available: {e}")
    SECURE_CONFIG_AVAILABLE = False

# Initialize cost tracking
try:
    from cost_tracking_middleware import cost_tracker
    COST_TRACKING_AVAILABLE = True
    logger.info("✅ Cost tracking available")
except ImportError as e:
    logger.warning(f"⚠️ Cost tracking not available: {e}")
    COST_TRACKING_AVAILABLE = False
    cost_tracker = None

# Initialize secure configuration manager
config_manager = None
if SECURE_CONFIG_AVAILABLE:
    try:
        config_manager = initialize_secrets()
        logger.info("✅ Secure configuration manager initialized")
    except Exception as e:
        logger.error(f"💥 Failed to initialize secure configuration: {e}")
        raise

# Initialize resilient HTTP client
try:
    from pyairtable_common.http import get_mcp_client
    RESILIENT_HTTP_AVAILABLE = True
    logger.info("✅ Resilient HTTP client available")
except ImportError as e:
    logger.warning(f"⚠️ Resilient HTTP client not available: {e}")
    RESILIENT_HTTP_AVAILABLE = False

# Configuration with secure secret retrieval
GEMINI_API_KEY = None
REDIS_PASSWORD = None

if config_manager:
    try:
        GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
        REDIS_PASSWORD = get_secret("REDIS_PASSWORD")
    except Exception as e:
        logger.error(f"💥 Failed to get secrets from secure config: {e}")
        raise ValueError("Required secrets could not be retrieved from secure configuration")
else:
    # Fallback to environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")

# Other configuration (non-secret)
MCP_SERVER_STDIO_COMMAND = os.getenv("MCP_SERVER_STDIO_COMMAND", "python -m src.server")
MCP_SERVER_WORKING_DIR = os.getenv("MCP_SERVER_WORKING_DIR", "/Users/kg/IdeaProjects/mcp-server-py")
MCP_SERVER_HTTP_URL = os.getenv("MCP_SERVER_HTTP_URL", "http://localhost:8001")
USE_HTTP_MCP = os.getenv("USE_HTTP_MCP", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_REDIS_SESSIONS = os.getenv("USE_REDIS_SESSIONS", "true").lower() == "true"
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "5"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Session storage (Redis-backed for persistence)
sessions: Dict[str, Dict[str, Any]] = {}  # Fallback for non-Redis mode


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
    cost_info: Optional[Dict[str, Any]] = None  # Include cost tracking information


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class RedisSessionManager:
    """Redis-backed session manager for persistent conversation storage"""
    
    def __init__(self, redis_url: str, password: Optional[str] = None):
        # Build connection URL with password if provided
        if password and "://" in redis_url:
            protocol, rest = redis_url.split("://", 1)
            redis_url = f"{protocol}://:{password}@{rest}"
        
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.session_timeout = SESSION_TIMEOUT
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis_client.ping()
            logger.info(f"✅ Connected to Redis for session storage")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis session manager closed")
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session from Redis or create new one"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        try:
            # Get session data from Redis
            session_data = await self.redis_client.get(f"session:{session_id}")
            
            if session_data:
                session = json.loads(session_data)
                # Update last activity timestamp
                session["last_activity"] = datetime.now().isoformat()
                await self._save_session(session_id, session)
                logger.debug(f"Retrieved session {session_id} from Redis")
                return session
            else:
                # Create new session
                return await self._create_new_session(session_id)
                
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            # Return new session as fallback
            return await self._create_new_session(session_id)
    
    async def _create_new_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session"""
        session = {
            "history": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        await self._save_session(session_id, session)
        logger.info(f"Created new session {session_id}")
        return session
    
    async def _save_session(self, session_id: str, session: Dict[str, Any]):
        """Save session to Redis with TTL"""
        if not self.redis_client:
            return
        
        try:
            session_json = json.dumps(session, default=str)  # Handle datetime serialization
            await self.redis_client.setex(
                f"session:{session_id}", 
                self.session_timeout, 
                session_json
            )
            logger.debug(f"Saved session {session_id} to Redis")
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
    
    async def add_message(self, session_id: str, role: str, message: str, 
                         tools_used: List[str] = None):
        """Add a message to the session history"""
        session = await self.get_session(session_id)
        
        message_entry = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if tools_used:
            message_entry["tools_used"] = tools_used
        
        session["history"].append(message_entry)
        session["last_activity"] = datetime.now().isoformat()
        
        await self._save_session(session_id, session)
        logger.debug(f"Added {role} message to session {session_id}")
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if not self.redis_client:
            return False
        
        try:
            deleted = await self.redis_client.delete(f"session:{session_id}")
            logger.info(f"Cleared session {session_id}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return False
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get only the history of a session"""
        session = await self.get_session(session_id)
        return session.get("history", [])
    
    async def cleanup_expired_sessions(self):
        """Cleanup is handled by Redis TTL, but we can check for manual cleanup"""
        if not self.redis_client:
            return
        
        try:
            # Get all session keys
            session_keys = await self.redis_client.keys("session:*")
            logger.info(f"Found {len(session_keys)} active sessions in Redis")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")


class InMemorySessionManager:
    """Fallback in-memory session manager (for development/testing)"""
    
    def __init__(self):
        self.sessions = sessions  # Use global sessions dict
    
    async def initialize(self):
        logger.info("Using in-memory session storage (development mode)")
    
    async def close(self):
        pass
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "session_id": session_id
            }
        
        self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
        return self.sessions[session_id]
    
    async def add_message(self, session_id: str, role: str, message: str, 
                         tools_used: List[str] = None):
        session = await self.get_session(session_id)
        
        message_entry = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if tools_used:
            message_entry["tools_used"] = tools_used
        
        session["history"].append(message_entry)
    
    async def clear_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        session = await self.get_session(session_id)
        return session.get("history", [])
    
    async def cleanup_expired_sessions(self):
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            last_activity = datetime.fromisoformat(session["last_activity"])
            if (now - last_activity).seconds > SESSION_TIMEOUT:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Orchestrator Service...")
    logger.info(f"Using Gemini model: {DEFAULT_MODEL}")
    if USE_HTTP_MCP:
        logger.info(f"HTTP MCP Server URL: {MCP_SERVER_HTTP_URL}")
    else:
        logger.info(f"Stdio MCP Server command: {MCP_SERVER_STDIO_COMMAND}")
    
    # Initialize session manager
    global session_manager
    try:
        await session_manager.initialize()
        logger.info("✅ Session manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize session manager: {e}")
        if USE_REDIS_SESSIONS:
            logger.warning("Falling back to in-memory session storage")
            session_manager = InMemorySessionManager()
            await session_manager.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Orchestrator Service...")
    
    # Close session manager
    try:
        await session_manager.close()
        logger.info("Session manager closed")
    except Exception as e:
        logger.error(f"Error closing session manager: {e}")
    
    # Close HTTP MCP client
    if USE_HTTP_MCP and hasattr(mcp_client, 'close'):
        logger.info("Closing HTTP MCP client connection...")
        await mcp_client.close()
    
    # Close secure configuration manager
    if config_manager:
        await close_secrets()
        logger.info("Closed secure configuration manager")


# Initialize FastAPI app
app = FastAPI(
    title="LLM Orchestrator",
    description="Gemini 2.5 Flash integration with MCP tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with security hardening
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Add security middleware
if SECURE_CONFIG_AVAILABLE:
    setup_security_middleware(app, rate_limit_calls=500, rate_limit_period=60)

# Add circuit breaker middleware
if RESILIENT_HTTP_AVAILABLE:
    try:
        from pyairtable_common.middleware import add_circuit_breaker_middleware, SERVICE_CONFIGS
        add_circuit_breaker_middleware(app, default_config=SERVICE_CONFIGS["llm_service"])
        logger.info("✅ Circuit breaker middleware enabled")
    except ImportError as e:
        logger.warning(f"⚠️ Circuit breaker middleware not available: {e}")


class MCPHttpClient:
    """HTTP client for communicating with MCP server (eliminates subprocess overhead)"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = None  # Will be initialized lazily
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_expiry: Optional[datetime] = None
    
    async def _get_client(self):
        """Get or create resilient HTTP client"""
        if self.client is None:
            if RESILIENT_HTTP_AVAILABLE:
                self.client = await get_mcp_client(self.base_url)
                logger.info("✅ Using resilient HTTP client with circuit breaker protection")
            else:
                # Fallback to basic httpx client
                self.client = httpx.AsyncClient(
                    timeout=30.0,
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10
                    )
                )
                logger.warning("⚠️ Using basic HTTP client (no circuit breaker protection)")
        return self.client
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server via HTTP"""
        now = datetime.now()
        if self._tools_cache and self._cache_expiry and now < self._cache_expiry:
            logger.info("Using cached tools")
            return self._tools_cache
        
        try:
            logger.info(f"Fetching tools from MCP server at {self.base_url}")
            client = await self._get_client()
            
            if RESILIENT_HTTP_AVAILABLE:
                response = await client.get("tools")
            else:
                response = await client.get(f"{self.base_url}/tools")
                response.raise_for_status()
            
            data = response.json()
            tools = data.get("tools", [])
            
            # Cache tools for 5 minutes
            self._tools_cache = tools
            self._cache_expiry = now + timedelta(minutes=5)
            
            logger.info(f"Retrieved {len(tools)} tools from HTTP MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Error getting tools from HTTP MCP server: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via HTTP MCP server (no subprocess overhead!)"""
        try:
            logger.info(f"Calling tool {tool_name} via HTTP")
            client = await self._get_client()
            
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            
            if RESILIENT_HTTP_AVAILABLE:
                response = await client.post("tools/call", json=payload)
            else:
                response = await client.post(f"{self.base_url}/tools/call", json=payload)
                response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success", True):
                logger.error(f"Tool execution failed: {data.get('error')}")
                return {"error": data.get("error", "Unknown error")}
            
            # Extract text content from MCP TextContent format
            result = data.get("result", [])
            if result and isinstance(result, list) and result[0].get("type") == "text":
                text_content = result[0].get("text", "")
                try:
                    # Try to parse as JSON if possible
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    return {"content": text_content}
            
            logger.info(f"Tool {tool_name} executed successfully via HTTP")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} via HTTP: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            if RESILIENT_HTTP_AVAILABLE:
                await self.client.close()
            else:
                await self.client.aclose()


class MCPClient:
    """Legacy client for communicating with MCP server via stdio"""
    
    def __init__(self, command: str, working_dir: str):
        self.command = command.split()
        self.working_dir = working_dir
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_expiry: Optional[datetime] = None
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server"""
        now = datetime.now()
        if self._tools_cache and self._cache_expiry and now < self._cache_expiry:
            return self._tools_cache
        
        try:
            # Create MCP client request for tools
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            # Execute MCP server
            process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.working_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send request
            request_data = json.dumps(request) + "\n"
            stdout, stderr = await process.communicate(request_data.encode())
            
            if process.returncode != 0:
                logger.error(f"MCP server error: {stderr.decode()}")
                return []
            
            # Parse response
            response = json.loads(stdout.decode().strip())
            tools = response.get("result", {}).get("tools", [])
            
            # Cache tools for 5 minutes
            self._tools_cache = tools
            self._cache_expiry = now + timedelta(minutes=5)
            
            logger.info(f"Retrieved {len(tools)} tools from MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Error getting tools from MCP server: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via MCP server"""
        try:
            # Create MCP client request for tool execution
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Execute MCP server
            process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.working_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send request
            request_data = json.dumps(request) + "\n"
            stdout, stderr = await process.communicate(request_data.encode())
            
            if process.returncode != 0:
                logger.error(f"MCP tool execution error: {stderr.decode()}")
                return {"error": f"Tool execution failed: {stderr.decode()}"}
            
            # Parse response
            response = json.loads(stdout.decode().strip())
            result = response.get("result", {})
            
            logger.info(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}


# Initialize MCP client (HTTP mode for better performance)
if USE_HTTP_MCP:
    logger.info(f"Using HTTP MCP client connecting to {MCP_SERVER_HTTP_URL}")
    mcp_client = MCPHttpClient(MCP_SERVER_HTTP_URL)
else:
    logger.info("Using legacy stdio MCP client")
    mcp_client = MCPClient(MCP_SERVER_STDIO_COMMAND, MCP_SERVER_WORKING_DIR)

# Initialize hybrid session manager (PostgreSQL primary, Redis fallback)
try:
    from hybrid_session_manager import HybridSessionManager
    
    session_manager = HybridSessionManager(
        redis_url=REDIS_URL,
        redis_password=REDIS_PASSWORD,
        session_timeout=SESSION_TIMEOUT,
        use_postgres=True,  # Try PostgreSQL first
        use_redis_fallback=USE_REDIS_SESSIONS  # Fall back to Redis if enabled
    )
    logger.info("Using hybrid session manager (PostgreSQL → Redis → Memory)")
except ImportError:
    # Fallback to original managers if hybrid not available
    if USE_REDIS_SESSIONS:
        logger.info(f"Using Redis session manager connecting to {REDIS_URL}")
        session_manager = RedisSessionManager(REDIS_URL, REDIS_PASSWORD)
    else:
        logger.info("Using in-memory session manager (development mode)")
        session_manager = InMemorySessionManager()


# Legacy functions replaced by session manager
# These are kept for backwards compatibility but should be deprecated

async def get_session_legacy(session_id: str) -> Dict[str, Any]:
    """Get or create session (legacy wrapper)"""
    return await session_manager.get_session(session_id)

async def cleanup_expired_sessions():
    """Remove expired sessions (legacy wrapper)"""
    await session_manager.cleanup_expired_sessions()


# Add endpoint to test function calling
@app.get("/function-calling/status")
async def function_calling_status():
    """Get status of function calling implementation"""
    try:
        from gemini_function_calling import GeminiFunctionCallHandler
        
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


# Cost tracking and budget management endpoints
@app.get("/cost-tracking/status")
async def cost_tracking_status():
    """Get cost tracking system status"""
    if not COST_TRACKING_AVAILABLE or not cost_tracker:
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


@app.get("/sessions/{session_id}/cost-summary")
async def get_session_cost_summary(session_id: str):
    """Get cost summary for a specific session"""
    if not COST_TRACKING_AVAILABLE or not cost_tracker:
        return {"cost_tracking": "disabled"}
    
    return await cost_tracker.get_session_cost_summary(session_id)


@app.post("/sessions/{session_id}/budget")
async def set_session_budget(session_id: str, budget_data: Dict[str, Any]):
    """Set budget limit for a session"""
    if not COST_TRACKING_AVAILABLE or not cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracking not available")
    
    budget_limit = budget_data.get("limit")
    if not budget_limit or not isinstance(budget_limit, (int, float)):
        raise HTTPException(status_code=400, detail="Budget limit must be a number")
    
    if budget_limit <= 0:
        raise HTTPException(status_code=400, detail="Budget limit must be positive")
    
    cost_tracker.set_session_budget(session_id, float(budget_limit))
    
    return {
        "session_id": session_id,
        "budget_limit": str(budget_limit),
        "status": "budget_set",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/sessions/{session_id}/budget")
async def get_session_budget_status(session_id: str, user_id: Optional[str] = None):
    """Get budget status for a session"""
    if not COST_TRACKING_AVAILABLE or not cost_tracker:
        return {"cost_tracking": "disabled"}
    
    return cost_tracker.get_budget_status(session_id, user_id)


@app.post("/users/{user_id}/budget")
async def set_user_budget(user_id: str, budget_data: Dict[str, Any]):
    """Set budget limit for a user"""
    if not COST_TRACKING_AVAILABLE or not cost_tracker:
        raise HTTPException(status_code=503, detail="Cost tracking not available")
    
    budget_limit = budget_data.get("limit")
    if not budget_limit or not isinstance(budget_limit, (int, float)):
        raise HTTPException(status_code=400, detail="Budget limit must be a number")
    
    if budget_limit <= 0:
        raise HTTPException(status_code=400, detail="Budget limit must be positive")
    
    cost_tracker.set_user_budget(user_id, float(budget_limit))
    
    return {
        "user_id": user_id,
        "budget_limit": str(budget_limit),
        "status": "budget_set",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/cost-tracking/analytics")
async def get_cost_analytics(days: int = 7):
    """Get cost analytics for the specified number of days"""
    if not COST_TRACKING_AVAILABLE:
        return {"cost_tracking": "disabled"}
    
    try:
        from pyairtable_common.database import get_async_session
        from pyairtable_common.database.models import ApiUsageLog
        from sqlalchemy import select, func
        from datetime import datetime, timedelta, timezone
        
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


@app.get("/health")
async def health_check():
    """Health check endpoint with session manager status"""
    base_status = {"status": "healthy", "service": "llm-orchestrator"}
    
    # Add session manager health info if hybrid manager
    if hasattr(session_manager, 'health_check'):
        try:
            session_health = await session_manager.health_check()
            base_status["session_manager"] = session_health
        except Exception as e:
            base_status["session_manager"] = {"error": str(e)}
    else:
        base_status["session_manager"] = {"type": "legacy", "status": "unknown"}
    
    return base_status


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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with native Gemini function calling and budget enforcement"""
    await cleanup_expired_sessions()
    session = await session_manager.get_session(request.session_id)
    
    # Pre-request budget checking to prevent expensive calls
    if COST_TRACKING_AVAILABLE and cost_tracker:
        try:
            budget_check = await cost_tracker.check_budget_before_request(
                session_id=request.session_id,
                user_id=session.get("user_id"),
                model_name=DEFAULT_MODEL,
                input_text=request.message,
                estimated_output_tokens=1500  # Conservative estimate for chat responses
            )
            
            if not budget_check.get("allowed", True):
                budget_info = budget_check.get("budget_check", {})
                limits_exceeded = budget_info.get("limits_exceeded", [])
                
                error_msg = "Request blocked - budget limit would be exceeded. "
                for limit in limits_exceeded:
                    error_msg += f"{limit['type'].title()} budget: estimated ${budget_check.get('estimated_cost', '0.00')} would exceed ${limit['limit']}. "
                
                # Log the prevention
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
    
    try:
        # Import the function calling module
        from gemini_function_calling import GeminiFunctionCallHandler, create_gemini_model_with_tools
        
        # Configure generation with thinking budget
        thinking_budget = request.thinking_budget or THINKING_BUDGET
        generation_config = genai.types.GenerationConfig(
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=thinking_budget
            ),
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Prepare system instruction
        system_instruction = f"""You are an AI assistant that can help users interact with their Airtable data. 
You have access to Airtable tools for various operations like listing tables, getting records, creating/updating/deleting records, and searching.

When users ask about Airtable data, use the appropriate tools to help them. Always provide clear explanations of what you're doing and what the results mean.

Current conversation context:
- Session ID: {request.session_id}
- Base ID: {request.base_id or 'Not specified - ask user for their Airtable base ID if needed'}

If the user hasn't provided a base ID and you need one for Airtable operations, ask them to provide it.
"""
        
        # Create Gemini model with function calling
        model = create_gemini_model_with_tools(
            model_name=DEFAULT_MODEL,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # Create function call handler
        function_handler = GeminiFunctionCallHandler(mcp_client)
        
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
        
        # Use native function calling
        start_time = datetime.now()
        result = await function_handler.chat_with_functions(
            model=model,
            messages=messages,
            system_instruction=system_instruction
        )
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response_text = result.get("response", "I apologize, but I couldn't process your request.")
        tools_used = result.get("function_calls_made", [])
        thinking_process = f"Used native function calling with {len(tools_used)} tool calls. Budget: {thinking_budget}"
        
        # If no tools were available, add a note
        if not result.get("tools_available", False):
            thinking_process += " (No tools available - using direct chat)"
        
        # Track cost and usage
        cost_info = {"cost_tracking": "disabled", "cost": "0.00"}
        if COST_TRACKING_AVAILABLE and cost_tracker:
            try:
                cost_info = await cost_tracker.track_api_call(
                    session_id=request.session_id,
                    user_id=session.get("user_id"),  # Get from session if available
                    model_name=DEFAULT_MODEL,
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
        
        # Update session history with cost information
        await session_manager.add_message(
            request.session_id, 
            "user", 
            request.message,
            token_count=int(cost_info.get("cost_data", {}).get("input_tokens", len(request.message.split()) * 1.3)),
            cost=cost_info.get("cost_data", {}).get("input_cost", "0.00")
        )
        await session_manager.add_message(
            request.session_id, 
            "assistant", 
            response_text, 
            tools_used=tools_used,
            token_count=int(cost_info.get("cost_data", {}).get("output_tokens", len(response_text.split()) * 1.3)),
            cost=cost_info.get("cost_data", {}).get("total_cost", "0.00"),
            model_used=DEFAULT_MODEL,
            thinking_process=thinking_process,
            response_time_ms=response_time_ms
        )
        
        # Create response with cost information
        chat_response = ChatResponse(
            response=response_text,
            thinking_process=thinking_process,
            tools_used=tools_used,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Add comprehensive cost info to response if tracking is enabled
        if COST_TRACKING_AVAILABLE and cost_info.get("cost_tracking") in ["success", "budget_exceeded"]:
            cost_data = cost_info.get("cost_data", {})
            chat_response.cost_info = {
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
                "model": DEFAULT_MODEL,
                "response_time_ms": response_time_ms,
                "budget_warnings": cost_info.get("budget_warnings", []),
                "budget_status": cost_info.get("budget_status")
            }
        elif COST_TRACKING_AVAILABLE and cost_info.get("cost_tracking") == "error":
            chat_response.cost_info = {
                "tracking_status": "error",
                "error": cost_info.get("error", "Unknown cost tracking error"),
                "total_cost": "0.00",
                "model": DEFAULT_MODEL,
                "response_time_ms": response_time_ms
            }
        
        return chat_response
        
    except ImportError:
        # Fallback to legacy keyword-based tool detection
        logger.warning("Native function calling not available, falling back to keyword matching")
        return await chat_legacy_keyword_matching(request, session)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def chat_legacy_keyword_matching(request: ChatRequest, session: Dict[str, Any]) -> ChatResponse:
    """Legacy chat endpoint using keyword matching (fallback)"""
    try:
        # Get available tools
        available_tools = await mcp_client.get_tools()
        
        # Configure generation with thinking budget
        thinking_budget = request.thinking_budget or THINKING_BUDGET
        generation_config = genai.types.GenerationConfig(
            thinking_config=genai.types.ThinkingConfig(
                thinking_budget=thinking_budget
            ),
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        # Prepare system prompt with tools
        system_prompt = f"""You are an AI assistant that can help users interact with their Airtable data. 
You have access to the following tools for Airtable operations:

{json.dumps(available_tools, indent=2)}

When a user asks to work with Airtable data, analyze their request and use the appropriate tools.
Always provide clear explanations of what you're doing and what the results mean.

Current conversation context:
- Session ID: {request.session_id}
- Base ID: {request.base_id or 'Not specified - ask user for their Airtable base ID'}

Previous conversation history:
{json.dumps(session['history'][-10:], indent=2) if session['history'] else 'No previous messages'}
"""
        
        # Initialize Gemini model
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        
        # Determine if tools are needed
        tools_used = []
        user_message = request.message
        
        # Simple tool detection (legacy approach)
        needs_tools = any(keyword in user_message.lower() for keyword in [
            "table", "record", "create", "update", "delete", "search", "metadata", "airtable"
        ])
        
        if needs_tools and available_tools:
            response_text = await handle_tool_request_legacy(user_message, request.base_id, available_tools)
            thinking_process = f"Used legacy keyword matching. Budget: {thinking_budget}"
        else:
            # Regular chat without tools
            response = model.generate_content(user_message)
            response_text = response.text
            thinking_process = getattr(response, 'thinking', None) or f"Direct chat response. Budget: {thinking_budget}"
        
        # Update session history using session manager
        await session_manager.add_message(request.session_id, "user", user_message)
        await session_manager.add_message(request.session_id, "assistant", response_text, tools_used)
        
        return ChatResponse(
            response=response_text,
            thinking_process=thinking_process,
            tools_used=tools_used,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in legacy chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_tool_request_legacy(message: str, base_id: Optional[str], available_tools: List[Dict[str, Any]]) -> str:
    """Handle requests that need tool calling (legacy keyword matching)"""
    message_lower = message.lower()
    
    if not base_id:
        return "I need your Airtable base ID to help you. Please provide your base ID (it looks like 'appXXXXXXXXXXXXXX')."
    
    # Simple pattern matching for demo
    if "metadata table" in message_lower or "describe all tables" in message_lower:
        # Use create_metadata_table tool
        result = await mcp_client.call_tool("create_metadata_table", {"base_id": base_id})
        if "error" in result:
            return f"Error creating metadata table: {result['error']}"
        
        return f"I've analyzed your Airtable base and created metadata! Here's what I found:\n\n{json.dumps(result, indent=2)}"
    
    elif "list tables" in message_lower:
        # Use list_tables tool
        result = await mcp_client.call_tool("list_tables", {"base_id": base_id})
        if "error" in result:
            return f"Error listing tables: {result['error']}"
        
        return f"Here are the tables in your base:\n\n{json.dumps(result, indent=2)}"
    
    else:
        return f"I understand you want to work with Airtable data, but I need more specific instructions. I can help you:\n\n- List tables in your base\n- Create a metadata table describing all tables\n- Get records from specific tables\n- Create, update, or delete records\n\nWhat would you like to do?"


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


@app.get("/health/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    try:
        if RESILIENT_HTTP_AVAILABLE:
            from pyairtable_common.resilience import circuit_breaker_registry
            stats = await circuit_breaker_registry.get_all_stats()
            return stats
        else:
            return {
                "circuit_breakers": {},
                "total_breakers": 0,
                "message": "Circuit breaker monitoring not available",
                "generated_at": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving circuit breaker status")


@app.get("/health/services")
async def get_services_health():
    """Get health status of all dependent services"""
    try:
        if RESILIENT_HTTP_AVAILABLE:
            from pyairtable_common.http import service_registry
            health_checks = await service_registry.health_check_all()
            return health_checks
        else:
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


# Include budget management API
try:
    from budget_api import budget_router
    app.include_router(budget_router)
    logger.info("✅ Budget management API endpoints enabled")
except ImportError as e:
    logger.warning(f"⚠️ Budget API not available: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)))