"""
Configuration management for LLM Orchestrator
Handles secure configuration loading and environment variables
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Secure configuration imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

try:
    from pyairtable_common.config import initialize_secrets, get_secret, close_secrets
    from pyairtable_common.middleware import setup_security_middleware
    SECURE_CONFIG_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"⚠️ Secure configuration not available: {e}")
    SECURE_CONFIG_AVAILABLE = False

# Resilient HTTP client
try:
    from pyairtable_common.middleware import add_circuit_breaker_middleware, SERVICE_CONFIGS
    CIRCUIT_BREAKER_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"⚠️ Circuit breaker middleware not available: {e}")
    CIRCUIT_BREAKER_AVAILABLE = False


def load_configuration() -> Tuple[Dict[str, Any], Optional[Any]]:
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


def setup_middleware(app, config: Dict[str, Any]):
    """Setup middleware for the FastAPI app"""
    import uuid
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response
    
    # Custom middleware for distributed tracing
    class DistributedTracingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            # Extract trace ID from incoming request or generate new one
            trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
            
            # Add trace ID to request state for use in handlers
            request.state.trace_id = trace_id
            
            # Log request start with trace ID
            logger.info(f"[TRACE:{trace_id}] LLM Orchestrator request: {request.method} {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            
            # Log request completion
            logger.info(f"[TRACE:{trace_id}] LLM Orchestrator response: {response.status_code}")
            
            return response
    
    # Add distributed tracing middleware
    app.add_middleware(DistributedTracingMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["cors_origins"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Trace-ID"],
    )
    
    # Add security middleware
    if SECURE_CONFIG_AVAILABLE:
        setup_security_middleware(app, rate_limit_calls=500, rate_limit_period=60)
    
    # Add circuit breaker middleware
    if CIRCUIT_BREAKER_AVAILABLE:
        add_circuit_breaker_middleware(app, default_config=SERVICE_CONFIGS["llm_service"])
        logger.info("✅ Circuit breaker middleware enabled")


async def cleanup_configuration(config_manager):
    """Cleanup configuration resources"""
    if config_manager:
        await close_secrets()
        logger.info("Closed secure configuration manager")