"""
Application factory for LLM Orchestrator
Handles component initialization and lifecycle management
"""

import logging
from typing import Dict, Any, Tuple

from chat import ChatHandler
from session.manager import HybridSessionManager
from mcp.http_client import MCPHttpClient
from mcp.stdio_client import MCPStdioClient
from cost.tracking import CostTrackingManager

logger = logging.getLogger(__name__)


async def initialize_components(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """
    Initialize all application components
    
    Returns:
        Tuple of (session_manager, mcp_client, chat_handler, cost_tracker)
    """
    logger.info("Starting LLM Orchestrator Service...")
    logger.info(f"Using Gemini model: {config['default_model']}")
    
    # Initialize cost tracking
    cost_tracker = CostTrackingManager()
    logger.info(f"Cost tracking: {'enabled' if cost_tracker.enabled else 'disabled'}")
    
    # Initialize MCP client
    if config["use_http_mcp"]:
        logger.info(f"HTTP MCP Server URL: {config['mcp_server_http_url']}")
        mcp_client = MCPHttpClient(config["mcp_server_http_url"])
    else:
        logger.info(f"Stdio MCP Server command: {config['mcp_server_stdio_command']}")
        mcp_client = MCPStdioClient(
            config["mcp_server_stdio_command"], 
            config["mcp_server_working_dir"]
        )
    
    # Initialize session manager
    session_manager = HybridSessionManager(
        redis_url=config["redis_url"],
        redis_password=config["redis_password"],
        session_timeout=config["session_timeout"],
        use_postgres=True,
        use_redis_fallback=config["use_redis_sessions"]
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
        default_model=config["default_model"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        thinking_budget=config["thinking_budget"]
    )
    
    logger.info("✅ LLM Orchestrator Service started successfully")
    
    return session_manager, mcp_client, chat_handler, cost_tracker


async def cleanup_components(session_manager, mcp_client, config_manager):
    """Cleanup all application components"""
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
    
    # Cleanup configuration is handled in config.py