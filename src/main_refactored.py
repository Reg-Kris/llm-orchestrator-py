"""
LLM Orchestrator Service - Refactored with PyAirtableService Base Class
Integrates Gemini 2.5 Flash with MCP tools for Airtable operations
"""

import os
import logging
import sys
from typing import Dict, Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel

# Add pyairtable-common to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../pyairtable-common'))

from pyairtable_common.service import PyAirtableService, ServiceConfig, create_service

# Import modular components
from chat import ChatRequest, ChatResponse
from config import load_configuration, cleanup_configuration
from app_factory import initialize_components, cleanup_components
from endpoints import (
    function_calling_status_handler, 
    cost_tracking_status_handler,
    health_check_handler,
    cost_analytics_handler,
    circuit_breaker_status_handler,
    services_health_handler
)

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class LLMOrchestratorService(PyAirtableService):
    """
    LLM Orchestrator service extending PyAirtableService base class.
    """
    
    def __init__(self):
        # Load configuration
        self.app_config, self.config_manager = load_configuration()
        
        # Initialize service configuration
        config = ServiceConfig(
            title="LLM Orchestrator",
            description="Gemini 2.5 Flash integration with MCP tools - Modular Architecture",
            version="2.0.0",
            service_name="llm-orchestrator",
            port=int(os.getenv("PORT", 8003)),
            api_key=os.getenv("API_KEY"),
            startup_tasks=[self._initialize_components],
            shutdown_tasks=[self._cleanup_components],
            health_check_dependencies=[self._check_components_health]
        )
        
        super().__init__(config)
        
        # Component storage
        self.session_manager: Optional[Any] = None
        self.mcp_client: Optional[Any] = None
        self.chat_handler: Optional[Any] = None
        self.cost_tracker: Optional[Any] = None
        
        # Setup routes
        self._setup_llm_routes()
    
    async def _initialize_components(self) -> None:
        """Initialize LLM orchestrator components."""
        try:
            self.session_manager, self.mcp_client, self.chat_handler, self.cost_tracker = \
                await initialize_components(self.app_config)
            self.logger.info("✅ LLM orchestrator components initialized successfully")
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize components: {e}")
            raise
    
    async def _cleanup_components(self) -> None:
        """Cleanup LLM orchestrator components."""
        try:
            await cleanup_components(self.session_manager, self.mcp_client, self.config_manager)
            await cleanup_configuration(self.config_manager)
            self.logger.info("✅ LLM orchestrator components cleaned up successfully")
        except Exception as e:
            self.logger.error(f"⚠️ Error during component cleanup: {e}")
    
    async def _check_components_health(self) -> Dict[str, Any]:
        """Check health of internal components."""
        return {
            "name": "internal_components",
            "status": "healthy" if all([
                self.session_manager,
                self.mcp_client, 
                self.chat_handler
            ]) else "unhealthy",
            "components": {
                "session_manager": "ok" if self.session_manager else "missing",
                "mcp_client": "ok" if self.mcp_client else "missing",
                "chat_handler": "ok" if self.chat_handler else "missing",
                "cost_tracker": "ok" if self.cost_tracker else "disabled"
            }
        }
    
    def _setup_llm_routes(self) -> None:
        """Setup LLM orchestrator specific routes."""
        
        # Main API Endpoints
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Main chat endpoint with native Gemini function calling and budget enforcement"""
            if not self.chat_handler:
                raise HTTPException(status_code=503, detail="Chat handler not initialized")
            return await self.chat_handler.handle_chat_request(request)

        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools"""
            if not self.mcp_client:
                raise HTTPException(status_code=503, detail="MCP client not initialized")
            tools = await self.mcp_client.get_tools()
            return {"tools": tools}

        @self.app.post("/execute-tool")
        async def execute_tool(request: ToolExecutionRequest):
            """Execute a specific MCP tool directly"""
            if not self.mcp_client:
                raise HTTPException(status_code=503, detail="MCP client not initialized")
            result = await self.mcp_client.call_tool(request.tool_name, request.arguments)
            return {"result": result, "tool": request.tool_name}

        @self.app.get("/sessions/{session_id}/history")
        async def get_session_history(session_id: str):
            """Get conversation history for a session"""
            if not self.session_manager:
                raise HTTPException(status_code=503, detail="Session manager not initialized")
            try:
                history = await self.session_manager.get_session_history(session_id)
                return {"session_id": session_id, "history": history}
            except Exception as e:
                self.logger.error(f"Error getting session history: {e}")
                raise HTTPException(status_code=500, detail="Error retrieving session history")

        @self.app.delete("/sessions/{session_id}")
        async def clear_session(session_id: str):
            """Clear session history"""
            if not self.session_manager:
                raise HTTPException(status_code=503, detail="Session manager not initialized")
            try:
                cleared = await self.session_manager.clear_session(session_id)
                if cleared:
                    return {"message": f"Session {session_id} cleared"}
                else:
                    raise HTTPException(status_code=404, detail="Session not found")
            except Exception as e:
                self.logger.error(f"Error clearing session: {e}")
                raise HTTPException(status_code=500, detail="Error clearing session")

        # Status and Analytics Endpoints
        @self.app.get("/function-calling/status")
        async def function_calling_status():
            """Get status of function calling implementation"""
            return await function_calling_status_handler(self.mcp_client)

        @self.app.get("/cost-tracking/status")
        async def cost_tracking_status():
            """Get cost tracking system status"""
            return await cost_tracking_status_handler(self.cost_tracker)

        @self.app.get("/cost-tracking/analytics")
        async def get_cost_analytics(days: int = 7):
            """Get cost analytics for the specified number of days"""
            return await cost_analytics_handler(days)

        @self.app.get("/sessions/{session_id}/cost-summary")
        async def get_session_cost_summary(session_id: str):
            """Get cost summary for a specific session"""
            if not self.cost_tracker or not self.cost_tracker.enabled:
                return {"cost_tracking": "disabled"}
            return await self.cost_tracker.get_session_cost_summary(session_id)

        @self.app.get("/health/circuit-breakers")
        async def get_circuit_breaker_status():
            """Get status of all circuit breakers"""
            return await circuit_breaker_status_handler()

        @self.app.get("/health/services")
        async def get_services_health():
            """Get health status of all dependent services"""
            return await services_health_handler()
        
        # Include budget management API
        try:
            from budget_api import budget_router
            self.app.include_router(budget_router)
            self.logger.info("✅ Budget management API endpoints enabled")
        except ImportError as e:
            self.logger.warning(f"⚠️ Budget API not available: {e}")
    
    async def health_check(self) -> Optional[Dict[str, Any]]:
        """Custom health check for LLM orchestrator."""
        if not self.session_manager or not self.cost_tracker:
            return {"custom_health": "components_not_ready"}
        
        # Delegate to existing health check handler
        try:
            health_result = await health_check_handler(self.session_manager, self.cost_tracker)
            return {"custom_health": health_result}
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"custom_health": "failed", "error": str(e)}


def create_llm_orchestrator_service() -> LLMOrchestratorService:
    """Factory function to create LLM orchestrator service."""
    return LLMOrchestratorService()


if __name__ == "__main__":
    service = create_llm_orchestrator_service()
    service.run()