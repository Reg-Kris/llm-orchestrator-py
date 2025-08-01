"""
HTTP client for communicating with MCP server
Eliminates subprocess overhead and provides better performance
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)


class MCPHttpClient:
    """HTTP client for communicating with MCP server (eliminates subprocess overhead)"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = None  # Will be initialized lazily
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_expiry: Optional[datetime] = None
        
        # Check for resilient HTTP client availability
        self.resilient_http_available = False
        try:
            # Add path to pyairtable-common
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../pyairtable-common'))
            from pyairtable_common.http import get_mcp_client
            self.resilient_http_available = True
        except ImportError:
            logger.warning("⚠️ Resilient HTTP client not available, using basic httpx client")
    
    async def _get_client(self):
        """Get or create resilient HTTP client"""
        if self.client is None:
            if self.resilient_http_available:
                from pyairtable_common.http import get_mcp_client
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
            
            if self.resilient_http_available:
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
            
            if self.resilient_http_available:
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
            if self.resilient_http_available:
                await self.client.close()
            else:
                await self.client.aclose()
            logger.info("HTTP MCP client closed")