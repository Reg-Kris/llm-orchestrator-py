"""
Stdio client for communicating with MCP server (legacy support)
Uses subprocess communication with the MCP server
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MCPStdioClient:
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
            logger.info("Using cached tools")
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
            
            logger.info(f"Retrieved {len(tools)} tools from stdio MCP server")
            return tools
            
        except Exception as e:
            logger.error(f"Error getting tools from MCP server: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via MCP server"""
        try:
            logger.info(f"Calling tool {tool_name} via stdio")
            
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
            
            logger.info(f"Tool {tool_name} executed successfully via stdio")
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the stdio client (no-op for stdio)"""
        logger.info("Stdio MCP client closed")