"""
LLM Orchestrator Service
Integrates Gemini 2.5 Flash with MCP tools for Airtable operations
"""

import os
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_SERVER_STDIO_COMMAND = os.getenv("MCP_SERVER_STDIO_COMMAND", "python -m src.server")
MCP_SERVER_WORKING_DIR = os.getenv("MCP_SERVER_WORKING_DIR", "/Users/kg/IdeaProjects/mcp-server-py")
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "5"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}


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


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Orchestrator Service...")
    logger.info(f"Using Gemini model: {DEFAULT_MODEL}")
    logger.info(f"MCP Server command: {MCP_SERVER_STDIO_COMMAND}")
    yield
    # Shutdown
    logger.info("Shutting down LLM Orchestrator Service...")


# Initialize FastAPI app
app = FastAPI(
    title="LLM Orchestrator",
    description="Gemini 2.5 Flash integration with MCP tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MCPClient:
    """Client for communicating with MCP server via stdio"""
    
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


# Initialize MCP client
mcp_client = MCPClient(MCP_SERVER_STDIO_COMMAND, MCP_SERVER_WORKING_DIR)


def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
    
    sessions[session_id]["last_activity"] = datetime.now()
    return sessions[session_id]


def cleanup_expired_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    expired = []
    
    for session_id, session in sessions.items():
        if (now - session["last_activity"]).seconds > SESSION_TIMEOUT:
            expired.append(session_id)
    
    for session_id in expired:
        del sessions[session_id]
        logger.info(f"Cleaned up expired session: {session_id}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "llm-orchestrator"}


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
    """Main chat endpoint with LLM and tool integration"""
    cleanup_expired_sessions()
    session = get_session(request.session_id)
    
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
        
        # Simple tool detection (in production, let Gemini decide)
        needs_tools = any(keyword in user_message.lower() for keyword in [
            "table", "record", "create", "update", "delete", "search", "metadata", "airtable"
        ])
        
        if needs_tools and available_tools:
            # For now, we'll handle tool calling manually
            # In a full implementation, you'd use Gemini's function calling
            response_text = await handle_tool_request(user_message, request.base_id, available_tools)
            thinking_process = f"Analyzed request and determined tools were needed. Budget: {thinking_budget}"
        else:
            # Regular chat without tools
            response = model.generate_content(user_message)
            response_text = response.text
            thinking_process = getattr(response, 'thinking', None)
        
        # Update session history
        session["history"].append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat()
        })
        session["history"].append({
            "role": "assistant", 
            "message": response_text,
            "tools_used": tools_used,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response_text,
            thinking_process=thinking_process,
            tools_used=tools_used,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_tool_request(message: str, base_id: Optional[str], available_tools: List[Dict[str, Any]]) -> str:
    """Handle requests that need tool calling"""
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": sessions[session_id]["history"]}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8003)))