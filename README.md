# llm-orchestrator-py

Gemini 2.5 Flash integration service for tool calling and chat orchestration

## Overview

This microservice orchestrates conversations between users and Gemini 2.5 Flash, with the ability to call MCP tools through the mcp-server-py service. It handles:
- Chat conversations with context management
- Tool discovery and execution
- Gemini 2.5 Flash thinking budget configuration
- Conversation history and session management

## Features

- **Gemini 2.5 Flash Integration**: Latest model with thinking capabilities
- **MCP Tool Integration**: Connects to mcp-server-py for Airtable operations
- **Thinking Budget Control**: Configure LLM reasoning depth
- **Session Management**: Track conversation history
- **Async Operations**: Non-blocking tool execution

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Reg-Kris/llm-orchestrator-py.git
cd llm-orchestrator-py

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the service
uvicorn src.main:app --reload --port 8003
```

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Send message and get LLM response with tool calling
- `GET /tools` - List available MCP tools
- `POST /execute-tool` - Execute specific MCP tool directly
- `GET /sessions/{session_id}/history` - Get conversation history
- `DELETE /sessions/{session_id}` - Clear session history

## Environment Variables

```
GEMINI_API_KEY=your_gemini_api_key
MCP_SERVER_URL=http://localhost:8001
THINKING_BUDGET=5
LOG_LEVEL=INFO
```

## Example Usage

```bash
# Start a chat conversation
curl -X POST http://localhost:8003/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a metadata table describing all tables in my Airtable base",
    "session_id": "user123",
    "base_id": "appXXXXXXXXXXXXXX"
  }'

# List available tools
curl http://localhost:8003/tools
```

## Docker

```bash
# Build image
docker build -t llm-orchestrator-py .

# Run container
docker run -p 8003:8003 --env-file .env llm-orchestrator-py
```