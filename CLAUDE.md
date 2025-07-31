# LLM Orchestrator - Claude Context

## 🎯 Service Purpose
This is the **conversational AI brain** of PyAirtable - orchestrating interactions between users, Gemini 2.5 Flash, and MCP tools. It manages chat sessions, executes tool calls, and provides the natural language interface to Airtable operations.

## 🏗️ Current State
- **LLM Integration**: ✅ Gemini 2.5 Flash working
- **Tool Calling**: ⚠️ Manual detection (not using native function calling)
- **Session Management**: ⚠️ In-memory only (lost on restart)
- **MCP Communication**: ❌ Subprocess spawning (inefficient)
- **Streaming**: ❌ Not implemented
- **Cost Tracking**: ❌ No token usage monitoring

## 🧠 Core Functionality

### Chat Processing Flow
```python
User Message → Tool Detection → MCP Tool Selection → Tool Execution → Response Generation
     ↓              ↓                    ↓                  ↓              ↓
  Session ID    Keywords?          Available Tools    Subprocess      Gemini 2.5
```

### Current Implementation Issues
1. **Simple keyword matching** for tool detection
2. **New subprocess** for every MCP call
3. **No conversation memory** persistence
4. **No streaming responses** (full generation only)
5. **No cost tracking** or token limits

## 🚀 Immediate Priorities

1. **Fix MCP Connection** (CRITICAL)
   ```python
   # Current: Subprocess per call
   process = await asyncio.create_subprocess_exec(...)
   
   # Needed: Persistent connection
   self.mcp_connection = await mcp.connect(...)
   ```

2. **Implement Native Function Calling** (HIGH)
   ```python
   # Use Gemini's native function calling
   model = genai.GenerativeModel(
       model_name="gemini-2.5-flash",
       tools=[convert_mcp_to_gemini_tools(mcp_tools)]
   )
   ```

3. **Add Redis Session Storage** (HIGH)
   ```python
   # Replace in-memory with Redis
   async def get_session(session_id: str):
       data = await redis.get(f"session:{session_id}")
       return json.loads(data) if data else create_new_session()
   ```

## 🔮 Future Enhancements

### Phase 1 (Next Sprint)
- [ ] Native Gemini function calling
- [ ] Redis-backed session persistence
- [ ] Streaming response support
- [ ] Token usage tracking

### Phase 2 (Next Month)
- [ ] Multi-model support (Claude, GPT-4)
- [ ] Conversation memory with PostgreSQL
- [ ] Cost tracking and budgets
- [ ] Advanced prompt engineering

### Phase 3 (Future)
- [ ] RAG with Airtable data
- [ ] Custom prompt templates
- [ ] Multi-turn planning
- [ ] Parallel tool execution

## ⚠️ Known Issues
1. **Session data loss** on service restart
2. **No streaming** - users wait for full response
3. **Basic tool detection** - misses complex requests
4. **No cost control** - unlimited token usage

## 🧪 Testing Strategy
```python
# Priority test coverage:
- Chat endpoint with various prompts
- Tool detection accuracy tests
- Session management edge cases
- Error handling for LLM failures
- Cost calculation accuracy
```

## 🔧 Technical Details
- **LLM**: Gemini 2.5 Flash (gemini-2.5-flash)
- **SDK**: google-generativeai 0.8.3
- **Framework**: FastAPI with async
- **Session Store**: In-memory dict (needs Redis)

## 📊 Performance Targets
- **Response Time**: < 2s for simple queries
- **Streaming Latency**: < 500ms to first token
- **Session Load**: < 50ms from Redis
- **Concurrent Users**: 50+ simultaneous chats

## 🤝 Service Dependencies
```
Frontend → API Gateway → LLM Orchestrator → MCP Server → Airtable Gateway
                              ↓
                         Gemini API
```

## 💡 Development Tips
1. Test prompts in Gemini AI Studio first
2. Monitor token usage - Gemini 2.5 Flash has limits
3. Use thinking budget wisely (affects cost)
4. Cache common tool patterns

## 🚨 Critical Configuration
```python
# Required environment variables:
GEMINI_API_KEY=your_api_key_here
MCP_SERVER_STDIO_COMMAND="python -m src.server"
THINKING_BUDGET=5  # Control reasoning depth
MAX_TOKENS=4000   # Response length limit
SESSION_TIMEOUT=3600  # 1 hour
```

## 🔒 Security Considerations
- **Session Isolation**: Each user has separate context
- **Prompt Injection**: Sanitize user messages
- **Token Limits**: Prevent abuse with limits
- **API Key Security**: Never log API keys

## 📈 Monitoring Metrics
```python
# Key metrics to track:
llm_requests_total{model}           # LLM API calls
llm_tokens_used{type}              # Input/output tokens
llm_response_time_seconds          # Generation latency
active_sessions_total              # Concurrent users
tool_detection_accuracy            # Success rate
```

## 💰 Cost Management
```python
# Gemini 2.5 Flash Pricing (as of 2025):
# Input: $0.0001 per 1K tokens
# Output: $0.0003 per 1K tokens

# Add cost tracking:
cost = (input_tokens * 0.0001 + output_tokens * 0.0003) / 1000

# Implement budgets:
if session.total_cost > MAX_BUDGET:
    return "Budget exceeded, please contact admin"
```

## 🎯 Conversation Patterns
```python
# Common patterns to handle:
1. "List all tables" → list_tables tool
2. "Create a metadata table" → create_metadata_table tool
3. "Show me records where..." → search_records tool
4. "Update the record..." → update_record tool

# Complex patterns needing improvement:
- Multi-step operations
- Conditional logic
- Bulk operations
- Data analysis requests
```

Remember: This service is the **user's gateway** to AI-powered Airtable operations. Every improvement here directly enhances the user experience!