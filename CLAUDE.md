# LLM Orchestrator - Claude Context

## 🎯 Service Purpose (✅ PHASE 1 COMPLETE - MODULAR ARCHITECTURE)
This is the **conversational AI brain** of PyAirtable - orchestrating interactions between users, Gemini 2.5 Flash, and MCP tools. It manages chat sessions, executes tool calls, and provides the natural language interface to Airtable operations.

## 🏗️ Current State

### Deployment Status
- **Environment**: ✅ Local Kubernetes (Minikube)
- **Services Running**: ✅ 7 out of 9 services operational
- **Database Analysis**: ✅ Airtable test database analyzed (34 tables, 539 fields)
- **Metadata Tool**: ✅ Table analysis tool executed successfully

### Service Status
- **LLM Integration**: ✅ Gemini 2.5 Flash working with native function calling
- **Tool Calling**: ✅ Native Gemini function calling (replaced keyword matching)
- **Session Management**: ✅ PostgreSQL + Redis hybrid persistence (survives restarts!)
- **MCP Communication**: ✅ Resilient HTTP client with circuit breaker protection
- **Cost Tracking**: ✅ Complete cost tracking with real token counting, budgets and pre-request validation
- **Circuit Breakers**: ✅ Protection against cascading failures
- **Streaming**: ❌ Not implemented
- **Advanced Features**: ✅ Security headers, rate limiting, correlation IDs

### Recent Fixes Applied
- ✅ Pydantic v2 compatibility issues resolved
- ✅ Gemini ThinkingConfig configuration fixed
- ✅ SQLAlchemy metadata handling updated
- ✅ Service deployment to Kubernetes completed

## 🧠 Core Functionality

### Chat Processing Flow
```python
User Message → Tool Detection → MCP Tool Selection → Tool Execution → Response Generation
     ↓              ↓                    ↓                  ↓              ↓
  Session ID    Keywords?          Available Tools    Subprocess      Gemini 2.5
```

### Current Implementation Issues
1. ✅ ~~**Simple keyword matching** for tool detection~~ FIXED with native function calling
2. ✅ ~~**New subprocess** for every MCP call~~ FIXED with HTTP client
3. ✅ ~~**No conversation memory** persistence~~ FIXED with PostgreSQL + Redis
4. **No streaming responses** (full generation only)
5. ✅ ~~**No cost tracking** or token limits~~ FIXED with comprehensive budget management

## 🚀 Recent Improvements

1. **Circuit Breaker Protection** ✅ (NEW!)
   ```python
   # Resilient HTTP client with circuit breaker protection
   from pyairtable_common.http import get_mcp_client
   mcp_client = await get_mcp_client("http://mcp-server:8001")
   # Automatic failure detection and recovery!
   ```

2. **Native Function Calling** ✅ (COMPLETED)
   ```python
   # Using Gemini's native function calling instead of keyword matching
   model = genai.GenerativeModel(
       model_name="gemini-2.5-flash",
       tools=[convert_mcp_to_gemini_tools(mcp_tools)]
   )
   ```

3. **PostgreSQL Session Management** ✅ (COMPLETED)
   ```python
   # Hybrid session manager (PostgreSQL primary, Redis fallback)
   session_manager = HybridSessionManager(
       redis_url=REDIS_URL,
       use_postgres=True
   )
   ```

4. **Cost Tracking & Budgets** ✅ (COMPLETED)
   ```python
   # Pre-request budget validation to prevent expensive calls
   budget_check = await cost_tracker.check_budget_before_request(
       session_id=session_id, model_name="gemini-2.5-flash",
       input_text=input_text, estimated_output_tokens=1500
   )
   if not budget_check.get("allowed", True):
       raise HTTPException(status_code=429, detail="Budget limit exceeded")
   
   # Post-request cost tracking with real token counting
   cost_info = await cost_tracker.track_api_call(
       session_id=session_id, model_name="gemini-2.5-flash",
       input_text=input_text, output_text=output_text, success=True
   )
   ```

## 🚀 Remaining Priorities

1. **Streaming Response Support** (HIGH)
   ```python
   # Add streaming responses for better UX
   async def stream_chat_response(request):
       async for chunk in model.generate_content_stream(prompt):
           yield chunk.text
   ```

2. **Advanced Testing** (MEDIUM)
   ```python
   # Comprehensive test suite for all functionality
   # - Chat endpoint tests
   # - Circuit breaker behavior
   # - Cost tracking accuracy
   # - Session management edge cases
   ```

## 🔮 Future Enhancements

### Phase 1 (Next Sprint)
- [x] Native Gemini function calling ✅ COMPLETED
- [x] PostgreSQL+Redis session persistence ✅ COMPLETED  
- [x] Real token counting & cost tracking ✅ COMPLETED
- [ ] Streaming response support

### Phase 2 (Next Month)
- [ ] Multi-model support (Claude, GPT-4)
- [x] Conversation memory with PostgreSQL ✅ COMPLETED
- [x] Cost tracking and budgets ✅ COMPLETED
- [ ] Advanced prompt engineering

### Phase 3 (Future)
- [ ] RAG with Airtable data
- [ ] Custom prompt templates
- [ ] Multi-turn planning
- [ ] Parallel tool execution

## ⚠️ Known Issues
1. **No streaming** - users wait for full response (affects UX)
2. **Limited test coverage** - needs comprehensive testing
3. **No distributed tracing** - harder to debug across services

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
circuit_breaker_state{service}     # Circuit breaker status
service_health_status{service}     # Service health
cost_tracking_total{session}       # Cost tracking
```

## 🔄 Circuit Breaker Monitoring
```python
# Circuit breaker endpoints:
GET /health/circuit-breakers       # All circuit breaker stats
GET /health/services              # Service health checks

# Example response:
{
    "circuit_breakers": {
        "mcp-server-tools": {
            "state": "closed",
            "stats": {
                "total_requests": 1250,
                "success_rate": 0.96,
                "consecutive_failures": 0,
                "avg_response_time_ms": 145
            }
        }
    }
}
```

## 💰 Cost Management & Budget Control

### Real Token Counting ✅
```python
# Using Gemini SDK for accurate token counting (not estimates!)
token_counts = await cost_calculator.count_tokens_real(
    model_name="gemini-2.5-flash",
    input_text=input_text,
    output_text=output_text,
    thinking_text=thinking_text
)

# Pricing (as of 2025-01-01):
# Input: $0.000075 per 1K tokens
# Output: $0.0003 per 1K tokens  
# Thinking: $0.000075 per 1K tokens
```

### Pre-Request Budget Validation ✅
```python
# Check budget BEFORE making expensive API calls
budget_check = await cost_tracker.check_budget_before_request(
    session_id=session_id,
    user_id=user_id,
    model_name="gemini-2.5-flash", 
    input_text=input_text,
    estimated_output_tokens=1500
)

if not budget_check.get("allowed", True):
    # Block the request - prevents overspend!
    raise HTTPException(status_code=429, detail="Budget limit exceeded")
```

### Database-Backed Budget Management ✅
```python
# Set session budget
POST /budgets/session/{session_id}
{
    "budget_limit": 10.00,
    "alert_threshold": 0.8
}

# Set user budget (monthly)
POST /budgets/user/{user_id}
{
    "budget_limit": 50.00,
    "reset_period": "monthly"
}

# Check budget status
GET /budgets/status/{session_id}?user_id={user_id}
```

### Cost Analytics ✅
```python
# Real-time session cost summary
GET /sessions/{session_id}/cost-summary
{
    "total_calls": 25,
    "total_cost": "2.47",
    "total_tokens": 12450,
    "avg_response_time_ms": 1250
}

# Usage analytics with breakdown
GET /cost-tracking/analytics?days=7
{
    "total_stats": {
        "total_calls": 1205,
        "total_cost": "127.83",
        "success_rate": 96.2
    },
    "daily_breakdown": [...]
}
```

### Comprehensive ChatResponse ✅
```python
# Chat responses now include detailed cost info
{
    "response": "Here are your tables...",
    "cost_info": {
        "tracking_status": "success",
        "total_cost": "0.00234",
        "token_details": {
            "input_tokens": 125,
            "output_tokens": 87,
            "thinking_tokens": 23,
            "counting_method": "real_api"
        },
        "cost_breakdown": {
            "input_cost": "0.00094",
            "output_cost": "0.00261",
            "thinking_cost": "0.00017"
        },
        "budget_warnings": [],
        "model": "gemini-2.5-flash"
    }
}
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