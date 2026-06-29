# AI Agent with mimOE Local Inference

A lightweight AI agent that connects to [mimOE Studio](https://developer.mimik.com/) for local, on-device inference using OpenAI-compatible APIs.

## Overview

This project demonstrates:
- Connecting to mimOE Studio's local inference endpoint
- Building a simple conversational AI agent
- Using FastAPI to expose agent capabilities via HTTP
- Managing conversation history for multi-turn interactions

## Prerequisites

1. **mimOE Studio** - Downloaded and running
   - Download from: https://developer.mimik.com/mimOE-studio-early-access-download-v2
   - A local model loaded (e.g., SmolLM2)
   - API endpoint running on `http://localhost:8000/api/v1`

2. **Python 3.8+**

## Setup

1. **Clone/download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` if needed (defaults work if mimOE is running locally):
   ```
   MIMOE_API_URL=http://localhost:8000/api/v1
   MIMOE_MODEL_NAME=smollm2
   AGENT_PORT=8001
   ```

## Usage

### Option 1: FastAPI Server

```bash
python main.py
```

The agent will be available at `http://localhost:8001`

**API Endpoints:**
- `GET /health` - Health check
- `GET /mimoe/status` - Check mimOE connection
- `POST /chat` - Send message to agent
  ```bash
  curl -X POST http://localhost:8001/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the capital of France?"}'
  ```
- `POST /chat/reset` - Clear conversation history
- `GET /docs` - Interactive API documentation (Swagger UI)

### Option 2: Direct Python

```python
from agent import SimpleAIAgent

agent = SimpleAIAgent()
response = agent.think("What is machine learning?")
print(response)
```

## Architecture

### Key Components

1. **`agent.py`** - Core agent logic
   - `SimpleAIAgent`: Main agent class with conversation management
   - Factory functions for specialized agents (task planning, code assistance)
   - Builds prompts with conversation context

2. **`utils.py`** - API utilities
   - `call_mimoe_inference()`: Makes calls to mimOE endpoint
   - `test_mimoe_connection()`: Validates connectivity
   - Environment variable handling

3. **`main.py`** - FastAPI application
   - Exposes agent as HTTP API
   - Manages conversation state
   - Health checks and status endpoints

## Design Choices

- **No Framework Dependencies**: Uses raw HTTP requests instead of LangChain/LlamaIndex for simplicity and clarity
- **Conversation History**: Maintains last 3 exchanges to stay within token limits
- **OpenAI-Compatible API**: Works with any OpenAI-compatible inference endpoint
- **Lightweight**: Minimal dependencies (FastAPI, requests, pydantic)
- **Local-First**: All processing happens on-device via mimOE

## Troubleshooting

**Cannot connect to mimOE endpoint**
- Ensure mimOE Studio is running
- Check that a model is loaded in mimOE
- Verify the endpoint URL in `.env`

**Token limit errors**
- Reduce conversation history length in `agent.py`
- Reduce `max_tokens` in API calls

**Slow responses**
- This is expected for local inference on CPU
- Larger models are slower than SmolLM2

## Next Steps

- Add memory/persistence layer (Redis, database)
- Implement tool/function calling
- Add multiple specialized agents
- Create web UI frontend
- Add streaming responses
- Integrate with external APIs

## License

MIT

## Resources

- [mimOE Documentation](https://developer.mimik.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
