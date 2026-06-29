"""FastAPI server for the AI Agent."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import SimpleAIAgent
from utils import test_mimoe_connection, call_mimoe_inference

# Load environment variables
load_dotenv()

app = FastAPI(title="mimOE AI Agent", version="1.0.0")

# Initialize default agent
agent = SimpleAIAgent()


class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/mimoe/status")
async def check_mimoe_status():
    """Check connection to mimOE endpoint."""
    status = test_mimoe_connection()
    return status


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Send a message to the AI agent.
    
    Args:
        request: QueryRequest with 'query' field
        
    Returns:
        QueryResponse with the agent's response
    """
    try:
        response = agent.think(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat/reset")
async def reset_chat():
    """Reset conversation history."""
    agent.reset_history()
    return {"message": "Conversation history cleared"}


@app.get("/info")
async def get_info():
    """Get information about the agent."""
    return {
        "name": "mimOE Local AI Agent",
        "version": "1.0.0",
        "description": "AI Agent using local mimOE inference endpoint",
        "endpoints": {
            "POST /chat": "Send message to agent",
            "POST /chat/reset": "Reset conversation",
            "GET /health": "Health check",
            "GET /mimoe/status": "Check mimOE connection",
            "GET /info": "This endpoint"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", 8001))
    print(f"Starting AI Agent on port {port}...")
    print(f"API documentation: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
