"""Utility functions for mimOE API interactions."""
import requests
import os
from typing import Optional


def get_mimoe_endpoint() -> str:
    """Get the mimOE API endpoint from environment."""
    return os.getenv("MIMOE_API_URL", "http://localhost:8000/api/v1")


def call_mimoe_inference(prompt: str, model: Optional[str] = None, max_tokens: int = 500) -> str:
    """
    Call the mimOE local inference endpoint.
    
    Args:
        prompt: The input prompt for the model
        model: Model name (defaults to env variable)
        max_tokens: Maximum tokens in response
        
    Returns:
        The model's response text
        
    Raises:
        requests.RequestException: If API call fails
    """
    endpoint = get_mimoe_endpoint()
    model_name = model or os.getenv("MIMOE_MODEL_NAME", "smollm2")
    
    # OpenAI-compatible API format
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(
            f"{endpoint}/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise Exception(f"Cannot connect to mimOE endpoint at {endpoint}. Is mimOE Studio running?")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling mimOE API: {str(e)}")


def test_mimoe_connection() -> dict:
    """Test connection to mimOE endpoint."""
    endpoint = get_mimoe_endpoint()
    try:
        response = requests.get(f"{endpoint}/models", timeout=5)
        response.raise_for_status()
        return {"status": "connected", "models": response.json()}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}
