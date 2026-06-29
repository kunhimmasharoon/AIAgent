"""Simple AI Agent that interfaces with mimOE local inference."""
from typing import Optional, List
from utils import call_mimoe_inference


class SimpleAIAgent:
    """A lightweight AI agent using mimOE local inference."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            system_prompt: Optional system prompt to guide behavior
        """
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.conversation_history: List[dict] = []
    
    def _build_prompt(self, user_input: str) -> str:
        """Build the full prompt including system context and history."""
        prompt_parts = [self.system_prompt]
        
        # Add conversation context (last 3 exchanges to stay within token limits)
        recent_history = self.conversation_history[-6:]  # 3 exchanges = 6 messages
        for msg in recent_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        
        # Add current user input
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def think(self, user_input: str) -> str:
        """
        Process user input and generate a response using mimOE.
        
        Args:
            user_input: The user's message
            
        Returns:
            The agent's response
        """
        # Build full prompt
        full_prompt = self._build_prompt(user_input)
        
        # Call mimOE inference
        response = call_mimoe_inference(full_prompt, max_tokens=500)
        
        # Clean up response (remove "Assistant:" if included)
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        
        # Store in history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def create_task_agent():
    """Create a specialized agent for task planning and execution."""
    system_prompt = """You are a helpful task planning assistant. When given a goal:
1. Break it down into concrete steps
2. Identify any dependencies
3. Suggest an execution order
Be concise and practical."""
    return SimpleAIAgent(system_prompt)


def create_code_agent():
    """Create a specialized agent for code-related tasks."""
    system_prompt = """You are a code assistant. Help with:
1. Code explanations
2. Bug fixes
3. Design suggestions
Keep responses concise and practical."""
    return SimpleAIAgent(system_prompt)
