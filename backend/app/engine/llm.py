"""LLM Client for interacting with language models."""
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import tiktoken
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from app.config import settings


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response: Optional[Dict[str, Any]] = None


# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    # OpenAI
    "gpt-4-turbo-preview": (10.0, 30.0),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.5, 1.5),
    # Anthropic
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call."""
    if model not in MODEL_PRICING:
        # Default pricing for unknown models
        input_price, output_price = (10.0, 30.0)
    else:
        input_price, output_price = MODEL_PRICING[model]
    
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class LLMClient:
    """Client for interacting with various LLM providers."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.anthropic_api_key = anthropic_api_key or settings.anthropic_api_key
        
        self._openai_client: Optional[AsyncOpenAI] = None
        self._anthropic_client: Optional[AsyncAnthropic] = None
    
    @property
    def openai(self) -> AsyncOpenAI:
        """Get the OpenAI client."""
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        return self._openai_client
    
    @property
    def anthropic(self) -> AsyncAnthropic:
        """Get the Anthropic client."""
        if self._anthropic_client is None:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            self._anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
        return self._anthropic_client
    
    def _is_anthropic_model(self, model: str) -> bool:
        """Check if a model is an Anthropic model."""
        return model.startswith("claude")
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to settings.default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt (for Anthropic models)
        
        Returns:
            LLMResponse with the completion
        """
        model = model or settings.default_model
        
        if self._is_anthropic_model(model):
            return await self._complete_anthropic(
                messages, model, temperature, max_tokens, system_prompt
            )
        else:
            return await self._complete_openai(
                messages, model, temperature, max_tokens, system_prompt
            )
    
    async def _complete_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> LLMResponse:
        """Complete using OpenAI."""
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        return LLMResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
    
    async def _complete_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> LLMResponse:
        """Complete using Anthropic."""
        response = await self.anthropic.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
        )
        
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        return LLMResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
    
    async def generate_agent_code(
        self,
        user_query: str,
        context_info: Dict[str, Any],
        memory: Dict[str, Any],
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate agent code for processing a query.
        
        The agent receives information about the context (size, type)
        but not the actual content. It generates code to process the context.
        """
        system_prompt = """You are a Letta agent that processes large contexts by writing Python code.

You have access to these variables and functions in your REPL environment:
- `context`: The full context string (may be very large, 500K+ characters)
- `len(context)`: Get the size of the context
- `context[start:end]`: Slice the context to get a portion
- `llm_query(prompt: str) -> str`: Spawn a child agent to answer a question. The prompt can include data from context.
- `FINAL(result: str)`: Call this with your final answer when done. YOU MUST call this to complete.
- `memory`: A dict of persistent memory from previous runs (read-only in code, updated automatically)

IMPORTANT RULES:
1. NEVER try to include the full context in a prompt - it's too large!
2. Use chunking: split context into smaller pieces and process each with llm_query()
3. For summarization/extraction tasks, process chunks and aggregate results
4. For search tasks, scan chunks for relevant sections, then analyze those
5. Always call FINAL(result) at the end with your answer
6. Keep your code simple and readable
7. Handle errors gracefully

Example for processing large context:
```python
# Split into manageable chunks
chunk_size = 50000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

results = []
for i, chunk in enumerate(chunks):
    result = llm_query(f"Extract key facts from this text (chunk {i+1}/{len(chunks)}):\\n{chunk}")
    results.append(result)

# Aggregate results
summary = llm_query(f"Combine these extracted facts into a coherent summary:\\n" + "\\n---\\n".join(results))
FINAL(summary)
```

Write Python code to answer the user's query. Output ONLY the code, no explanations."""

        user_message = f"""Context Information:
- Size: {context_info.get('size', 'unknown')} characters
- Type: {context_info.get('type', 'text')}
- Hash: {context_info.get('hash', 'none')[:16] if context_info.get('hash') else 'none'}...

Memory from previous runs:
{json.dumps(memory, indent=2) if memory else "No previous memory"}

User Query: {user_query}

Generate Python code to answer this query. Remember to call FINAL(result) at the end."""

        return await self.complete(
            messages=[{"role": "user", "content": user_message}],
            model=model,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more consistent code
        )
    
    async def child_agent_query(
        self,
        prompt: str,
        parent_memory: Dict[str, Any],
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Execute a child agent query (called via llm_query in the REPL).
        
        Child agents answer specific questions and can access parent memory.
        """
        system_prompt = """You are a Letta child agent helping to process a large document.

You receive a specific prompt from the parent agent and should provide a direct, helpful answer.
Focus on the specific task given to you. Be concise but thorough.

If you're asked to extract information, provide it in a structured format.
If you're asked to summarize, be comprehensive but concise.
If you're asked to analyze, provide clear insights."""

        user_message = f"""Memory context (from parent agent):
{json.dumps(parent_memory, indent=2) if parent_memory else "No memory context"}

Task: {prompt}"""

        return await self.complete(
            messages=[{"role": "user", "content": user_message}],
            model=model,
            system_prompt=system_prompt,
            temperature=0.5,
        )
