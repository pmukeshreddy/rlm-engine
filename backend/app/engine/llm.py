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


# Context window sizes in tokens
MODEL_CONTEXT_WINDOWS = {
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16384,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-sonnet-4-6": 200000,
}

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
    "claude-sonnet-4-6": (3.0, 15.0),
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call."""
    if model not in MODEL_PRICING:
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


def _get_rlm_system_prompt(context_length: int, max_chunk_chars: int, model: str) -> str:
    """
    Build the RLM system prompt following the paper (Appendix C).

    Key differences from naive map-reduce:
    - Iterative: LLM is queried in a loop, can probe context first, then process
    - Persistent REPL: variables survive across iterations
    - Only truncated stdout metadata is shown back to the LLM
    - LLM decides its own chunking strategy based on what it observes
    """
    # Calculate sub-LLM capacity in chars
    sub_llm_chars = max_chunk_chars

    return f"""You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a text string with {context_length} total characters.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around {sub_llm_chars:,} chars) inside your REPL environment.
3. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.
4. A 'MAX_CHUNK_CHARS' variable ({max_chunk_chars:,}) indicating the maximum characters per sub-LLM call.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context.

Use variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around {sub_llm_chars:,} characters in their context window, so don't be afraid to put a lot of context into them. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want to peek at the first 10000 characters of the context:
```repl
chunk = context[:10000]
print(f"First 10000 characters: {{chunk}}")
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer:
```repl
query = "What is the main theme of the book?"
chunk_size = MAX_CHUNK_CHARS - 500
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
buffers = []
for i, chunk in enumerate(chunks):
    finding = llm_query(f"Extract information relevant to this question: {{query}}\\nText:\\n{{chunk}}")
    if "not relevant" not in finding.lower():
        buffers.append(finding)
    print(f"Processed chunk {{i+1}}/{{len(chunks)}}")
```

Then in a next step, synthesize the findings:
```repl
final_answer = llm_query(f"Based on these findings, give a short direct answer to: {{query}}\\n\\nFindings:\\n" + "\\n".join(buffers))
print(final_answer)
```

Then return: FINAL(final_answer)

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

IMPORTANT: Your final answer should be SHORT and DIRECT -- typically 1-2 sentences or a few words. Use llm_query() to distill findings into a brief answer before calling FINAL().

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer."""


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
        if self._openai_client is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        return self._openai_client

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic_client is None:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            self._anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
        return self._anthropic_client

    def _is_anthropic_model(self, model: str) -> bool:
        return model.startswith("claude")

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
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

    async def rlm_iteration(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        One iteration of the RLM loop.

        Sends the full conversation history (system prompt + user query +
        previous code + stdout observations) and gets the next code block
        or final answer from the LLM.

        This is the core of Algorithm 1: hist -> LLM_M(hist) -> code
        """
        model = model or settings.default_model

        # Calculate safe max_tokens
        context_window = MODEL_CONTEXT_WINDOWS.get(model, 16384)
        all_text = system_prompt + " ".join(m["content"] for m in conversation_history)
        input_tokens = count_tokens(all_text, model)
        max_output = min(4096, context_window - input_tokens - 200)
        max_output = max(max_output, 512)

        return await self.complete(
            messages=conversation_history,
            model=model,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=max_output,
        )

    async def generate_agent_code(
        self,
        user_query: str,
        context_info: Dict[str, Any],
        memory: Dict[str, Any],
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Legacy single-shot code generation (used by old interface).
        The new iterative RLM loop uses rlm_iteration() instead.
        """
        system_prompt = """You are a Letta agent that processes large contexts by writing Python code.

You have access to these variables and functions in your REPL environment:
- `context`: The full context string (may be very large, 500K+ characters)
- `len(context)`: Get the size of the context
- `context[start:end]`: Slice the context to get a portion
- `llm_query(prompt: str) -> str`: Spawn a child agent to answer a question. The prompt can include data from context.
- `FINAL(result: str)`: Call this with your final answer when done. YOU MUST call this to complete.
- `memory`: A dict of persistent memory from previous runs (read-only direct access)
- `get_memory(key: str, default=None) -> Any`: Get a value from persistent memory
- `set_memory(key: str, value: Any)`: Store a value in persistent memory for future runs
- `MAX_CHUNK_CHARS`: Maximum characters per chunk that fits in the model's context window. ALWAYS use this for chunk sizing.

IMPORTANT RULES:
1. NEVER try to include the full context in a prompt - it's too large!
2. Use chunking: split context into smaller pieces and process each with llm_query()
3. CRITICAL: Each llm_query() prompt MUST be under MAX_CHUNK_CHARS characters total (including your instructions + the chunk text). Use MAX_CHUNK_CHARS to size your chunks.
4. For QA tasks, scan chunks for relevant sections, then answer from those specific sections
5. For summarization tasks, process chunks and aggregate results
6. Always call FINAL(result) at the end with your answer
7. FINAL(result) MUST be a short, direct answer -- typically 1-2 sentences or a few words. NEVER pass long text to FINAL(). Always use llm_query() to distill findings into a brief answer first.
8. Keep your code simple and readable
9. Handle errors gracefully
10. Use set_memory() to persist useful information for future queries

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
            temperature=0.3,
        )

    async def child_agent_query(
        self,
        prompt: str,
        parent_memory: Dict[str, Any],
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Execute a child agent query (called via llm_query in the REPL).
        """
        model = model or settings.default_model

        system_prompt = """You are a Letta child agent helping to process a large document.

You receive a specific prompt from the parent agent and should provide a direct, helpful answer.
Focus on the specific task given to you. Be concise but thorough.

If you're asked to extract information, provide it in a structured format.
If you're asked to summarize, be comprehensive but concise.
If you're asked to analyze, provide clear insights.
If you're asked whether something is relevant, be honest - say NOT RELEVANT if it truly isn't."""

        user_message = f"""Memory context (from parent agent):
{json.dumps(parent_memory, indent=2) if parent_memory else "No memory context"}

Task: {prompt}"""

        # Cap max_tokens based on model's context window
        context_window = MODEL_CONTEXT_WINDOWS.get(model, 16384)
        input_tokens = count_tokens(system_prompt + user_message, model)
        max_output = min(1024, context_window - input_tokens - 100)
        max_output = max(max_output, 256)

        return await self.complete(
            messages=[{"role": "user", "content": user_message}],
            model=model,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=max_output,
        )
