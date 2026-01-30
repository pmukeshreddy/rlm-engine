# RLM Engine

Recursive reasoning orchestration layer on foundation models. LLM generates Python code to process large contexts, executes in sandboxed REPL, spawns child agents as needed.

**Core idea:** Instead of cramming everything into one prompt, let the LLM write code that strategically queries itself. No model training required.

## How It Works

```
User Query + Large Context (500K+ chars)
            ↓
    LLM generates Python code
    (sees context metadata, not full content)
            ↓
    Sandboxed REPL execution
            ↓
    Code calls llm_query() → spawns child agents
    Code calls FINAL(result) → returns answer
            ↓
    Execution tree with full cost/token tracking
```

The agent receives context metadata (size, hash, type) and writes code to process it. Available in the REPL:

- `context` - full context string
- `llm_query(prompt)` - spawn child agent with any prompt (can include sliced context)
- `FINAL(result)` - return final answer
- `memory` - persistent dict across runs

Example generated code:
```python
chunk_size = 50000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

results = []
for i, chunk in enumerate(chunks):
    result = llm_query(f"Extract key facts from chunk {i+1}/{len(chunks)}:\n{chunk}")
    results.append(result)

summary = llm_query(f"Combine these facts:\n" + "\n---\n".join(results))
FINAL(summary)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (React)                    │
│  - Execution tree visualization                          │
│  - Real-time streaming via SSE                           │
│  - Cost/token tracking dashboard                         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                      │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Sessions  │  │  Executions │  │    Stats    │      │
│  │   + Memory  │  │  + Stream   │  │  + Costs    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│                            │                             │
│                            ▼                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │                  LettaAgent                      │    │
│  │  - Generates code via LLM                        │    │
│  │  - Executes in sandboxed REPL                    │    │
│  │  - Tracks child agent calls                      │    │
│  │  - Supports recursive depth (configurable)       │    │
│  └─────────────────────────────────────────────────┘    │
│                            │                             │
│                            ▼                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │              REPLExecutor (Sandbox)              │    │
│  │  - Restricted builtins                           │    │
│  │  - Blocks dangerous imports (os, subprocess)     │    │
│  │  - Async child calls via event loop bridge       │    │
│  │  - Timeout enforcement                           │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   OpenAI / Anthropic    │
              └─────────────────────────┘
```

## Features

- **Recursive child agents** - children can spawn children (up to configurable depth)
- **Streaming execution** - real-time SSE updates as agents run
- **Execution tree visualization** - see the full call graph
- **Cost tracking** - per-execution and aggregate token/cost metrics
- **Session memory** - persistent key-value store across executions
- **Multi-provider** - OpenAI and Anthropic support

## Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add API keys
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Environment variables:
```
DATABASE_URL=postgresql://...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=gpt-4-turbo-preview
MAX_RECURSION_DEPTH=10
EXECUTION_TIMEOUT=300
```

## API

```bash
# Create session with context
POST /api/sessions
{"name": "my-doc", "context": "...500K chars..."}

# Execute query
POST /api/execute
{"session_id": "...", "user_query": "Summarize the key points"}

# Stream execution (SSE)
POST /api/execute/stream
{"session_id": "...", "user_query": "..."}

# Get execution tree
GET /api/executions/{id}/tree

# Usage stats
GET /api/stats
```

## Config

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_CONTEXT_SIZE` | 500,000 | Max context characters |
| `DEFAULT_CHUNK_SIZE` | 50,000 | Chunk size for splitting |
| `MAX_RECURSION_DEPTH` | 10 | Max child agent depth |
| `EXECUTION_TIMEOUT` | 300 | Timeout in seconds |

## Deploy

Configured for Render (see `render.yaml`). Frontend on Vercel/Netlify works too.

## Why This Approach

Traditional LLM usage: stuff everything in one prompt, hope it fits.

This approach: let the LLM decide how to process the context. It can:
- Chunk and map-reduce
- Search for relevant sections first
- Build up answers iteratively
- Use different strategies for different query types

The orchestration happens at inference time, no training required. Swap foundation models instantly.

## Related Work

Similar philosophy to test-time compute scaling and iterative refinement approaches. The LLM writes its own processing logic rather than following a fixed pipeline.

## License

MIT
