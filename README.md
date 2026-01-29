# RLM Engine - Runtime for Letta

A powerful system for processing large contexts (500K+ tokens) with AI agents. The RLM Engine stores large contexts as variables instead of in prompts, allowing agents to write code that processes them in chunks via spawned child agents.

## Architecture

```
USER QUERY + LONG CONTEXT (500K+ tokens)
            │
            ▼
┌─────────────────────────────────────────┐
│           RLM ENGINE                     │
│                                          │
│  1. Context too big for prompt?          │
│     → Store as variable, not in prompt   │
│                                          │
│  2. Initialize REPL environment          │
│     context = "<500K token document>"    │
│     llm_query = function to spawn child  │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│         ROOT LETTA AGENT                 │
│                                          │
│  Sees: "context has 500K chars"          │
│  NOT the actual content                  │
│                                          │
│  Writes code:                            │
│  ```python                               │
│  chunks = split(context, 50000)          │
│  results = []                            │
│  for chunk in chunks:                    │
│      r = llm_query(f"Extract facts from: │
│          {chunk}")                       │
│      results.append(r)                   │
│  FINAL(aggregate(results))               │
│  ```                                     │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│         REPL EXECUTOR                    │
│                                          │
│  Runs the code                           │
│  When hits llm_query() →                 │
│      Spawns CHILD LETTA AGENT            │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│       CHILD LETTA AGENTS (x10)           │
│                                          │
│  - Receives 50K chunk (fits in context)  │
│  - Inherits parent's memory (optional)   │
│  - Processes, returns result             │
│  - Learnings merge back to parent memory │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│         POSTGRES                         │
│                                          │
│  Stores:                                 │
│  - Full execution trace (tree structure) │
│  - Each agent's code + output            │
│  - Memory snapshots at each step         │
│  - Costs per call                        │
└─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│         REACT DASHBOARD                  │
│                                          │
│  - View execution tree                   │
│  - See what each agent did               │
│  - Memory diff between steps             │
│  - Total cost breakdown                  │
└─────────────────────────────────────────┘
```

## Features

- **Large Context Processing**: Handle 500K+ character contexts by storing them as variables instead of in prompts
- **Agent Code Generation**: Root agent writes Python code to process the context
- **Child Agent Spawning**: `llm_query()` function spawns child agents for processing chunks
- **Sandboxed REPL**: Executes agent-generated code safely with restricted access
- **Persistent Sessions**: Memory persists across executions within a session
- **Full Execution Tracing**: Tree-structured traces with timing, tokens, and costs
- **React Dashboard**: Beautiful UI for viewing executions, sessions, and stats
- **Multi-model Support**: Works with OpenAI (GPT-4) and Anthropic (Claude) models

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
cd "RLM Runtime for Letta"
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. Start the services:
```bash
docker-compose up -d
```

4. Access the dashboard at http://localhost:3000

### Manual Setup

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Start PostgreSQL (or use Docker)
docker run -d --name rlm-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rlm_engine \
  -p 5432:5432 \
  postgres:16-alpine

# Run the server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## API Usage

### Create a Session with Context

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Analysis",
    "context": "<your large text here - 500K+ chars>"
  }'
```

### Run an Execution

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Summarize the key findings in this document",
    "session_id": "<session-id>"
  }'
```

### Direct Execution (without session)

```bash
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Extract all dates mentioned",
    "context": "<your large text>"
  }'
```

## How It Works

### 1. Context Storage
When you provide a large context, it's stored as a variable in the REPL environment. The root agent only sees metadata (size, hash) - not the actual content.

### 2. Code Generation
The root agent generates Python code to process the context. It has access to:
- `context`: The full text (as a variable)
- `llm_query(prompt)`: Spawn a child agent
- `FINAL(result)`: Return the final answer
- Standard Python builtins for string manipulation

### 3. REPL Execution
The generated code runs in a sandboxed environment. When `llm_query()` is called, it spawns a child agent with the given prompt.

### 4. Child Agents
Child agents receive prompts that include chunks of the context. They process these chunks and return results to the parent.

### 5. Result Aggregation
The root agent's code aggregates child results and calls `FINAL()` with the answer.

## Example Agent Code

The root agent might generate code like this:

```python
# Split context into manageable chunks
chunk_size = 50000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

# Process each chunk
summaries = []
for i, chunk in enumerate(chunks):
    summary = llm_query(f"""
        Summarize the key points from this section (chunk {i+1}/{len(chunks)}):
        
        {chunk}
    """)
    summaries.append(summary)

# Combine summaries
final_summary = llm_query(f"""
    Combine these section summaries into a coherent final summary:
    
    {chr(10).join(summaries)}
""")

FINAL(final_summary)
```

## Dashboard Features

- **Execution Tree**: Visual tree showing root and child agent relationships
- **Code View**: See the code generated by each agent
- **Memory Diff**: View changes to persistent memory between steps
- **Cost Tracking**: Total and per-node token usage and costs
- **Session Management**: Create, view, and manage sessions

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_MODEL` | Default LLM model | `gpt-4-turbo-preview` |
| `MAX_CONTEXT_SIZE` | Maximum context size | 500000 |
| `DEFAULT_CHUNK_SIZE` | Default chunk size | 50000 |
| `MAX_RECURSION_DEPTH` | Max child agent depth | 10 |
| `EXECUTION_TIMEOUT` | Execution timeout (seconds) | 300 |

## Supported Models

### OpenAI
- gpt-4-turbo-preview
- gpt-4o
- gpt-4o-mini
- gpt-4
- gpt-3.5-turbo

### Anthropic
- claude-3-opus-20240229
- claude-3-5-sonnet-20241022
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

## Project Structure

```
RLM Runtime for Letta/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── engine/        # Core RLM engine
│   │   │   ├── agent.py   # Letta agent implementation
│   │   │   ├── llm.py     # LLM client
│   │   │   └── repl.py    # REPL executor
│   │   ├── models/        # SQLAlchemy models
│   │   └── repositories/  # Database operations
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/           # API client
│   │   ├── components/    # React components
│   │   └── pages/         # Page components
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```


## License

MIT
