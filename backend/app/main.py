"""Main FastAPI application for RLM Engine."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db, close_db
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="RLM Engine",
    description="""
    Runtime for Letta - A system for processing large contexts with AI agents.
    
    ## Features
    
    - **Large Context Processing**: Handle 500K+ character contexts by storing them
      as variables instead of in the prompt.
    
    - **REPL Execution**: Agents write Python code that executes in a sandboxed
      REPL environment.
    
    - **Child Agent Spawning**: Agents can spawn child agents via `llm_query()` to
      process chunks of the context in parallel.
    
    - **Persistent Memory**: Sessions maintain memory across executions.
    
    - **Execution Tracing**: Full execution tree with costs, timing, and memory diffs.
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RLM Engine",
        "version": "0.1.0",
        "description": "Runtime for Letta - Large context processing with AI agents",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
