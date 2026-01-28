"""Configuration settings for RLM Engine."""
from pydantic_settings import BaseSettings
from typing import Optional
import os


def get_database_url(async_driver: bool = True) -> str:
    """Get database URL, converting Render's format if needed."""
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rlm_engine")
    
    # Render provides postgres:// but SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    # For async, use asyncpg driver
    if async_driver and "postgresql://" in url and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    return url


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database - computed from DATABASE_URL env var
    database_url: str = get_database_url(async_driver=True)
    database_url_sync: str = get_database_url(async_driver=False)
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4-turbo-preview"
    
    # Engine Settings
    max_context_size: int = 500_000  # Maximum characters for context
    default_chunk_size: int = 50_000  # Default chunk size for splitting
    max_recursion_depth: int = 10  # Maximum depth of child agents
    execution_timeout: int = 300  # Seconds before execution times out
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
