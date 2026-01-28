"""Configuration settings for RLM Engine."""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os


def convert_database_url(url: str, async_driver: bool = True) -> str:
    """Convert database URL to proper format for SQLAlchemy."""
    # Render provides postgres:// but SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    # For async, use asyncpg driver
    if async_driver and "postgresql://" in url and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    return url


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database - will be validated to add asyncpg driver
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rlm_engine"
    
    @field_validator("database_url", mode="before")
    @classmethod
    def convert_db_url(cls, v: str) -> str:
        """Convert DATABASE_URL to async format."""
        if v:
            return convert_database_url(v, async_driver=True)
        return "postgresql+asyncpg://postgres:postgres@localhost:5432/rlm_engine"
    
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
