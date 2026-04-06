"""Base model for SQLAlchemy ORM."""
from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import DeclarativeBase
from uuid import uuid4


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())
