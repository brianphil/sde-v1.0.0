"""SQLAlchemy Database Setup with Async Support.

This module provides:
- Async SQLAlchemy engine and session management
- Database connection pooling
- Base declarative class for ORM models
- Database initialization and health checks
- Session dependency for FastAPI
"""

import os
from typing import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://senga_user:senga_pass@localhost:5432/senga_db")
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "20"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"

# SQLAlchemy Base for ORM models
Base = declarative_base()

# Global engine and session factory (initialized on startup)
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_database_engine(
    database_url: str | None = None,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    echo: bool | None = None,
    pool_pre_ping: bool = True,
) -> AsyncEngine:
    """Create async SQLAlchemy engine with connection pooling.

    Args:
        database_url: Database connection URL (uses env var if None)
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        echo: Echo SQL statements to logs
        pool_pre_ping: Enable connection health checks

    Returns:
        AsyncEngine instance
    """
    url = database_url or DATABASE_URL
    pool_sz = pool_size if pool_size is not None else DATABASE_POOL_SIZE
    max_ovf = max_overflow if max_overflow is not None else DATABASE_MAX_OVERFLOW
    echo_sql = echo if echo is not None else DATABASE_ECHO

    # Use QueuePool for production, NullPool for SQLite
    is_sqlite = "sqlite" in url
    poolclass = NullPool if is_sqlite else QueuePool

    logger.info(f"Creating database engine: {url.split('@')[1] if '@' in url else 'sqlite'}")

    # Build engine kwargs - SQLite doesn't support pool parameters
    engine_kwargs = {
        "echo": echo_sql,
        "poolclass": poolclass,
    }

    if not is_sqlite:
        # Only add pool parameters for non-SQLite databases
        engine_kwargs.update({
            "pool_size": pool_sz,
            "max_overflow": max_ovf,
            "pool_pre_ping": pool_pre_ping,
            "pool_recycle": 3600,
        })

    return create_async_engine(url, **engine_kwargs)


def create_session_factory(engine_instance: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async session factory.

    Args:
        engine_instance: SQLAlchemy async engine

    Returns:
        Session factory
    """
    return async_sessionmaker(
        engine_instance,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual flush control
        autocommit=False,  # Manual commit control
    )


async def init_database() -> None:
    """Initialize database engine and session factory.

    Called on application startup.
    """
    global engine, async_session_factory

    try:
        engine = create_database_engine()
        async_session_factory = create_session_factory(engine)

        # Test connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database() -> None:
    """Close database connections.

    Called on application shutdown.
    """
    global engine

    if engine:
        await engine.dispose()
        logger.info("Database connections closed")


async def create_tables() -> None:
    """Create all database tables.

    WARNING: Only use for development/testing.
    In production, use Alembic migrations.
    """
    global engine

    if not engine:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created")


async def drop_tables() -> None:
    """Drop all database tables.

    WARNING: Destructive operation. Use with caution.
    """
    global engine

    if not engine:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.warning("Database tables dropped")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database sessions.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession instance
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Yields:
        AsyncSession instance
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health() -> dict:
    """Check database connection health.

    Returns:
        dict: Health status with connection info
    """
    if not engine:
        return {
            "status": "unhealthy",
            "error": "Database not initialized"
        }

    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()

        return {
            "status": "healthy",
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Utility functions for testing
async def reset_database() -> None:
    """Reset database (drop and recreate tables).

    WARNING: Only for testing. Destroys all data.
    """
    await drop_tables()
    await create_tables()
    logger.warning("Database reset complete")
