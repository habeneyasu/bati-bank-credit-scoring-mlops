"""
Database connection management with connection pooling and error handling.

Provides singleton pattern for database connection management.
"""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from src.utils.config import settings
from src.utils.logging import get_logger
from src.database.exceptions import (
    DatabaseConnectionError,
    DatabaseError
)

logger = get_logger(__name__)


class DatabaseManager:
    """
    Database connection manager with connection pooling.
    
    Implements singleton pattern to ensure single database connection pool.
    """
    
    _instance: Optional['DatabaseManager'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager (only once due to singleton)."""
        if self._engine is None:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection and session factory."""
        try:
            database_url = self._get_database_url()
            
            logger.info(
                "Initializing database connection",
                extra={
                    "database_host": settings.database_host,
                    "database_name": settings.database_name,
                    "pool_size": settings.database_pool_size
                }
            )
            
            # Create engine with connection pooling
            self._engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=settings.database_echo,  # Log SQL queries if enabled
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "credit_scoring_mlops"
                }
            )
            
            # Create session factory
            self._session_factory = scoped_session(
                sessionmaker(
                    bind=self._engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False
                )
            )
            
            # Test connection
            self._test_connection()
            
            logger.info("Database connection initialized successfully")
            
        except OperationalError as e:
            logger.error(
                "Failed to connect to database",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseConnectionError(
                f"Failed to connect to database: {str(e)}",
                original_error=e
            )
        except Exception as e:
            logger.error(
                "Unexpected error initializing database",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseConnectionError(
                f"Unexpected error initializing database: {str(e)}",
                original_error=e
            )
    
    def _get_database_url(self) -> str:
        """
        Get database connection URL.
        
        Reads from environment variables (.env file) with the following priority:
        1. DATABASE_URL (if set, overrides all other settings)
        2. Individual components (DATABASE_USER, DATABASE_PASSWORD, etc.)
        
        Returns:
            Database connection URL string
            
        Raises:
            DatabaseConnectionError: If required settings are missing
        """
        # Priority 1: Use full connection URL if provided
        if settings.database_url:
            logger.debug("Using DATABASE_URL from environment")
            return settings.database_url
        
        # Priority 2: Build URL from individual components
        if not settings.database_user:
            raise DatabaseConnectionError(
                "DATABASE_USER is required. Set it in .env file or DATABASE_URL environment variable."
            )
        
        if not settings.database_name:
            raise DatabaseConnectionError(
                "DATABASE_NAME is required. Set it in .env file or DATABASE_URL environment variable."
            )
        
        # Build connection URL
        if settings.database_password:
            # URL encode password to handle special characters
            from urllib.parse import quote_plus
            encoded_password = quote_plus(settings.database_password)
            url = (
                f"postgresql://{settings.database_user}:{encoded_password}"
                f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
            )
        else:
            # No password (local development with trust authentication)
            url = (
                f"postgresql://{settings.database_user}"
                f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
            )
        
        logger.debug(
            "Built database URL from components",
            extra={
                "host": settings.database_host,
                "port": settings.database_port,
                "database": settings.database_name,
                "user": settings.database_user,
                "has_password": bool(settings.database_password)
            }
        )
        
        return url
    
    def _test_connection(self):
        """Test database connection."""
        try:
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.debug("Database connection test successful")
        except Exception as e:
            logger.error(
                "Database connection test failed",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseConnectionError(
                f"Database connection test failed: {str(e)}",
                original_error=e
            )
    
    @property
    def engine(self) -> Engine:
        """
        Get database engine.
        
        Returns:
            SQLAlchemy engine instance
        """
        if self._engine is None:
            self._initialize_connection()
        return self._engine
    
    @property
    def session_factory(self) -> scoped_session:
        """
        Get session factory.
        
        Returns:
            SQLAlchemy session factory
        """
        if self._session_factory is None:
            self._initialize_connection()
        return self._session_factory
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session context manager.
        
        Yields:
            Database session
            
        Raises:
            DatabaseError: If session creation or commit fails
        """
        session = self.session_factory()
        try:
            logger.debug("Database session created")
            yield session
            session.commit()
            logger.debug("Database session committed")
        except IntegrityError as e:
            session.rollback()
            logger.error(
                "Database integrity error, transaction rolled back",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Database integrity error: {str(e)}",
                original_error=e
            )
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                "Database error, transaction rolled back",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Database error: {str(e)}",
                original_error=e
            )
        except Exception as e:
            session.rollback()
            logger.error(
                "Unexpected error in database session",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Unexpected database error: {str(e)}",
                original_error=e
            )
        finally:
            session.close()
            logger.debug("Database session closed")
    
    def create_tables(self):
        """
        Create all database tables.
        
        Raises:
            DatabaseError: If table creation fails
        """
        try:
            from src.database.models import Base
            logger.info("Creating database tables")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(
                "Failed to create database tables",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to create database tables: {str(e)}",
                original_error=e
            )
    
    def drop_tables(self):
        """
        Drop all database tables (use with caution!).
        
        Raises:
            DatabaseError: If table dropping fails
        """
        try:
            from src.database.models import Base
            logger.warning("Dropping all database tables")
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(
                "Failed to drop database tables",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(
                f"Failed to drop database tables: {str(e)}",
                original_error=e
            )
    
    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
            self._engine = None
            self._session_factory = None


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session context manager (convenience function).
    
    Yields:
        Database session
        
    Example:
        with get_db_session() as session:
            user = session.query(User).filter_by(username="admin").first()
    """
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session
