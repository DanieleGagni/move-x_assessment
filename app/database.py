import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration from environment variable
# Default works for local development, docker-compose overrides it
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/bikedb")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    echo=False  # Set to True for SQL query logging during development
)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

# Dependency function for FastAPI endpoints
def get_db():
    """
    Creates a new database session for each request
    and closes it when the request is finished
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database tables
def init_db():
    """
    Create all tables defined in models.py
    Call this on application startup
    """
    from app.models import BikeRental  # Import here to avoid circular imports
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")