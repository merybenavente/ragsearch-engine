import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import vector_db
from ..infrastructure.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("RAGSearch Engine starting up", version="0.1.0")

    # Check for required environment variables
    if not os.environ.get("COHERE_API_KEY"):
        logger.warning(
            "COHERE_API_KEY not found in environment variables. "
            "Embedding operations will fail at runtime. "
            "Please set COHERE_API_KEY to use RAGSearch Engine."
        )

    yield
    # Shutdown
    logger.info("RAGSearch Engine shutting down")


app = FastAPI(
    title="RAGSearch Engine",
    description="A personal project exploring semantic search for RAG applications with multiple vector index implementations. Built with FastAPI, featuring clean architecture and comprehensive testing.",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS from environment variables
# CORS_ORIGINS can be:
# - Comma-separated list: "https://example.com,https://app.example.com"
# - "*" for all origins (development only)
# - Empty/unset defaults to ["*"] for backward compatibility
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env == "*":
    cors_origins = ["*"]
elif cors_origins_env == "":
    cors_origins = ["*"]  # Default to permissive for development
else:
    # Parse comma-separated origins
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(vector_db.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"message": "RAGSearch Engine is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    logger.info("Detailed health check endpoint accessed")
    return {
        "status": "healthy",
        "service": "ragsearch-engine",
        "version": "0.1.0",
        "endpoints": {
            "libraries": "/api/v1/libraries",
            "documents": "/api/v1/libraries/{library_id}/documents",
            "search": "/api/v1/libraries/{library_id}/search",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
