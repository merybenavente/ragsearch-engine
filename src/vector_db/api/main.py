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
    logger.info("Vector Database API starting up", version="0.1.0")
    
    # Check for required environment variables
    if not os.environ.get("COHERE_API_KEY"):
        logger.warning(
            "COHERE_API_KEY not found in environment variables. "
            "Embedding operations will fail at runtime. "
            "Please set COHERE_API_KEY to use the vector database."
        )
    
    yield
    # Shutdown
    logger.info("Vector Database API shutting down")

app = FastAPI(
    title="Vector Database API",
    description="A FastAPI backend for vector database operations with document indexing and similarity search",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vector_db.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"message": "Vector Database API is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    logger.info("Detailed health check endpoint accessed")
    return {
        "status": "healthy",
        "service": "vector-db",
        "version": "0.1.0",
        "endpoints": {
            "libraries": "/api/v1/libraries",
            "documents": "/api/v1/libraries/{library_id}/documents",
            "search": "/api/v1/libraries/{library_id}/search"
        }
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
