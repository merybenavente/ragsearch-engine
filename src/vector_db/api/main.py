from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import vector_db
from ..infrastructure.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Vector Database API",
    description="A FastAPI backend for vector database operations with document indexing and similarity search",
    version="0.1.0"
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
app.include_router(vector_db.router, prefix="/api")


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
            "libraries": "/api/libraries",
            "documents": "/api/libraries/{library_id}/documents",
            "search": "/api/libraries/{library_id}/search"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Vector Database API starting up", version="0.1.0")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Vector Database API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
