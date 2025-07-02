import os
import ssl
import logging
from fastapi import FastAPI
from middleware import add_middlewares

# Import the APIRouter from routes.py
from routes import router as api_router

# Disable SSL verification (temporary workaround)
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Transcription & Summary Service",
    description="API for uploading videos, getting transcriptions, and generating summaries",
    version="1.0.0"
)

# Add CORS and rate limiter middleware
add_middlewares(app)

# Register all API routes from routes.py
app.include_router(api_router)

@app.get("/api/health")
async def health_check():
    from datetime import datetime
    return {"status": "healthy", "timestamp": str(datetime.now())}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Video Transcription & Summary Service with Authentication")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )