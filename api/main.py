"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import single_hop, multi_hop, health, tasks

app = FastAPI(
    title="RAG Testcase Generator",
    description="API for generating RAG testcases.",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url=None  # 禁用 ReDoc
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(single_hop.router, prefix="/single-hop", tags=["single-hop"])
app.include_router(multi_hop.router, prefix="/multi-hop", tags=["multi-hop"])
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])

@app.get("/")
async def root():
    return {
        "message": "RAG Testcase Generator API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    import logging
    from dotenv import load_dotenv

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv('single_hop.env')
    logger.info("Environment variables loaded from single_hop.env")

    # Server configuration
    HOST = "0.0.0.0"
    PORT = 10500

    logger.info(f"Starting RAG Testcase Generator API on {HOST}:{PORT}")
    logger.info(f"API Documentation: http://{HOST}:{PORT}/docs")

    # Run server
    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        timeout_graceful_shutdown=5,
        log_level="info",
        access_log=True
    )
