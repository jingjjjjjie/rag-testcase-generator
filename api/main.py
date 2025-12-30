"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from api.routers import single_hop, multi_hop, health, tasks, config
from api.config_manager import config_manager

app = FastAPI(
    title="RAG Testcase Generator",
    description="API for generating RAG testcases.",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url=None  # 禁用 ReDoc
)


def custom_openapi():
    """Customize OpenAPI schema to show current config values in examples"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Get current config values
    current_config = config_manager.get_config()

    # Update the PUT /config endpoint schema to show current values as example
    if "/config/" in openapi_schema["paths"]:
        put_endpoint = openapi_schema["paths"]["/config/"].get("put")
        if put_endpoint and "requestBody" in put_endpoint:
            # Set example values to current config
            put_endpoint["requestBody"]["content"]["application/json"]["example"] = current_config.model_dump()

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

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
app.include_router(config.router, prefix="/config", tags=["config"])
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
