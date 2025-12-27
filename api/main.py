"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import single_hop, multi_hop, health

app = FastAPI(
    title="RAG Testcase Generator",
    description="API for generating RAG testcases.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(single_hop.router, prefix="/api/v1/single-hop", tags=["single-hop"])
app.include_router(multi_hop.router, prefix="/api/v1/multi-hop", tags=["multi-hop"])

@app.get("/")
async def root():
    return {
        "message": "RAG Testcase Generator API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=10500, timeout_graceful_shutdown=5)
