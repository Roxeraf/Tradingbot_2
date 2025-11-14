"""
FastAPI Application
Main API application with all routers
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import bot_control, trading, strategies

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Bot API",
    description="REST API for controlling and monitoring the crypto trading bot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(bot_control.router)
app.include_router(trading.router)
app.include_router(strategies.router)


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Crypto Trading Bot API")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Crypto Trading Bot API")


@app.get("/")
async def root():
    """
    Root endpoint

    Returns basic API information
    """
    return {
        "name": "Crypto Trading Bot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }
