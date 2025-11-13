"""
FastAPI application for Crypto Trading Bot
Provides REST API and WebSocket endpoints
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import asyncio

from src.config.settings import Settings
from src.api.routers import trading, strategies, backtest, settings_router, websocket
from src.api.models.responses import HealthResponse, ErrorResponse
from src.monitoring.logger import setup_logger, get_logger

# Initialize settings and logger
settings = Settings()
logger = setup_logger(settings.LOG_LEVEL, settings.LOG_TO_FILE)

# Create FastAPI app
app = FastAPI(
    title="Crypto Trading Bot API",
    description="REST API for cryptocurrency trading bot",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["Strategies"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
app.include_router(settings_router.router, prefix="/api/settings", tags=["Settings"])
app.include_router(websocket.router, prefix="/api/ws", tags=["WebSocket"])


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 80)
    logger.info("Starting Crypto Trading Bot API")
    logger.info("=" * 80)
    logger.info(f"API Host: {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Environment: {'Testnet' if settings.TESTNET else 'Production'}")
    logger.info(f"Exchange: {settings.EXCHANGE_NAME}")
    logger.info(f"Strategy: {settings.STRATEGY_NAME}")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Crypto Trading Bot API")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Enable auto-reload for development
        log_level=settings.LOG_LEVEL.lower()
    )
