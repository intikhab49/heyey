"""
CryptoAion AI - Universal Crypto Prediction Engine
Complete FastAPI Server with full API functionality
"""
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from simple_config import settings
from tortoise.contrib.fastapi import register_tortoise
from tortoise import Tortoise
from mangum import Mangum
import logging

# Import all routers
from routes.prediction import router as prediction_router
from routes.coingecko import router as coingecko_router
from routes.yfinance import router as yfinance_router
from routes.auth_routes import router as auth_router
from routes.data_router import router as data_router
from routes.websocket_router import router as websocket_router

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    try:
        logger.info("🚀 Starting CryptoAion AI - Universal Crypto Prediction Engine...")
        
        # Initialize Tortoise ORM
        await Tortoise.init(
            db_url=settings.DATABASE_URL,
            modules={'models': ['models.user']}
        )
        await Tortoise.generate_schemas()
        
        logger.info("✅ Database initialized successfully")
        logger.info("🎯 CryptoAion AI is ready for universal crypto predictions!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("🛑 Shutting down CryptoAion AI...")
        await Tortoise.close_connections()
        logger.info("✅ Database connections closed")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")
        raise

# Create FastAPI app with complete configuration
app = FastAPI(
    title="CryptoAion AI - Universal Crypto Prediction API",
    description="""
    🚀 **Fast, accurate, and universal cryptocurrency price prediction using advanced machine learning**
    
    ## Features:
    - ✅ **Universal Support**: Any crypto symbol (BTC, ETH, DOGE, SHIB, etc.)
    - ✅ **All Timeframes**: 30m, 1h, 4h, 24h predictions
    - ✅ **Rich Metrics**: R², MAPE, confidence scores, technical indicators
    - ✅ **Real-time**: WebSocket streaming predictions
    - ✅ **Optimized**: Expert-tuned hyperparameters for accuracy & speed
    
    ## Quick Start:
    - Get prediction: `GET /api/predict/{symbol}?timeframe=1h`
    - Force retrain: `GET /api/predict/{symbol}?timeframe=1h&force_retrain=true`
    - Historical data: `GET /api/data/historical/{symbol}?timeframe=1h`
    - WebSocket: `WS /ws/{symbol}/{timeframe}`
    """,
    version="2.0.0 - Best Version",
    docs_url="/docs",  # This should fix the 404 on /api/docs
    redoc_url="/redoc",
    lifespan=lifespan
)

# Create Mangum handler for serverless deployment
handler = Mangum(app)

# CORS - Public API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers - Complete Public API Suite
app.include_router(prediction_router, prefix="/api/predict", tags=["🎯 Predictions"])
app.include_router(data_router, prefix="/api/data", tags=["📊 Market Data"]) 
app.include_router(websocket_router, tags=["⚡ Real-time Streaming"])
app.include_router(auth_router, prefix="/api/auth", tags=["🔐 Authentication"])
app.include_router(coingecko_router, prefix="/api/coingecko", tags=["🦎 CoinGecko API"])
app.include_router(yfinance_router, prefix="/api/yfinance", tags=["📈 Yahoo Finance"])

# Root endpoint
@app.get("/", tags=["🏠 Home"])
async def root():
    """Welcome to CryptoAion AI - Universal Crypto Prediction Engine"""
    return {
        "message": "🚀 Welcome to CryptoAion AI - Universal Crypto Prediction Engine",
        "version": "2.0.0 - Best Version", 
        "status": "✅ Online",
        "docs": "/docs",
        "features": [
            "✅ Universal crypto support (any symbol)",
            "✅ All timeframes (30m, 1h, 4h, 24h)",
            "✅ Rich model performance metrics",
            "✅ Real-time WebSocket predictions",
            "✅ Expert-tuned for accuracy & speed"
        ]
    }

# Health check endpoint
@app.get("/health", tags=["🏠 Home"])
async def health():
    """Health check endpoint"""
    return {"status": "✅ healthy", "service": "CryptoAion AI"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "status": "❌ error"
        }
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
