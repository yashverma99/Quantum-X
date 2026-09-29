import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db
from app.api import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mediqai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing MediQAI backend services...")
    # Initialize database tables gracefully without blocking startup
    try:
        init_db()
        logger.info("Database schemas initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}. Running with database disconnected.")
    yield
    logger.info("Shutting down MediQAI backend services...")


app = FastAPI(
    title=f"{settings.PROJECT_NAME} — Hybrid Quantum Machine Learning API",
    description=(
        "Research & clinical decision-support platform exploring Hybrid Classical-Quantum ML "
        "for high-dimensional biomedical risk screening (SIH26139). "
        "Strictly for research use; not an autonomous diagnostic tool."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "research_objective": (
            "Determine whether and under what conditions hybrid quantum machine learning "
            "provides useful predictive value for high-dimensional biomedical data."
        ),
        "disclaimer": (
            "Research and decision-support prototype. Results are not a medical diagnosis "
            "and should not replace professional medical assessment."
        ),
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred in MediQAI runtime."}
    )
