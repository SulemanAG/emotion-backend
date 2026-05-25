import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base, SessionLocal
from app.services.aggregation_service import AggregationService
from app.routers import health, federated, models_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Handles startup database initialization and initial global model seeding.
    """
    logger.info("Initializing database and models...")
    # Create all database tables
    Base.metadata.create_all(bind=engine)
    
    # Seed initial model weights (Version 1, Round 0) if database is fresh
    db = SessionLocal()
    try:
        seeded_model = AggregationService.seed_initial_model(db)
        if seeded_model:
            logger.info(f"Database seeded successfully. Created global model v{seeded_model.version} (Round {seeded_model.round_number}).")
        else:
            logger.info("Database already containing active global models. Seeding skipped.")
    except Exception as e:
        logger.error(f"Startup seeding failed: {str(e)}")
    finally:
        db.close()
        
    yield
    logger.info("Server shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-ready FastAPI backend managing federated learning rounds (FedAvg) and model weights distribution for LSTM text emotion recognition and CNN speech emotion recognition.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for testing and connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(federated.router)
app.include_router(models_router.router)
