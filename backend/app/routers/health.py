from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """
    Simple health check endpoint returning the status of the service.
    """
    return {"status": "healthy"}
