import os
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import LatestModelResponse
from app.services.version_service import VersionService
from app.services.aggregation_service import AggregationService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/latest-model", response_model=LatestModelResponse, tags=["Model Distribution"])
def latest_model(db: Session = Depends(get_db)):
    """
    Returns the metadata of the latest aggregated global model.
    If no model has been registered yet, it seeds the initial model (V1, Round 0) dynamically.
    """
    latest = VersionService.get_latest_global_model(db)
    if not latest:
        # Dynamic seeding if empty to ensure clients always get a base model
        logger.info("No global model found in database on request. Attempting to seed initial model.")
        latest = AggregationService.seed_initial_model(db)
        if not latest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No global models available on the server."
            )
            
    return {
        "version": latest.version,
        "round": latest.round_number,
        "download_url": f"/download-model/{latest.version}"
    }


@router.get("/download-model/{version}", tags=["Model Distribution"])
def download_model(version: int, db: Session = Depends(get_db)):
    """
    Downloads the global model weight file for the specified version.
    """
    model_record = VersionService.get_model_by_version(db, version)
    if not model_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Global model version {version} not found in database."
        )
        
    file_path = model_record.model_path
    if not os.path.exists(file_path):
        logger.error(f"Model file for version {version} not found on disk at: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model weight file for version {version} is missing from server storage."
        )
        
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"global_model_v{version}.npy"
    )
