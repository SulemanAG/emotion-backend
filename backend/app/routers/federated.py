import os
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Client, FederatedRound, ClientUpdate
from app.schemas import ClientRegister, ClientResponse, RoundStatusResponse, UploadUpdateResponse
from app.services.storage_service import StorageService
from app.services.version_service import VersionService
from app.services.aggregation_service import AggregationService
from app.utils.file_utils import save_uploaded_file, validate_numpy_file

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/register-client", response_model=ClientResponse, status_code=status.HTTP_201_CREATED, tags=["Federated Management"])
def register_client(client_data: ClientRegister, db: Session = Depends(get_db)):
    """
    Registers a new federated client. If the client is already registered, returns existing info.
    """
    # Check if client already exists
    existing_client = db.query(Client).filter(Client.client_id == client_data.client_id).first()
    if existing_client:
        logger.info(f"Client '{client_data.client_id}' already registered.")
        return existing_client
        
    # Register client
    new_client = Client(client_id=client_data.client_id)
    db.add(new_client)
    try:
        db.commit()
        db.refresh(new_client)
        logger.info(f"Successfully registered client: {client_data.client_id}")
        return new_client
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering client {client_data.client_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register client due to a database error."
        )


@router.post("/upload-update", response_model=UploadUpdateResponse, tags=["Federated Management"])
def upload_update(
    background_tasks: BackgroundTasks,
    client_id: str = Form(..., description="The registered ID of the client"),
    round_number: int = Form(..., description="The federated learning round number"),
    sample_count: int = Form(..., description="Number of training samples used locally by client"),
    weights_file: UploadFile = File(..., description="Client model weights serialized as .npy"),
    db: Session = Depends(get_db)
):
    """
    Handles local weight updates uploaded by clients.
    Saves updates, performs validations, and triggers FedAvg aggregation once enough updates are gathered.
    """
    # 1. Validate Client
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Client '{client_id}' is not registered. Please register first."
        )
        
    # 2. Validate Federated Round
    fed_round = db.query(FederatedRound).filter(FederatedRound.round_number == round_number).first()
    if not fed_round:
        # If the requested round does not exist, check if it's ahead of current round
        # To make it robust, let's create a round record if we are running the system dynamically, 
        # but standard flow dictates that rounds are created sequentially.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federated round {round_number} has not been initialized or does not exist."
        )
        
    if fed_round.status != "waiting":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Federated round {round_number} is no longer accepting uploads. Status: {fed_round.status}."
        )
        
    # 3. Check for Duplicate Uploads
    duplicate = db.query(ClientUpdate).filter(
        ClientUpdate.client_id == client_id,
        ClientUpdate.round_number == round_number
    ).first()
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client '{client_id}' has already uploaded weights for round {round_number}."
        )
        
    if sample_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sample count must be greater than zero."
        )
        
    # 4. Save file securely
    destination_path = StorageService.get_client_update_path(client_id, round_number)
    try:
        save_uploaded_file(weights_file, destination_path)
    except Exception as e:
        logger.error(f"Failed to write uploaded file to storage: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server failed to save weight file. Please try again."
        )
        
    # 5. Validate NumPy Weights File
    if not validate_numpy_file(destination_path):
        if destination_path.exists():
            os.remove(destination_path)
        logger.error(f"Client {client_id} uploaded a corrupted or invalid NumPy weights file.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or readable NumPy (.npy) file."
        )

    # 6. Retrieve active model version
    latest_model = VersionService.get_latest_global_model(db)
    current_model_version = latest_model.version if latest_model else 1
    
    # 7. Create ClientUpdate database record
    client_update = ClientUpdate(
        client_id=client_id,
        round_number=round_number,
        model_version=current_model_version,
        sample_count=sample_count,
        update_file_path=str(destination_path)
    )
    db.add(client_update)
    
    # Increment received clients under lock safety
    fed_round.received_clients += 1
    
    aggregation_triggered = False
    
    # 8. Check if aggregation limit met
    if fed_round.received_clients >= fed_round.expected_clients:
        # Mark round status as "aggregating" to lock further uploads immediately
        fed_round.status = "aggregating"
        aggregation_triggered = True
        
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        if destination_path.exists():
            os.remove(destination_path)
        logger.error(f"Failed to commit upload transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database transaction failed while saving client update."
        )
        
    if aggregation_triggered:
        # Trigger FedAvg aggregation in a background task
        background_tasks.add_task(AggregationService.trigger_aggregation, db, round_number)
        logger.info(f"Aggregation successfully triggered for round {round_number}.")
        
    return {
        "success": True,
        "aggregation_triggered": aggregation_triggered,
        "message": f"Successfully received weights from client '{client_id}' for round {round_number}."
    }


@router.get("/round-status/{round_number}", response_model=RoundStatusResponse, tags=["Federated Management"])
def round_status(round_number: int, db: Session = Depends(get_db)):
    """
    Returns the participation status, current uploads, and operational state of a specific federated round.
    """
    fed_round = db.query(FederatedRound).filter(FederatedRound.round_number == round_number).first()
    if not fed_round:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Federated round {round_number} not found."
        )
        
    return {
        "round": fed_round.round_number,
        "expected_clients": fed_round.expected_clients,
        "received_clients": fed_round.received_clients,
        "status": fed_round.status
    }
