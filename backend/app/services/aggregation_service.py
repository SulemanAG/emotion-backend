import logging
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session
from app.models import FederatedRound, ClientUpdate, GlobalModel
from app.utils.fedavg import perform_fedavg
from app.services.storage_service import StorageService
from app.services.version_service import VersionService
from app.config import settings

logger = logging.getLogger(__name__)


class AggregationService:
    @staticmethod
    def trigger_aggregation(db: Session, round_number: int) -> bool:
        """
        Triggers FedAvg aggregation for the specified round number.
        Loads all client updates, averages weights, saves the new model version, and updates database records.
        """
        # Fetch current round
        db_round = db.query(FederatedRound).filter(FederatedRound.round_number == round_number).first()
        if not db_round:
            logger.error(f"Federated round {round_number} not found in database.")
            return False
            
        if db_round.status == "completed":
            logger.warning(f"Federated round {round_number} is already completed. Skipping aggregation.")
            return True
            
        # Get client updates for this round
        updates = db.query(ClientUpdate).filter(ClientUpdate.round_number == round_number).all()
        if len(updates) < db_round.expected_clients:
            logger.warning(f"Insufficient updates for round {round_number}. Expected {db_round.expected_clients}, got {len(updates)}")
            return False

        # Gather paths and sample counts
        paths = [up.update_file_path for up in updates]
        sample_counts = [up.sample_count for up in updates]
        
        try:
            logger.info(f"Starting FedAvg aggregation for round {round_number} using {len(updates)} clients.")
            
            # Perform aggregation using our FedAvg utility
            aggregated_weights = perform_fedavg(paths, sample_counts)
            
            # Fetch latest global model version to increment
            latest_model = VersionService.get_latest_global_model(db)
            next_version = (latest_model.version + 1) if latest_model else 1
            
            # Save aggregated weights
            new_model_path = StorageService.get_global_model_path(next_version)
            new_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert list of inhomogeneous layer weights to object array before saving
            if isinstance(aggregated_weights, list):
                arr = np.empty(len(aggregated_weights), dtype=object)
                for i, w in enumerate(aggregated_weights):
                    arr[i] = w
                aggregated_weights = arr
                
            # Save as numpy file
            np.save(new_model_path, aggregated_weights, allow_pickle=True)
            
            # Create global model record
            VersionService.create_global_model(
                db=db,
                version=next_version,
                round_number=round_number,
                model_path=str(new_model_path)
            )
            
            # Mark the round as completed
            db_round.status = "completed"
            db_round.received_clients = len(updates)
            db.commit()
            logger.info(f"Round {round_number} successfully completed and aggregated to Version {next_version}.")
            
            # Automatically spawn/seed the next federated round
            next_round_number = round_number + 1
            existing_next = db.query(FederatedRound).filter(FederatedRound.round_number == next_round_number).first()
            if not existing_next:
                new_round = FederatedRound(
                    round_number=next_round_number,
                    expected_clients=settings.EXPECTED_CLIENTS,
                    received_clients=0,
                    status="waiting"
                )
                db.add(new_round)
                db.commit()
                logger.info(f"Automatically initialized next federated round {next_round_number} in waiting state.")
                
            return True
            
        except Exception as e:
            logger.exception(f"FedAvg aggregation failed for round {round_number}: {str(e)}")
            db_round.status = "failed"
            db.commit()
            raise e

    @staticmethod
    def seed_initial_model(db: Session) -> Optional[GlobalModel]:
        """
        Seeds the first global model (Version 1, Round 0) if no global models exist in the database.
        This provides clients with an initial model to download and start training on.
        """
        latest_model = VersionService.get_latest_global_model(db)
        if latest_model is not None:
            return None
            
        logger.info("No global models found. Seeding initial global model weights (Version 1, Round 0).")
        
        # Structure a list representing typical LSTM and CNN layers for Emotion Recognition
        # LSTM: embedding layer, lstm weights, dense output
        # CNN: convolutional weights, pooling, dense output
        initial_weights = [
            np.random.normal(0, 0.1, size=(500, 32)).astype(np.float32),  # Text: Vocab size 500, embedding dim 32
            np.random.normal(0, 0.1, size=(32, 16)).astype(np.float32),   # Text: LSTM Hidden projection Dense
            np.random.normal(0, 0.1, size=(16,)).astype(np.float32),
            np.random.normal(0, 0.1, size=(128, 64)).astype(np.float32),  # Speech: MFCC features (128 flatten) to 64
            np.random.normal(0, 0.1, size=(64, 6)).astype(np.float32),    # Classification: 6 classes (e.g. Happy, Sad, Angry, Fear, Neutral, Surprise)
            np.random.normal(0, 0.1, size=(6,)).astype(np.float32)
        ]
        
        # Save seeded model
        next_version = 1
        model_path = StorageService.get_global_model_path(next_version)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert list of inhomogeneous layer weights to object array before saving
        seeded_weights_arr = np.empty(len(initial_weights), dtype=object)
        for i, w in enumerate(initial_weights):
            seeded_weights_arr[i] = w
            
        np.save(model_path, seeded_weights_arr, allow_pickle=True)
        
        # Store in DB
        db_model = VersionService.create_global_model(
            db=db,
            version=next_version,
            round_number=0,
            model_path=str(model_path)
        )
        
        # Ensure round 1 exists in waiting state
        round_1 = db.query(FederatedRound).filter(FederatedRound.round_number == 1).first()
        if not round_1:
            db_round = FederatedRound(
                round_number=1,
                expected_clients=settings.EXPECTED_CLIENTS,
                received_clients=0,
                status="waiting"
            )
            db.add(db_round)
            db.commit()
            logger.info("Initialized federated round 1.")
            
        return db_model
