from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models import GlobalModel

class VersionService:
    @staticmethod
    def get_latest_global_model(db: Session) -> Optional[GlobalModel]:
        """
        Retrieves the latest global model based on the version number.
        """
        return db.query(GlobalModel).order_by(desc(GlobalModel.version)).first()

    @staticmethod
    def get_model_by_version(db: Session, version: int) -> Optional[GlobalModel]:
        """
        Retrieves a specific global model by its version number.
        """
        return db.query(GlobalModel).filter(GlobalModel.version == version).first()

    @staticmethod
    def create_global_model(db: Session, version: int, round_number: int, model_path: str) -> GlobalModel:
        """
        Creates and stores database metadata for a new aggregated global model.
        """
        db_model = GlobalModel(
            version=version,
            round_number=round_number,
            model_path=str(model_path)
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model
