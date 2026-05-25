import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from app.database import Base

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, unique=True, index=True, nullable=False)
    registered_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class FederatedRound(Base):
    __tablename__ = "federated_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, index=True, nullable=False)
    expected_clients = Column(Integer, default=3, nullable=False)
    received_clients = Column(Integer, default=0, nullable=False)
    status = Column(String, default="waiting", nullable=False)  # "waiting", "completed", "failed"
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ClientUpdate(Base):
    __tablename__ = "client_updates"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=False)
    round_number = Column(Integer, ForeignKey("federated_rounds.round_number"), nullable=False)
    model_version = Column(Integer, nullable=False)
    sample_count = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    update_file_path = Column(String, nullable=False)


class GlobalModel(Base):
    __tablename__ = "global_models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(Integer, unique=True, index=True, nullable=False)
    round_number = Column(Integer, nullable=False)
    model_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
