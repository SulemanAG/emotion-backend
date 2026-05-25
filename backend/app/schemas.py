import datetime
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"

class ClientRegister(BaseModel):
    client_id: str = Field(..., description="Unique client identifier", min_length=1)

class ClientResponse(BaseModel):
    client_id: str
    registered_at: datetime.datetime

    class Config:
        from_attributes = True

class RoundStatusResponse(BaseModel):
    round: int = Field(..., alias="round")
    expected_clients: int
    received_clients: int
    status: str

    class Config:
        from_attributes = True
        populate_by_name = True

class LatestModelResponse(BaseModel):
    version: int
    round: int
    download_url: str

class UploadUpdateResponse(BaseModel):
    success: bool
    aggregation_triggered: bool
    message: str = ""
