from pathlib import Path
from app.config import settings

class StorageService:
    @staticmethod
    def get_client_update_path(client_id: str, round_number: int) -> Path:
        """
        Generates a secure, structured path for saving a client's round update file.
        Format: client_updates/<client_id>/round_<round_number>.npy
        """
        # Sanitize client_id to prevent directory traversal attacks
        sanitized_client_id = "".join(c for c in client_id if c.isalnum() or c in ("-", "_"))
        if not sanitized_client_id:
            sanitized_client_id = "default_client"
            
        return settings.CLIENT_UPDATES_DIR / sanitized_client_id / f"round_{round_number}.npy"

    @staticmethod
    def get_global_model_path(version: int) -> Path:
        """
        Generates a structured path for saving/loading a specific global model version.
        Format: global_models/global_model_v<version>.npy
        """
        return settings.GLOBAL_MODEL_DIR / f"global_model_v{version}.npy"
