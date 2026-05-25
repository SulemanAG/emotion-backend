import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Base Directory (root of the backend folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load local .env file if it exists (local development)
# For production environments like Railway, env vars are injected directly by the environment.
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)

class Settings:
    PROJECT_NAME: str = "Federated Multimodal Emotion Recognition Backend"
    
    # 2. Environment Settings
    ENV: str = os.getenv("ENV", "development")
    IS_PRODUCTION: bool = ENV.lower() in ("production", "prod") or "RAILWAY_ENVIRONMENT" in os.environ

    # 3. Database Configurations
    DEFAULT_DB_URL: str = f"sqlite:///{BASE_DIR}/database/federated.db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    
    # 4. Storage Directories Configurations
    MODEL_STORAGE_PATH: str = os.getenv("MODEL_STORAGE_PATH", "global_models")
    CLIENT_UPDATE_PATH: str = os.getenv("CLIENT_UPDATE_PATH", "client_updates")
    UPLOAD_PATH: str = os.getenv("UPLOAD_PATH", "uploads")
    
    # Resolve absolute Paths
    DB_DIR: Path = BASE_DIR / "database"
    GLOBAL_MODEL_DIR: Path = BASE_DIR / MODEL_STORAGE_PATH
    CLIENT_UPDATES_DIR: Path = BASE_DIR / CLIENT_UPDATE_PATH
    UPLOAD_DIR: Path = BASE_DIR / UPLOAD_PATH
    
    # 5. Federated Learning Configuration
    EXPECTED_CLIENTS: int = int(os.getenv("EXPECTED_CLIENTS", "3"))

    def create_dirs(self):
        """Ensure all dynamic storage directories exist in the environment."""
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.GLOBAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.CLIENT_UPDATES_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def validate_env(self):
        """
        Startup validation that checks the active config state
        and prints critical warnings if configuration errors are spotted.
        """
        warnings = []
        
        # A. Warn if using an ephemeral SQLite database in a production context
        if self.IS_PRODUCTION:
            if "sqlite" in self.DATABASE_URL.lower():
                warnings.append(
                    "[WARNING] SQLite database is running in a PRODUCTION context! SQLite storage is non-persistent\n"
                    "          on ephemeral server runtimes like Railway. It is strongly recommended to connect a proper\n"
                    "          relational database like PostgreSQL via setting the 'DATABASE_URL' environment variable."
                )
                
        # B. Info print showing client counts
        if "EXPECTED_CLIENTS" not in os.environ and not dotenv_path.exists():
            warnings.append(
                f"[INFO] EXPECTED_CLIENTS environment variable is not defined. Defaulting to: {self.EXPECTED_CLIENTS}"
            )
            
        # C. Output warnings directly to system console on startup
        if warnings:
            print("\n" + "!" * 80, file=sys.stderr)
            print("                FEDERATED BACKEND SYSTEM CONFIGURATION VALIDATION", file=sys.stderr)
            print("!" * 80, file=sys.stderr)
            for warn in warnings:
                print(warn, file=sys.stderr)
            print("!" * 80 + "\n", file=sys.stderr)

settings = Settings()
# Create folders on import to avoid race conditions when writing weights
settings.create_dirs()
# Execute startup verification checks
settings.validate_env()
