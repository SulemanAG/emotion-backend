import shutil
from pathlib import Path
import numpy as np
from fastapi import UploadFile

def save_uploaded_file(upload_file: UploadFile, destination: Path) -> Path:
    """
    Saves an uploaded file to the specified destination path securely.
    """
    # Ensure parent directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the file content using a buffer stream
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return destination

def validate_numpy_file(file_path: Path) -> bool:
    """
    Validates whether the saved file is a valid NumPy (.npy) file and can be opened successfully.
    """
    if not file_path.exists():
        return False
    try:
        # Load file with pickle allowed to inspect structure
        data = np.load(file_path, allow_pickle=True)
        # Basic sanity check
        return data is not None
    except Exception:
        return False
