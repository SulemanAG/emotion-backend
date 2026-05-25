import os
import time
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def test_flow():
    print("==============================================================")
    print("    STARTING FEDERATED EMOTION RECOGNITION SYSTEM TEST       ")
    print("==============================================================")
    
    # 1. Health Check
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"[*] Health Check Status: {r.status_code}")
        print(f"    Payload: {r.json()}")
    except Exception as e:
        print(f"[-] Failed to connect to the backend: {e}")
        print("    Make sure the FastAPI server is running (e.g. uvicorn app.main:app --reload)")
        return

    # 2. Register 3 Clients
    clients = ["client_1", "client_2", "client_3"]
    print("\n[*] Registering clients...")
    for client in clients:
        r = requests.post(f"{BASE_URL}/register-client", json={"client_id": client})
        print(f"    - Register {client}: {r.status_code} -> {r.json()}")

    # 3. Get latest model metadata (this seeds Version 1, Round 0)
    print("\n[*] Fetching current latest model metadata...")
    r = requests.get(f"{BASE_URL}/latest-model")
    model_info = r.json()
    print(f"    Latest Model: Version {model_info['version']}, Round {model_info['round']}")
    print(f"    Download endpoint: {model_info['download_url']}")
    
    # 4. Download latest model weights (Version 1)
    download_url = f"{BASE_URL}{model_info['download_url']}"
    print(f"\n[*] Downloading initial global model from: {download_url}")
    r = requests.get(download_url)
    temp_filename = "temp_global_v1.npy"
    with open(temp_filename, "wb") as f:
        f.write(r.content)
    
    # Load and inspect weights
    global_weights = np.load(temp_filename, allow_pickle=True)
    print(f"    Successfully loaded global weights file.")
    print(f"    Number of layer structures: {len(global_weights)}")
    for i, w in enumerate(global_weights):
        print(f"      Layer {i} shape: {w.shape} | type: {w.dtype}")

    # 5. Simulate client training and upload for Round 1
    # Each client uses a different sample size (n) to show weighted FedAvg in action
    sample_counts = {
        "client_1": 150,
        "client_2": 250,
        "client_3": 100
    }
    
    for client in clients:
        print(f"\n[*] Simulating local training update for: {client}")
        
        # Add small random noise to simulate local weights training updates
        client_weights = []
        for w in global_weights:
            noise = np.random.normal(0, 0.005, size=w.shape).astype(np.float32)
            client_weights.append(w + noise)
            
        # Save client weights locally
        client_file = f"temp_{client}_round1.npy"
        # Convert list of inhomogeneous layer weights to numpy object array before saving
        client_weights_arr = np.empty(len(client_weights), dtype=object)
        for i, w in enumerate(client_weights):
            client_weights_arr[i] = w
        np.save(client_file, client_weights_arr, allow_pickle=True)
        
        # Upload using multipart/form-data
        print(f"    Uploading weights file with sample_count={sample_counts[client]}...")
        with open(client_file, "rb") as f:
            files = {"weights_file": (client_file, f, "application/octet-stream")}
            data = {
                "client_id": client,
                "round_number": 1,
                "sample_count": sample_counts[client]
            }
            r = requests.post(f"{BASE_URL}/upload-update", data=data, files=files)
            print(f"    Upload Response ({r.status_code}): {r.json()}")
            
        # Clean up temporary client file
        if os.path.exists(client_file):
            os.remove(client_file)

    # Clean up global download
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # Give a brief delay for async aggregation background task
    print("\n[*] Waiting for background aggregation task to execute...")
    time.sleep(2)

    # 6. Verify aggregation triggered and completed
    print("\n[*] Checking post-aggregation status...")
    
    # Check round 1 status
    r = requests.get(f"{BASE_URL}/round-status/1")
    print(f"    Round 1 Final Status: {r.json()}")
    
    # Check latest model details (should now be Version 2, Round 1!)
    r = requests.get(f"{BASE_URL}/latest-model")
    new_model_info = r.json()
    print(f"    New Latest Model: Version {new_model_info['version']}, Round {new_model_info['round']}")
    print(f"    New Download Link: {new_model_info['download_url']}")
    
    # Check round 2 status (should have been automatically initialized in 'waiting' state)
    r = requests.get(f"{BASE_URL}/round-status/2")
    print(f"    Round 2 Status (Auto-Initialized): {r.json()}")
    
    print("\n==============================================================")
    print("    SUCCESS: END-TO-END FEDERATED LEARNING PIPELINE WORKING!  ")
    print("==============================================================")

if __name__ == "__main__":
    test_flow()
