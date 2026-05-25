# Federated Multimodal Emotion Recognition System Backend

A production-ready FastAPI backend for a Federated Multimodal Emotion Recognition system. The system manages federated learning rounds using the **Federated Averaging (FedAvg)** algorithm to aggregate local model weights uploaded by Android clients.

This backend supports two models simultaneously or modularly:
1. **LSTM Model** for Text Emotion Classification.
2. **MFCC + CNN Model** for Speech Emotion Classification.

Clients download the latest global model weights, train on local user data (speech/text inputs and feedback labels), and upload their updated local model weights. The backend coordinates wait-states, gathers updates from 3 clients, performs sample-count-weighted FedAvg, and releases the new global model version.

---

## Architecture & Federated Learning Flow

```mermaid
sequenceDiagram
    autonumber
    actor Client A as Android Client A
    actor Client B as Android Client B
    actor Client C as Android Client C
    participant Server as FastAPI Backend
    database DB as SQLite Database
    participant Storage as File Storage

    Note over Server, Storage: System starts & seeds Global Model V1
    
    Client A->>Server: GET /latest-model
    Server-->>Client A: Returns Version 1 Metadata
    Client A->>Server: GET /download-model/1
    Server-->>Client A: Downloads global_model_v1.npy
    Note over Client A: Performs local training on private text/speech data

    Client B->>Server: GET /latest-model
    Server-->>Client B: Returns Version 1 Metadata
    Client B->>Server: GET /download-model/1
    Server-->>Client B: Downloads global_model_v1.npy
    Note over Client B: Performs local training on private text/speech data

    Client A->>Server: POST /upload-update (Round 1, sample_count=150)
    Server->>Storage: Saves client_1/round_1.npy
    Server->>DB: Records ClientUpdate, increments received_clients = 1
    Server-->>Client A: {"success": true, "aggregation_triggered": false}

    Client B->>Server: POST /upload-update (Round 1, sample_count=250)
    Server->>Storage: Saves client_2/round_1.npy
    Server->>DB: Records ClientUpdate, increments received_clients = 2
    Server-->>Client B: {"success": true, "aggregation_triggered": false}

    Client C->>Server: POST /upload-update (Round 1, sample_count=100)
    Server->>Storage: Saves client_3/round_1.npy
    Server->>DB: Records ClientUpdate, increments received_clients = 3
    Note over Server: Threshold (3) met. Launches FedAvg background task
    Server-->>Client C: {"success": true, "aggregation_triggered": true}

    rect rgb(240, 248, 255)
        Note over Server, Storage: Background Task: perform_fedavg()
        Server->>Storage: Loads client_1, client_2, client_3 npy updates
        Server->>Server: Computes Weighted Average of Weights
        Server->>Storage: Saves global_models/global_model_v2.npy
        Server->>DB: Records GlobalModel V2, completes Round 1, initializes Round 2
    end
```

---

## Mathematical Foundation: Federated Averaging (FedAvg)

Standard Federated Averaging takes the local weights $w_t^i$ from each client $i$ in a federated learning round and computes the new global model weights $w_{t+1}$ using a weighted average proportional to the number of local training samples $n_i$ processed by that client.

The mathematical formulation implemented in the `perform_fedavg()` service is:

$$w_{t+1} = \sum_{i=1}^{K} \frac{n_i}{N} w_t^i = \frac{n_1 w^1 + n_2 w^2 + \dots + n_K w^K}{n_1 + n_2 + \dots + n_K}$$

Where:
* $K$ is the number of participating clients (here, $K = 3$).
* $n_i$ is the `sample_count` uploaded by client $i$.
* $N = \sum_{i=1}^{K} n_i$ is the total sum of samples across all participating clients in the round.
* $w_t^i$ represents the multi-layer weight tensor list of client $i$.

---

## Database Design

The database uses SQLite (via SQLAlchemy ORM). The schema design consists of four entities:

```mermaid
erDiagram
    CLIENT {
        int id PK
        string client_id UK
        datetime registered_at
    }
    FEDERATED_ROUND {
        int id PK
        int round_number UK
        int expected_clients
        int received_clients
        string status
        datetime created_at
    }
    CLIENT_UPDATE {
        int id PK
        string client_id FK
        int round_number FK
        int model_version
        int sample_count
        datetime upload_timestamp
        string update_file_path
    }
    GLOBAL_MODEL {
        int id PK
        int version UK
        int round_number
        string model_path
        datetime created_at
    }

    CLIENT ||--o{ CLIENT_UPDATE : "uploads"
    FEDERATED_ROUND ||--o{ CLIENT_UPDATE : "tracks"
```

---

## File Storage Structure

Model files and updates are compartmentalized cleanly inside root-level directories:

```
backend/
├── client_updates/
│   ├── client_1/
│   │   ├── round_1.npy
│   │   └── round_2.npy
│   └── client_2/
│       └── round_1.npy
└── global_models/
    ├── global_model_v1.npy  <-- Auto-seeded on startup
    └── global_model_v2.npy  <-- Generated after Round 1 FedAvg
```

---

## Setup and Configuration

This project is fully designed to support zero-credential Git practices and works safely in both local environments and cloud hosting runtimes (like Railway) without exposing any API keys or credentials in source control.

### 1. Local Configuration (.env Setup)
The project utilizes a `.env` file for local development configuration, which is ignored by Git. A template is provided in [.env.example](file:///c:/Users/Suleman%20Agasimani/OneDrive/Desktop/EIS/backend/.env.example).

To set up environment variables locally:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and adjust the variables for your system:
   ```env
   ENV=development
   DATABASE_URL=sqlite:///database/federated.db
   EXPECTED_CLIENTS=3
   MODEL_STORAGE_PATH=global_models
   CLIENT_UPDATE_PATH=client_updates
   UPLOAD_PATH=uploads
   ```

### 2. Local Setup Guide
1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the Backend Server:**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   *Note: On startup, the backend automatically seeds the base model (V1, Round 0) at `global_models/global_model_v1.npy` and verifies your environment variables. If critical parameters are missing or misconfigured, clear warnings are printed to stderr.*
4. **Run the Integration Test Flow:**
   ```bash
   python test_federated_flow.py
   ```

---

## Railway Cloud Deployment Guide

This backend is pre-configured for instant deployment on [Railway](https://railway.app) using either a Git-backed automatic deploy or the Railway CLI.

### 1. Railway Environment Configuration (Secrets & Variables)
When deploying to Railway, **do not upload the `.env` file** (which is automatically blocked by our `.gitignore`). Instead, navigate to your project dashboard's **Variables** tab and provision the following environment variables:

| Variable Key | Suggested Production Value | Description |
| :--- | :--- | :--- |
| `ENV` | `production` | Enables production checks & optimizations. |
| `EXPECTED_CLIENTS` | `3` | Number of clients to wait for before performing FedAvg. |
| `MODEL_STORAGE_PATH` | `global_models` | Directory name for global aggregated models. |
| `CLIENT_UPDATE_PATH` | `client_updates` | Directory name for client updates. |
| `DATABASE_URL` | `postgresql://user:pass@host:port/dbname` | *Strongly Recommended:* Railway ephemeral filesystems clear SQLite databases on service redeployments. Connect a Railway PostgreSQL service and paste the connection URL here. |

### 2. Deployment Configurations
We've included native deployment manifests at the root directory:
* **[`railway.json`](file:///c:/Users/Suleman%20Agasimani/OneDrive/Desktop/EIS/backend/railway.json):** Custom build configuration directing Railway to compile Python 3.11 using Nixpacks and execute the Uvicorn command bound to `$PORT`.
* **[`Procfile`](file:///c:/Users/Suleman%20Agasimani/OneDrive/Desktop/EIS/backend/Procfile):** Universal process file declaring Uvicorn runtime commands as a fallback/standard for web workers.

### 3. Deploying Steps
1. Push your local backend repository to a private or public GitHub repository. (Because our configuration handles secrets securely through environment variables and ignore sheets, **your repository is 100% safe to make public**).
2. Connect your GitHub repository to Railway:
   * Go to [Railway Dashboard](https://railway.app).
   * Click **New Project** -> **Deploy from GitHub repo**.
   * Select your repository.
3. Configure your variables (especially `DATABASE_URL` and `EXPECTED_CLIENTS`) under the service **Variables** tab.
4. Click **Deploy**. Railway will build the Nixpack virtual image and serve your Federated Learning endpoints!

---

## API Endpoints Reference

### 1. Health Check
* **Endpoint:** `GET /health`
* **Response:**
```json
{
  "status": "healthy"
}
```

### 2. Register Client
* **Endpoint:** `POST /register-client`
* **Body:**
```json
{
  "client_id": "client_mobile_104"
}
```
* **Response:**
```json
{
  "client_id": "client_mobile_104",
  "registered_at": "2026-05-25T17:16:34.123456"
}
```

### 3. Upload Client Model Weights
* **Endpoint:** `POST /upload-update`
* **Format:** `multipart/form-data`
* **Form Parameters:**
  * `client_id` (Text): `"client_mobile_104"`
  * `round_number` (Text/Int): `1`
  * `sample_count` (Text/Int): `120`
  * `weights_file` (Binary File): Uploaded `.npy` weights file
* **Response (Waiting for more clients):**
```json
{
  "success": true,
  "aggregation_triggered": false,
  "message": "Successfully received weights from client 'client_mobile_104' for round 1."
}
```
* **Response (3rd Client triggers Aggregation):**
```json
{
  "success": true,
  "aggregation_triggered": true,
  "message": "Successfully received weights from client 'client_mobile_106' for round 1."
}
```

### 4. Fetch Latest Model Details
* **Endpoint:** `GET /latest-model`
* **Response:**
```json
{
  "version": 1,
  "round": 0,
  "download_url": "/download-model/1"
}
```

### 5. Download Model weights
* **Endpoint:** `GET /download-model/{version}`
* **Returns:** Binary stream of `.npy` file.

### 6. Query Round Status
* **Endpoint:** `GET /round-status/{round_number}`
* **Response:**
```json
{
  "round": 1,
  "expected_clients": 3,
  "received_clients": 2,
  "status": "waiting"
}
```

---

## Example cURL Requests

### Register Client
```bash
curl -X POST "http://127.0.0.1:8000/register-client" \
     -H "Content-Type: application/json" \
     -d "{\"client_id\": \"client_demo_1\"}"
```

### Get Latest Model
```bash
curl -X GET "http://127.0.0.1:8000/latest-model"
```

### Download Model Version 1
```bash
curl -o global_model_v1.npy -X GET "http://127.0.0.1:8000/download-model/1"
```

### Upload Model Weights Update (For Client 1, Round 1)
```bash
# Generate a dummy numpy weights file first:
python -c "import numpy as np; np.save('my_weights.npy', [np.zeros((10, 5))], allow_pickle=True)"

# Upload the weights:
curl -X POST "http://127.0.0.1:8000/upload-update" \
     -F "client_id=client_demo_1" \
     -F "round_number=1" \
     -F "sample_count=180" \
     -F "weights_file=@my_weights.npy"
```

---

## Android Kotlin Integration Guide

To perform federated learning on Android, the mobile application should load the model weights, train on local speech/text data, extract the newly trained model weights, serialize them, and upload them to this backend.

Below are modular Kotlin classes demonstrating how to interact with this backend.

### 1. Retrofit Service Interface

Define the API client to register the client and upload local updates.

```kotlin
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

interface FederatedLearningApi {

    @POST("register-client")
    suspend fun registerClient(
        @Body request: RegisterRequest
    ): Response<RegisterResponse>

    @GET("latest-model")
    suspend fun getLatestModel(): Response<LatestModelMetadata>

    @GET
    @Streaming
    suspend fun downloadModel(
        @Url fileUrl: String
    ): Response<ResponseBody>

    @Multipart
    @POST("upload-update")
    suspend fun uploadUpdate(
        @Part("client_id") clientId: RequestBody,
        @Part("round_number") roundNumber: RequestBody,
        @Part("sample_count") sampleCount: RequestBody,
        @Part weightsFile: MultipartBody.Part
    ): Response<UploadResponse>
}

// Pydantic-matching DTO models
data class RegisterRequest(val client_id: String)
data class RegisterResponse(val client_id: String, val registered_at: String)
data class LatestModelMetadata(val version: Int, val round: Int, val download_url: String)
data class UploadResponse(val success: Boolean, val aggregation_triggered: Boolean, val message: String)
```

### 2. Serializing Android TFLite Weights to NPY Format

TensorFlow Lite supports on-device training using signature keys or the TFLite Model Personalization library. Alternatively, you can use PyTorch Mobile or custom float array serializations.

The following Kotlin utility demonstrates how to convert a nested list of floats (model layer weights) into a structured byte format resembling NumPy's uncompressed serialization standard or serialized binary files that can be easily parsed by the server.

```kotlin
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WeightsSerializer {

    /**
     * Serializes nested layer weights (List of Float Arrays) to a byte array.
     * To align with NumPy or custom binary loading, this function prefixes each array
     * with its dimension size, shape details, and raw float values.
     */
    fun serializeLayerWeights(layers: List<Array<FloatArray>>): ByteArray {
        val bos = ByteArrayOutputStream()
        
        // Write the number of layers
        bos.write(intToBytes(layers.size))
        
        for (layer in layers) {
            val rows = layer.size
            val cols = if (rows > 0) layer[0].size else 0
            
            // Write shapes
            bos.write(intToBytes(rows))
            bos.write(intToBytes(cols))
            
            // Write float elements in Little Endian format
            val buffer = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (row in layer) {
                for (value in row) {
                    buffer.putFloat(value)
                }
            }
            bos.write(buffer.array())
        }
        return bos.toByteArray()
    }

    private fun intToBytes(value: Int): ByteArray {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
    }
}
```

### 3. Triggering Retrofit Multipart File Upload from Android

This module demonstrates preparing the multipart request payload and executing the upload asynchronously inside a Kotlin Coroutine.

```kotlin
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

class FederatedLearningClient(
    private val api: FederatedLearningApi,
    private val context: Context
) {

    suspend fun uploadClientModelWeights(
        clientId: String,
        roundNumber: Int,
        sampleCount: Int,
        serializedWeights: ByteArray
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            // Write the serialized weights bytes to a temporary local file on cache
            val tempFile = File(context.cacheDir, "client_round_update.npy")
            tempFile.writeBytes(serializedWeights)

            // Prepare text inputs as request bodies
            val clientIdPart = clientId.toRequestBody("text/plain".toMediaTypeOrNull())
            val roundNumberPart = roundNumber.toString().toRequestBody("text/plain".toMediaTypeOrNull())
            val sampleCountPart = sampleCount.toString().toRequestBody("text/plain".toMediaTypeOrNull())

            // Prepare weight file multipart body
            val fileRequestBody = tempFile.asRequestBody("application/octet-stream".toMediaTypeOrNull())
            val filePart = MultipartBody.Part.createFormData(
                "weights_file",
                tempFile.name,
                fileRequestBody
            )

            // Execute POST request
            val response = api.uploadUpdate(
                clientId = clientIdPart,
                roundNumber = roundNumberPart,
                sampleCount = sampleCountPart,
                weightsFile = filePart
            )

            // Clean up temporary file
            if (tempFile.exists()) tempFile.delete()

            if (response.isSuccessful && response.body() != null) {
                val uploadResult = response.body()!!
                println("[FL] Upload success: ${uploadResult.success}. Aggregation: ${uploadResult.aggregation_triggered}")
                return@withContext uploadResult.success
            } else {
                println("[FL] Upload failed with code: ${response.code()}")
                return@withContext false
            }
        } catch (e: Exception) {
            println("[FL] Exception during weight upload: ${e.localizedMessage}")
            return@withContext false
        }
    }
}
```

---

## Operational Scenarios & Error Handling

The backend implements comprehensive validation to prevent state mismatch and model corruption:

1. **Invalid Client Verification:** Any `/upload-update` request checks the `client_id` against the `clients` table. Unregistered clients receive an `HTTP 401 Unauthorized`.
2. **Round State Locking:** If an update is uploaded for a round that is `completed`, `aggregating`, or `failed`, the backend blocks it with `HTTP 400 Bad Request`.
3. **Duplicate Upload Prevention:** A composite database index and query ensure that a single client cannot upload more than once per round, returning `HTTP 409 Conflict` on duplicates.
4. **NumPy Integrity Check:** Uploaded weights are immediately parsed and loaded on save. If the file is corrupted, truncated, or doesn't conform to `.npy` formats, the server deletes the corrupted file and throws `HTTP 400 Bad Request`.
5. **Shape and Structural Consistency Guard:** In `perform_fedavg()`, the server verifies layer count, individual layer shapes, and dictionary keys across all uploaded models. A mismatch throws a structured error, marks the round status as `failed` in the database, and prevents saving a corrupted global model.
