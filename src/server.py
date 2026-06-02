import os
from dotenv import load_dotenv
load_dotenv() # Load local environment variables from .env file

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import cloud database configurations and relational models
from cloud_database import init_db, get_db
from models import PostgresUser, PostgresAttendance

# ── Optional AI Engine (not required for cloud sync endpoints) ────────────────
# In cloud/serverless deployments the mobile app handles all on-device inference.
# These imports are only needed for the /api/v1/embeddings desktop endpoint.
AI_ENGINES_AVAILABLE = False
detector = None
preprocessor = None
recognizer = None

try:
    import cv2
    import numpy as np
    from detect import FaceDetector
    from preprocess import FacePreprocessor
    from recognize import FaceRecognizer

    print("[INIT] Loading Face Detector...")
    detector = FaceDetector(min_detection_confidence=0.8, model_selection=0)

    print("[INIT] Loading Face Preprocessor...")
    preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")

    print("[INIT] Loading Face Recognizer...")
    recognizer = FaceRecognizer()

    AI_ENGINES_AVAILABLE = True
    print("[INIT] ✅ Biometric AI engines loaded successfully.")
except Exception as e:
    print(f"[INIT] ⚠️  Biometric AI engines not available: {e}")
    print("[INIT] Sync endpoints (/api/v1/sync/*) are fully operational.")
    print("[INIT] Embedding endpoint (/api/v1/embeddings) requires local deployment with MediaPipe.")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BLINK — Biometric Embedding API Server",
    description="Calculates normalized 128D/512D face embeddings from uploaded images using MobileFaceNet ONNX model.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    # Bootstrap relational schemas (PostgreSQL or local SQLite fallback)
    init_db()

# Enable CORS so the React Native app can call it directly during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/embeddings")
async def calculate_embeddings(file: UploadFile = File(...)):
    """
    Receives an uploaded photo, crops and aligns the face, 
    calculates the 128D/512D face embedding, and returns it.
    """
    if not AI_ENGINES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Biometric AI engines unavailable in this deployment. Use on-device mobile inference instead."
        )
    try:
        # 1. Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")
            
        # 2. Run face detection to extract ROI
        detections = detector.detect_faces(img)
        
        face_crop = None
        if detections:
            # Focus on the most prominent face in the foreground
            detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
            primary_face = detections[0]
            bbox = primary_face['bbox']
            
            # Crop & Align face crop with 15% padding margin (matches desktop client) and rotation alignment
            face_crop = detector.crop_and_align(img, bbox, landmarks=primary_face.get('landmarks'), padding_ratio=0.15)
            
        if face_crop is None:
            # Fallback if no face was detected: resize raw image directly to target crop size
            print("[WARN] No face crop detected. Falling back to direct full-frame resize.")
            face_crop = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

        # 3. Preprocess face crop (LAB conversion + CLAHE lighting adjustment + normalization)
        tensor = preprocessor.preprocess(face_crop, apply_clahe=True)
        
        # 4. Run MobileFaceNet ONNX inference to generate embedding vector
        embedding = recognizer.generate_embedding(tensor)
        
        # 5. Return unit-normalized vector as standard float JSON list
        return {
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "model_precision": recognizer.model_precision
        }
        
    except Exception as e:
        print(f"[ERROR] Embedding computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal embedding error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "BLINK Core",
        "ai_engines_available": AI_ENGINES_AVAILABLE,
        "simulated_fallback": recognizer.simulated if recognizer else None,
    }

# ── Pydantic Request Payloads for Synchronization ────────────────────────────
class UserSyncPayload(BaseModel):
    id: str
    name: str
    encrypted_envelope_json: Optional[Dict[str, Any]] = None
    model_mode: str
    enrolled_at: str
    status: str

class AttendanceSyncPayload(BaseModel):
    id: str
    person_id: str
    timestamp: str
    status: str
    punctuality: str

# ── E2EE Synchronization API Endpoints ─────────────────────────────────────────
@app.post("/api/v1/sync/users")
def sync_users(payloads: List[UserSyncPayload], db: Session = Depends(get_db)):
    """
    Synchronizes users (E2EE envelopes) from the mobile device to the cloud database.
    Performs upserts (insert or update on primary key conflict).
    """
    synced_ids = []
    for p in payloads:
        # Check if user already exists in PostgreSQL/SQLite
        user = db.query(PostgresUser).filter(PostgresUser.id == p.id).first()
        if user:
            user.name = p.name
            user.encrypted_envelope_json = p.encrypted_envelope_json
            user.status = p.status
            user.model_mode = p.model_mode
        else:
            user = PostgresUser(
                id=p.id,
                name=p.name,
                encrypted_envelope_json=p.encrypted_envelope_json,
                model_mode=p.model_mode,
                enrolled_at=p.enrolled_at,
                status=p.status
            )
            db.add(user)
        synced_ids.append(p.id)
    db.commit()
    return {"success": True, "synced_count": len(synced_ids), "synced_ids": synced_ids}

@app.get("/api/v1/sync/users")
def get_synced_users(db: Session = Depends(get_db)):
    """
    Returns all registered users and their E2EE biometric envelopes from the cloud,
    allowing other authorized supervisor devices to download and match them offline.
    """
    users = db.query(PostgresUser).all()
    result = []
    for u in users:
        result.append({
            "id": u.id,
            "name": u.name,
            "encrypted_envelope_json": u.encrypted_envelope_json,
            "model_mode": u.model_mode,
            "enrolled_at": u.enrolled_at,
            "status": u.status
        })
    return {"success": True, "users": result}

@app.post("/api/v1/sync/attendance")
def sync_attendance(payloads: List[AttendanceSyncPayload], db: Session = Depends(get_db)):
    """
    Batch synchronizes local offline attendance logs to the central database.
    """
    synced_ids = []
    for p in payloads:
        log = db.query(PostgresAttendance).filter(PostgresAttendance.id == p.id).first()
        if not log:
            log = PostgresAttendance(
                id=p.id,
                person_id=p.person_id,
                timestamp=p.timestamp,
                status=p.status,
                punctuality=p.punctuality
            )
            db.add(log)
        synced_ids.append(p.id)
    db.commit()
    return {"success": True, "synced_count": len(synced_ids), "synced_ids": synced_ids}

if __name__ == "__main__":
    # Run server on port 8000 accessible from all network devices (critical for mobile debug testing)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
