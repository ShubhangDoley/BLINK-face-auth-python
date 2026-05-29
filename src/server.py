import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List

# Import our robust core BLINK pipeline components
from detect import FaceDetector
from preprocess import FacePreprocessor
from recognize import FaceRecognizer

# Initialize FastAPI App
app = FastAPI(
    title="BLINK — Biometric Embedding API Server",
    description="Calculates normalized 128D/512D face embeddings from uploaded images using MobileFaceNet ONNX model.",
    version="1.0.0"
)

# Enable CORS so our React Native app can call it directly during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize biometric engines
print("[INIT] Loading Face Detector...")
detector = FaceDetector(min_detection_confidence=0.8, model_selection=0)

print("[INIT] Loading Face Preprocessor...")
preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")

print("[INIT] Loading Face Recognizer...")
recognizer = FaceRecognizer()

@app.post("/api/v1/embeddings")
async def calculate_embeddings(file: UploadFile = File(...)):
    """
    Receives an uploaded photo, crops and aligns the face, 
    calculates the 128D/512D face embedding, and returns it.
    """
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
        "simulated_fallback": recognizer.simulated
    }

if __name__ == "__main__":
    # Run server on port 8000 accessible from all network devices (critical for mobile debug testing)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
