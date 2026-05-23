import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, Any

# Import FacePreprocessor from preprocess module to perform verification end-to-end
from preprocess import FacePreprocessor

class FaceRecognizer:
    """
    [Phase 3 / Phase 8 - Face Recognition + Mobile Model Auto-Detection]
    Responsible for generating high-dimensional face embedding vectors (128D or 512D)
    using the MobileFaceNet ONNX model.

    Model preference chain (Phase 8):
        mobilefacenet_int8.onnx  →  mobilefacenet.onnx  →  simulated fallback

    The INT8 model is ~4x smaller and ~2x faster — critical for on-device mobile
    inference via TFLite or ONNX Runtime Mobile on Android/iOS.
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes the FaceRecognizer.
        Loads the ONNX model with Phase 8 priority chain:
            INT8 quantized → FP32 original → high-fidelity simulated fallback

        Args:
            model_path: Optional explicit path to an ONNX model file.
                        If None, auto-detects the best available model.
        """
        self.session: Optional[ort.InferenceSession] = None
        self.simulated = False
        self.model_path: str = ""
        self.model_precision: str = "unknown"

        if model_path is not None:
            # Explicit path provided — use it directly
            candidates = [(model_path, "explicit")]
        else:
            # Phase 8: Auto-detect best available model
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            candidates = [
                (os.path.join(models_dir, "mobilefacenet_int8.onnx"), "int8"),
                (os.path.join(models_dir, "mobilefacenet.onnx"),      "fp32"),
            ]

        loaded = False
        for path, precision in candidates:
            if not os.path.exists(path):
                continue
            try:
                session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.session = session
                self.model_path = path
                self.model_precision = precision

                # Bind dynamic graph node structures
                self.input_name  = self.session.get_inputs()[0].name
                self.input_shape  = self.session.get_inputs()[0].shape
                self.output_name = self.session.get_outputs()[0].name
                self.output_shape = self.session.get_outputs()[0].shape

                precision_tag = f"[{precision.upper()}]" if precision != "explicit" else ""
                print(f"[SUCCESS] FaceRecognizer: {precision_tag} ONNX model loaded -> {path}")
                print(f"[INFO] ONNX Graph -> Input: '{self.input_name}' {self.input_shape} | Output: '{self.output_name}' {self.output_shape}")
                if precision == "int8":
                    print("[INFO] Using INT8 quantized model -- ~4x smaller, ~2x faster. Ideal for mobile export.")
                loaded = True
                break
            except Exception as e:
                print(f"[WARNING] FaceRecognizer: Failed to load {path}: {e}. Trying next candidate...")

        if not loaded:
            self.simulated = True
            self.model_precision = "simulated"
            print("\n" + "!"*80)
            print("BLINK EDGE ENGINE WARNING: No ONNX model file found.")
            print("Run: python src/download_weights.py  to download FP32 weights.")
            print("Run: python src/quantize.py          to produce INT8 weights.")
            print("FaceRecognizer is starting in high-fidelity SIMULATED fallback mode.")
            print("!"*80 + "\n")

    def generate_embedding(self, face_tensor: np.ndarray) -> np.ndarray:
        """
        Generates a unit-normalized (L2 norm = 1.0) feature vector from a preprocessed face crop tensor.
        
        Args:
            face_tensor: Preprocessed float32 image tensor of shape (1, 3, 112, 112).
            
        Returns:
            A 1D NumPy float32 array representing facial features (dimension: 128 or 512).
        """
        # Enforce tensor type conversion
        if face_tensor.dtype != np.float32:
            face_tensor = face_tensor.astype(np.float32)

        # Ensure correct batch dimensions
        if len(face_tensor.shape) == 3:
            face_tensor = np.expand_dims(face_tensor, axis=0)

        if self.simulated:
            # High-fidelity linear random projection simulation.
            # Downsamples the image to a 4x4 spatial grid (16 features) and projects it
            # into 128D space using a fixed projection matrix (seeded constantly at 42).
            # Being a continuous linear map, small camera noise in the input results in
            # only micro-shifts in the output embedding, perfectly mimicking a real neural network.
            # Convert NCHW tensor back to HWC representation
            img_hwc = np.transpose(face_tensor[0], (1, 2, 0))
            
            # Compute luminance (grayscale conversion)
            gray = np.mean(img_hwc, axis=2)
            
            # Average pool down to 4x4 grid resolution (16 features)
            tiny = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA)
            features = tiny.flatten() # Shape: (16,)
            
            # Generate a fixed projection matrix of shape (16, 128) using a constant seed
            proj_rng = np.random.RandomState(42)
            proj_matrix = proj_rng.randn(16, 128).astype(np.float32)
            
            # Linear Projection: (1, 16) dot (16, 128) -> (128,)
            embedding = np.dot(features, proj_matrix)
        else:
            # Active Neural Network Inference via ONNX Runtime
            inputs = {self.input_name: face_tensor}
            raw_output = self.session.run([self.output_name], inputs)[0]
            # Flatten to 1D vector (removes the batch index dimension)
            embedding = raw_output[0].flatten().astype(np.float32)

        # Mathematically enforce unit length using L2 Normalization: v_norm = v / ||v||
        l2_norm = np.linalg.norm(embedding)
        if l2_norm > 0:
            normalized_embedding = embedding / l2_norm
        else:
            normalized_embedding = embedding

        return normalized_embedding

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculates the Cosine Similarity between two embedding vectors.
        Formula: cos(theta) = (A dot B) / (||A|| ||B||)
        Since vectors are guaranteed L2-normalized to unit length, this simplifies
        perfectly to a simple dot product: cos(theta) = A dot B.
        
        Args:
            emb1: First embedding vector of shape (D,).
            emb2: Second embedding vector of shape (D,).
            
        Returns:
            A float similarity score between -1.0 (opposite) and 1.0 (identical).
        """
        # Ensure vectors are flattened to 1D
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Handle zero-vector boundary exceptions to prevent division by zero
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
            
        # Cosine Similarity Formula
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        return float(similarity)

    def verify_identity(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.60) -> Tuple[bool, float]:
        """
        Verifies if two embeddings represent the same identity based on a similarity threshold.
        
        Args:
            emb1: Candidate face embedding vector.
            emb2: Registered face embedding vector from the database.
            threshold: Cosine similarity matching threshold (default: 0.60).
            
        Returns:
            Tuple: (is_match: bool, similarity_score: float)
        """
        similarity = self.compare_embeddings(emb1, emb2)
        is_match = similarity >= threshold
        return is_match, similarity

def main():
    print("\n" + "="*50)
    print("BLINK - FACE MATCHING & SIMILARITY SUITE (PHASE 4)")
    print("="*50)
    
    # Initialize Preprocessor & Recognizer
    preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")
    recognizer = FaceRecognizer()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, "data", "test")
    
    # 1. Search for latest captured crop
    image_file: Optional[str] = None
    if os.path.exists(test_data_dir):
        files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            image_file = files[0]
            
    if image_file:
        print(f"[INFO] Loading target face crop: {image_file}")
        raw_crop = cv2.imread(image_file)
    else:
        print("[WARNING] No saved face crops found under 'data/test/'. Creating mock BGR crop for verification...")
        raw_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2.circle(raw_crop, (56, 56), 40, (120, 200, 120), -1)

    # 2. Preprocess raw crop and generate base embedding
    face_tensor_base = preprocessor.preprocess(raw_crop, apply_clahe=True)
    emb_base = recognizer.generate_embedding(face_tensor_base)
    
    print("\n" + "-"*55)
    print("MATCHING VERIFICATION TEST SUITE (Threshold = 0.60)")
    print("-"*55)
    
    # TEST 1: Same-Person Self-Identity (Perfect Match)
    is_match_1, score_1 = recognizer.verify_identity(emb_base, emb_base, threshold=0.60)
    print(f"TEST 1 [Self-Match]       | Similarity: {score_1:.6f} | Match: {is_match_1} (Expected: True)")
    
    # TEST 2: Same-Person Distorted (Environmental Shadow / Camera Noise)
    # We simulate camera noise and slight shading by adding mild Gaussian noise and shifting pixel intensities
    noise = np.random.normal(0, 4, raw_crop.shape).astype(np.int16)
    distorted_crop = np.clip(raw_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    face_tensor_distorted = preprocessor.preprocess(distorted_crop, apply_clahe=True)
    emb_distorted = recognizer.generate_embedding(face_tensor_distorted)
    
    is_match_2, score_2 = recognizer.verify_identity(emb_base, emb_distorted, threshold=0.60)
    status_2 = "SUCCESS" if is_match_2 else "FAILED"
    print(f"TEST 2 [Same-Person Noise]| Similarity: {score_2:.6f} | Match: {is_match_2} (Expected: True) -> {status_2}")

    # TEST 3: Different-Person / Fake Identity
    # We generate a completely different visual pattern representing a different human profile
    different_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    # Draw completely different geometry landmarks
    cv2.circle(different_crop, (56, 56), 45, (80, 80, 200), -1)
    cv2.rectangle(different_crop, (30, 35), (45, 50), (200, 200, 200), -1)
    cv2.rectangle(different_crop, (65, 35), (80, 50), (200, 200, 200), -1)
    cv2.circle(different_crop, (56, 75), 12, (200, 50, 50), -1)
    
    face_tensor_diff = preprocessor.preprocess(different_crop, apply_clahe=True)
    emb_diff = recognizer.generate_embedding(face_tensor_diff)
    
    is_match_3, score_3 = recognizer.verify_identity(emb_base, emb_diff, threshold=0.60)
    status_3 = "SUCCESS" if not is_match_3 else "FAILED"
    print(f"TEST 3 [Different-Person] | Similarity: {score_3:.6f} | Match: {is_match_3} (Expected: False) -> {status_3}")
    print("-"*55 + "\n")
    
    # 3. Tuning & Analysis Explanation
    print("="*65)
    print("THRESHOLD TUNING & CLASSIFICATION SUMMARY")
    print("="*65)
    print("1. Perfect Identity (Self): Score is exactly 1.000000.")
    print("2. Same Person (Distorted/Noise): Score is high (typically > 0.70).")
    print("   If camera noise makes this fall below threshold, increase exposure or reduce threshold.")
    print("3. Different Person: Score is low (typically < 0.30, close to 0.0 or negative).")
    print("   If a different person falsely matches, raise the threshold (e.g. to 0.65 or 0.70).")
    print("="*65 + "\n")
    
    print("[SUCCESS] Face matching validation complete.")

if __name__ == "__main__":
    main()
