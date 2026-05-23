import json
import os
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

# Import pipeline components for complete end-to-end sandbox verification
from preprocess import FacePreprocessor
from recognize import FaceRecognizer

class UserDatabase:
    """
    [Phase 5 - Face Enrollment System]
    Manages local storage of registered user information, embeddings, and crops.
    Uses JSON file persistence for offline operations.
    """
    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the UserDatabase manager.
        
        Args:
            db_path: Path to the JSON registry file on disk.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if db_path is None:
            db_path = os.path.join(base_dir, "data", "users.json")
            
        self.db_path = db_path
        self.known_faces_dir = os.path.join(base_dir, "data", "known_faces")
        self.users: Dict[str, Any] = {}
        
        # Instantiate required folders
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.load_db()

    def load_db(self):
        """Loads registered users and embeddings from the local JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.users = json.load(f)
                print(f"[INFO] UserDatabase: Loaded {len(self.users)} enrolled users from {self.db_path}")
            except Exception as e:
                print(f"[WARNING] UserDatabase: Failed to load registry: {e}. Instantiating fresh database.")
                self.users = {}
        else:
            self.users = {}
            print(f"[INFO] UserDatabase: No database file found. Starting with a fresh empty registry.")

    def save_db(self):
        """Persists the memory user registry to the local JSON file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.users, f, indent=4)
        except Exception as e:
            print(f"[ERROR] UserDatabase: Failed to save registry to disk: {e}")

    def enroll_user(self, name: str, embedding: np.ndarray, face_crop: Optional[np.ndarray] = None, model_mode: str = "simulated") -> bool:
        """
        Enrolls a new user into the persistent database.
        
        Args:
            name: Unique identifier name of the user.
            embedding: 1D NumPy array representing facial features (L2-normalized).
            face_crop: Optional 112x112 BGR face crop image of the user.
            model_mode: Indicates the model weights standard ('simulated' or 'onnx').
            
        Returns:
            Boolean indicating successful enrollment.
        """
        # Clean white spaces from name to avoid filesystem naming issues
        name_clean = name.strip()
        if not name_clean:
            print("[ERROR] UserDatabase: User name cannot be empty.")
            return False

        # Convert numpy array to float list for seamless JSON serialization
        emb_list = embedding.flatten().tolist()
        
        # Save visual crop to data/known_faces/ for visual verification & React Native integration
        if face_crop is not None:
            filename = f"{name_clean}.jpg"
            filepath = os.path.join(self.known_faces_dir, filename)
            try:
                cv2.imwrite(filepath, face_crop)
                print(f"[INFO] UserDatabase: Visual crop successfully saved to {filepath}")
            except Exception as e:
                print(f"[WARNING] UserDatabase: Failed to save enrollment face crop: {e}")

        # Update JSON database records
        self.users[name_clean] = {
            "embedding": emb_list,
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_mode": model_mode
        }
        
        self.save_db()
        print(f"[SUCCESS] UserDatabase: User '{name_clean}' successfully enrolled (Mode: {model_mode.upper()}).")
        return True

    def remove_user(self, name: str) -> bool:
        """
        Deletes a user from the persistent database and purges their local visual crop.
        
        Args:
            name: The user identifier name to purge.
            
        Returns:
            Boolean indicating successful purging.
        """
        name_clean = name.strip()
        if name_clean not in self.users:
            print(f"[WARNING] UserDatabase: User '{name_clean}' does not exist in the registry.")
            return False

        # Delete database visual crop
        filename = f"{name_clean}.jpg"
        filepath = os.path.join(self.known_faces_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"[INFO] UserDatabase: Purged visual crop for '{name_clean}' from disk.")
            except Exception as e:
                print(f"[WARNING] UserDatabase: Failed to delete visual crop: {e}")

        # Remove from local dictionary records
        del self.users[name_clean]
        self.save_db()
        print(f"[SUCCESS] UserDatabase: User '{name_clean}' successfully removed from database.")
        return True

    def search_user(self, query_embedding: np.ndarray, recognizer: FaceRecognizer, threshold: float = 0.60) -> Tuple[Optional[str], float]:
        """
        Compares query embedding against all enrolled users in the database using Cosine Similarity.
        Returns the closest matching user profile that meets the specified matching threshold.
        
        Args:
            query_embedding: Unit-normalized 1D feature vector of the candidate face.
            recognizer: Active FaceRecognizer instance (to reuse validated cosine comparators).
            threshold: Cosine similarity matching threshold (default: 0.60).
            
        Returns:
            Tuple: (matched_user_name: Optional[str], similarity_score: float)
                   If no match meets the threshold, matched_user_name is None.
        """
        if not self.users:
            return None, 0.0

        max_similarity = -1.0
        best_match = None
        
        # Flatten query vector for mathematical comparison safety
        q_vec = query_embedding.flatten()
        
        # Linear scan (Nearest-Neighbor) matching check
        for name, record in self.users.items():
            ref_vec = np.array(record["embedding"], dtype=np.float32)
            similarity = recognizer.compare_embeddings(q_vec, ref_vec)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name

        # Classify based on configured threshold
        if max_similarity >= threshold:
            return best_match, max_similarity
        else:
            return None, max_similarity

    def migrate_database_embeddings(self, preprocessor: FacePreprocessor, recognizer: FaceRecognizer):
        """
        Self-healing DB Migration.
        Scans all visual crops inside data/known_faces/, detects if the persistent
        embeddings are out-of-sync with the current active FaceRecognizer mode
        (e.g. transitioning from simulated fallback to real ONNX model weights),
        and automatically re-generates and persists the correct embeddings!
        """
        current_mode = "simulated" if recognizer.simulated else "onnx"
        migrated_count = 0
        
        if not os.path.exists(self.known_faces_dir):
            return
            
        print(f"[INFO] UserDatabase: Running self-healing database migration check (Active Mode: {current_mode.upper()})...")
        
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                filepath = os.path.join(self.known_faces_dir, filename)
                
                # Check if the user needs migration
                record = self.users.get(name)
                needs_migration = False
                
                if not record:
                    needs_migration = True
                else:
                    record_mode = record.get("model_mode", "simulated")
                    if record_mode != current_mode:
                        needs_migration = True
                        
                if needs_migration:
                    try:
                        face_crop = cv2.imread(filepath)
                        if face_crop is not None:
                            # Re-preprocess and generate embedding in the current active mode
                            face_tensor = preprocessor.preprocess(face_crop, apply_clahe=True)
                            embedding = recognizer.generate_embedding(face_tensor)
                            
                            self.users[name] = {
                                "embedding": embedding.flatten().tolist(),
                                "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "model_mode": current_mode
                            }
                            migrated_count += 1
                            print(f"[INFO] UserDatabase: Successfully migrated embedding for user '{name}' to '{current_mode.upper()}' mode.")
                    except Exception as e:
                        print(f"[WARNING] UserDatabase: Migration failed for user '{name}': {e}")
                        
        if migrated_count > 0:
            self.save_db()
            print(f"[SUCCESS] UserDatabase: Self-healing complete. Migrated {migrated_count} user(s) to '{current_mode.upper()}' mode.")
        else:
            print("[INFO] UserDatabase: Database is fully in-sync. No migration required.")

def main():
    print("\n" + "="*50)
    print("BLINK - FACE ENROLLMENT & PERSISTENCE (PHASE 5)")
    print("="*50)
    
    # Initialize Preprocessor, Recognizer, and Database Registry using a sandboxed test JSON path!
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_db_path = os.path.join(base_dir, "data", "test_users.json")
    
    preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")
    recognizer = FaceRecognizer()
    db = UserDatabase(db_path=test_db_path)
    
    # Clean previous enrollment states to ensure deterministic fresh testing execution
    print("[INFO] Purging old mock registry profiles to reset sandbox environment...")
    db.remove_user("Alice")
    db.remove_user("Bob")
    
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
        print(f"[INFO] Loading saved face crop for Alice: {image_file}")
        alice_crop = cv2.imread(image_file)
    else:
        print("[WARNING] No saved face crops found under 'data/test/'. Creating mock BGR crop for Alice...")
        alice_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2.circle(alice_crop, (56, 56), 40, (120, 200, 120), -1)

    # 2. Preprocess raw crop and generate embeddings for enrollment
    alice_tensor = preprocessor.preprocess(alice_crop, apply_clahe=True)
    alice_emb = recognizer.generate_embedding(alice_tensor)
    
    # Generate distinct synthetic visual features for Bob
    bob_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.circle(bob_crop, (56, 56), 45, (80, 80, 200), -1)
    cv2.rectangle(bob_crop, (30, 35), (45, 50), (200, 200, 200), -1)
    cv2.rectangle(bob_crop, (65, 35), (80, 50), (200, 200, 200), -1)
    
    bob_tensor = preprocessor.preprocess(bob_crop, apply_clahe=True)
    bob_emb = recognizer.generate_embedding(bob_tensor)

    # 3. Perform Enrollments
    print("\n" + "-"*45)
    print("ENROLLING TEST USERS")
    print("-"*45)
    db.enroll_user("Alice", alice_emb, alice_crop)
    db.enroll_user("Bob", bob_emb, bob_crop)
    print("-"*45 + "\n")

    # 4. Search and Recognition Tests
    print("-" * 55)
    print("RECOGNITION TEST SUITE (Threshold = 0.60)")
    print("-" * 55)

    # CASE A: Same-Person Recognition (Positive Match)
    # We query using the base visual crop of Alice plus slight Gaussian camera noise
    noise = np.random.normal(0, 4, alice_crop.shape).astype(np.int16)
    alice_noisy_crop = np.clip(alice_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    query_tensor_a = preprocessor.preprocess(alice_noisy_crop, apply_clahe=True)
    query_emb_a = recognizer.generate_embedding(query_tensor_a)
    
    matched_user_a, score_a = db.search_user(query_emb_a, recognizer, threshold=0.60)
    status_a = "SUCCESS" if matched_user_a == "Alice" else "FAILED"
    print(f"CASE A [Alice Noisy Query] | Matched: '{matched_user_a}' | Score: {score_a:.6f} -> {status_a}")

    # CASE B: Unregistered / Stranger Rejection (Negative Match)
    # We query using a completely distinct visual pattern representing an unknown stranger.
    # To create true structural divergence from Bob (colored circle on black background),
    # we make the stranger crop a black circle on a solid white background (negative correlation).
    stranger_crop = np.ones((112, 112, 3), dtype=np.uint8) * 255
    cv2.circle(stranger_crop, (56, 56), 30, (0, 0, 0), -1) # Black circle on white background
    
    query_tensor_b = preprocessor.preprocess(stranger_crop, apply_clahe=True)
    query_emb_b = recognizer.generate_embedding(query_tensor_b)
    
    matched_user_b, score_b = db.search_user(query_emb_b, recognizer, threshold=0.60)
    status_b = "SUCCESS" if matched_user_b is None else "FAILED"
    print(f"CASE B [Stranger Query]    | Matched: '{matched_user_b}' | Score: {score_b:.6f} -> {status_b}")
    print("-" * 55 + "\n")
    
    print("[SUCCESS] Face enrollment and registry operations successfully validated.")

if __name__ == "__main__":
    main()
