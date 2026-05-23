import os
import cv2
import numpy as np
from typing import Tuple, Optional

class FacePreprocessor:
    """
    [Phase 2 - Face Preprocessing]
    Standardizes BGR face crops into high-fidelity normalized tensors (NCHW format).
    Includes adaptive contrast enhancement (CLAHE) to neutralize extreme lighting variations.
    """
    def __init__(self, target_size: Tuple[int, int] = (112, 112), norm_mode: str = "mobilefacenet"):
        """
        Initializes the preprocessor.
        
        Args:
            target_size: Desired dimensions for the face crop, standard is (112, 112).
            norm_mode: Normalization standard ('mobilefacenet' or 'range_0_1').
                       - 'mobilefacenet' maps [0, 255] to [-0.996, 0.996] via (x - 127.5) / 128.0
                       - 'range_0_1' maps [0, 255] to [0.0, 1.0]
        """
        self.target_size = target_size
        self.norm_mode = norm_mode.lower()
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) setup
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe_enhancement(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Applies adaptive local histogram equalization to standardise lighting.
        Operates purely in the LAB color space L-channel (luminance) to preserve true skin hue.
        
        Args:
            bgr_img: Raw BGR face crop image.
            
        Returns:
            Contrast-enhanced BGR face crop.
        """
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE on the Lightness channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge channels back and convert to BGR
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return bgr_enhanced

    def preprocess(self, face_crop: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
        """
        Full Preprocessing Pipeline:
        Raw Crop (BGR) -> Resizing -> CLAHE (Optional) -> RGB Conversion -> Range Normalization -> NCHW formatting.
        
        Args:
            face_crop: Raw input BGR image crop of a face.
            apply_clahe: Enable/disable adaptive contrast enhancement.
            
        Returns:
            Preprocessed normalized NumPy array of shape (1, 3, 112, 112) in float32.
        """
        # 1. Enforce strict sizing constraints
        h, w = face_crop.shape[:2]
        if (w, h) != self.target_size:
            face_crop = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)
            
        # 2. Lighting & Contrast Equalization
        processed_img = face_crop
        if apply_clahe:
            processed_img = self.apply_clahe_enhancement(face_crop)
            
        # 3. Channel Order Normalization (BGR to RGB)
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # 4. Numerical Range Alignment
        img_float = rgb_img.astype(np.float32)
        if self.norm_mode == "mobilefacenet":
            # Map [0, 255] to [-0.99609, 0.99609]
            normalized = (img_float - 127.5) / 128.0
        elif self.norm_mode == "range_0_1":
            # Map [0, 255] to [0.0, 1.0]
            normalized = img_float / 255.0
        else:
            raise ValueError(f"[ERROR] Unknown normalization mode: {self.norm_mode}")
            
        # 5. Tensor Dimension Formatting (HWC -> CHW and expand to NCHW batch shape)
        # Transpose channel dimension: (112, 112, 3) -> (3, 112, 112)
        tensor = np.transpose(normalized, (2, 0, 1))
        # Add batch dimension: (3, 112, 112) -> (1, 3, 112, 112)
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor

def main():
    print("\n" + "="*50)
    print("BLINK - FACE PREPROCESSING VERIFICATION (PHASE 2)")
    print("="*50)
    
    # Instantiate the preprocessor with MobileFaceNet specifications
    preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(base_dir, "data", "test")
    
    # 1. Locate latest cropped image from Phase 1
    image_file: Optional[str] = None
    if os.path.exists(test_data_dir):
        files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            # Sort by modified time to fetch the latest captured face
            files.sort(key=os.path.getmtime, reverse=True)
            image_file = files[0]
            
    if image_file:
        print(f"[INFO] Loading saved face crop: {image_file}")
        raw_crop = cv2.imread(image_file)
    else:
        print("[WARNING] No saved face crops found under 'data/test/'. Creating synthetic face crop for algorithm verification...")
        # Create a beautiful synthetic gradient block representing face contours to verify numerical stability
        raw_crop = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2.circle(raw_crop, (56, 56), 40, (180, 180, 180), -1) # Mock face silhouette
        cv2.circle(raw_crop, (40, 45), 6, (50, 50, 50), -1)      # Left eye
        cv2.circle(raw_crop, (72, 45), 6, (50, 50, 50), -1)      # Right eye
        cv2.ellipse(raw_crop, (56, 75), (20, 10), 0, 0, 180, (50, 50, 50), 3) # Smiling mouth
        # Inject artificial uneven shadow in the bottom half to test CLAHE
        raw_crop[56:, :] = (raw_crop[56:, :] * 0.4).astype(np.uint8)

    # 2. Run Preprocessing Pipeline
    # Apply standard MobileFaceNet pipeline (with CLAHE)
    tensor_with_clahe = preprocessor.preprocess(raw_crop, apply_clahe=True)
    # Run a pipeline without CLAHE for comparison
    tensor_without_clahe = preprocessor.preprocess(raw_crop, apply_clahe=False)
    
    # 3. Report Tensor Architecture Metrics
    print("\n" + "-"*35)
    print("Generated Tensor Profile:")
    print("-"*35)
    print(f"Tensor Shape       : {tensor_with_clahe.shape}")
    print(f"Data Type          : {tensor_with_clahe.dtype}")
    print(f"Minimum Value      : {np.min(tensor_with_clahe):.6f}")
    print(f"Maximum Value      : {np.max(tensor_with_clahe):.6f}")
    print(f"Mean Value         : {np.mean(tensor_with_clahe):.6f}")
    print("-"*35 + "\n")
    
    # 4. Generate Visual Side-by-Side Comparison
    raw_resized = cv2.resize(raw_crop, (112, 112))
    clahe_bgr = preprocessor.apply_clahe_enhancement(raw_resized)
    
    # Create combined display canvas
    canvas = np.hstack((raw_resized, clahe_bgr))
    
    # Overlay Labels
    cv2.putText(canvas, "RAW FACE CROP", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CLAHE ENHANCED", (117, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Display double scale window for comfortable viewing
    display_canvas = cv2.resize(canvas, (448, 224), interpolation=cv2.INTER_NEAREST)
    print("[INFO] Displaying preprocessing comparison window. Press any key in the window to exit...")
    cv2.imshow("BLINK Preprocessing Comparison (Raw vs CLAHE)", display_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[SUCCESS] Preprocessing pipeline verified.")

if __name__ == "__main__":
    main()
