import os
import cv2
import numpy as np
import time
from typing import Tuple, List, Optional, Dict, Any

# Safe, environment-robust MediaPipe solutions import
try:
    from mediapipe.python.solutions import face_detection as mp_face_detection
except ImportError:
    import mediapipe.solutions.face_detection as mp_face_detection

# Import modular helper utilities
from utils import FPSCalculator, ensure_directories

class FaceDetector:
    """
    Modular Face Detector wrapper for MediaPipe Face Detection.
    Provides utility methods to extract, pad, crop, and resize face regions.
    """
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        """
        Initializes the MediaPipe Face Detection engine.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for detection to be considered successful.
            model_selection: 0 for short-range faces (within 2 meters from camera), 
                             1 for full-range faces (within 5 meters).
        """
        self.mp_face_detection = mp_face_detection
        # We don't use mp.solutions.drawing_utils because we want beautiful, custom edge-AI styling.
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs face detection on an incoming BGR frame.
        
        Args:
            frame: NumPy BGR image frame from the webcam.
            
        Returns:
            A list of dictionaries representing detected faces.
            Each dictionary contains:
                'bbox': (xmin, ymin, width, height) in pixel coordinates
                'score': float confidence score
                'landmarks': List of dicts with keys 'x', 'y' for landmarks
        """
        h, w, _ = frame.shape
        # MediaPipe requires RGB images. OpenCV reads images in BGR format.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bbox_relative = detection.location_data.relative_bounding_box
                
                # Convert normalized relative coordinates to absolute pixel coordinates
                xmin = int(bbox_relative.xmin * w)
                ymin = int(bbox_relative.ymin * h)
                width = int(bbox_relative.width * w)
                height = int(bbox_relative.height * h)
                
                # Retrieve standard landmarks (right eye, left eye, nose tip, mouth center, right ear tragus, left ear tragus)
                landmarks = []
                for keypoint in detection.location_data.relative_keypoints:
                    landmarks.append({
                        'x': int(keypoint.x * w),
                        'y': int(keypoint.y * h)
                    })
                
                detections.append({
                    'bbox': (xmin, ymin, width, height),
                    'score': detection.score[0] if detection.score else 0.0,
                    'landmarks': landmarks
                })
        
        return detections

    def crop_and_align(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], landmarks: Optional[List[Dict[str, int]]] = None, padding_ratio: float = 0.15, target_size: Tuple[int, int] = (112, 112)) -> Optional[np.ndarray]:
        """
        Extracts, pads, and resizes a face crop from the BGR frame.
        Applies rotation alignment using eye landmarks if available to keep the face horizontal.
        
        Args:
            frame: The original BGR frame.
            bbox: (xmin, ymin, width, height) bounding box coordinates.
            landmarks: Optional face landmarks list containing right eye and left eye.
            padding_ratio: Additional margins to pad the crop around the bounding box (defaults to 15%).
            target_size: Target tuple (width, height) to resize to. Recommended standard is (112, 112).
            
        Returns:
            A normalized 112x112 BGR cropped face image, or None if crop bounds are invalid.
        """
        h, w, _ = frame.shape

        # If landmarks are provided, apply rotation-based face alignment
        if landmarks and len(landmarks) >= 2:
            try:
                # Landmark 0 = Right eye (usually left on image), Landmark 1 = Left eye (usually right on image)
                right_eye = landmarks[0]
                left_eye = landmarks[1]
                
                # Calculate coordinates
                dy = left_eye['y'] - right_eye['y']
                dx = left_eye['x'] - right_eye['x']
                
                # Compute angle to rotate (in degrees)
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Midpoint between the eyes is the center of rotation
                eye_center = (
                    int((right_eye['x'] + left_eye['x']) / 2),
                    int((right_eye['y'] + left_eye['y']) / 2)
                )
                
                # Build the affine rotation matrix
                rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                
                # Warp the entire frame to align the face
                frame = cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
            except Exception as e:
                print(f"[WARNING] Face alignment rotation failed: {e}")

        xmin, ymin, width, height = bbox
        
        # Apply scaling margins to the bounding box to capture complete head profiles for alignment/liveness
        x_pad = int(width * padding_ratio)
        y_pad = int(height * padding_ratio)
        
        # Calculate new padded bounds
        x1 = max(0, xmin - x_pad)
        y1 = max(0, ymin - y_pad)
        x2 = min(w, xmin + width + x_pad)
        y2 = min(h, ymin + height + y_pad)
        
        # Check if the calculated crop is valid and has positive area
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None
            
        crop = frame[y1:y2, x1:x2]
        
        # Resize crop using INTER_AREA interpolation, which is optimal for downsampling
        resized_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        return resized_crop

    def draw_premium_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], score: float):
        """
        Draws a custom, high-tech premium bounding box with corner highlights.
        
        Args:
            frame: Frame to draw on.
            bbox: Bounding box tuple (xmin, ymin, width, height).
            score: Face detection confidence score.
        """
        xmin, ymin, width, height = bbox
        xmax, ymax = xmin + width, ymin + height
        
        # Sleek UI design palette: Neon Cyan highlight
        primary_color = (255, 255, 0)  # Cyan (BGR format: Blue=255, Green=255, Red=0)
        thickness = 1
        corner_len = min(20, int(width * 0.2))
        corner_thickness = 3
        
        # 1. Draw standard background bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (80, 80, 80), thickness, lineType=cv2.LINE_AA)
        
        # 2. Draw stylish dynamic corner brackets
        # Top-Left Corner
        cv2.line(frame, (xmin, ymin), (xmin + corner_len, ymin), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_len), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        # Top-Right Corner
        cv2.line(frame, (xmax, ymin), (xmax - corner_len, ymin), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_len), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        # Bottom-Left Corner
        cv2.line(frame, (xmin, ymax), (xmin + corner_len, ymax), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_len), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        # Bottom-Right Corner
        cv2.line(frame, (xmax, ymax), (xmax - corner_len, ymax), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_len), primary_color, corner_thickness, lineType=cv2.LINE_AA)
        
        # 3. Draw soft text background & status label
        score_text = f"FACE {score*100:.1f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.4
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(score_text, font, font_scale, text_thickness)
        
        # Put background tag above the face box if space allows, otherwise put it inside
        y_text = ymin - 10 if ymin - 10 > text_h else ymin + text_h + 10
        cv2.rectangle(frame, (xmin, y_text - text_h - 4), (xmin + text_w + 8, y_text + baseline), primary_color, -1)
        cv2.putText(frame, score_text, (xmin + 4, y_text - 2), font, font_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)


def main():
    # Make sure core environment directories are instantiated
    ensure_directories()
    
    # Initialize detector and FPS counters
    detector = FaceDetector(min_detection_confidence=0.6, model_selection=0)
    fps_calc = FPSCalculator(window_size=20)
    
    print("[INFO] Initializing webcam...")
    # Attempt opening system default camera
    cap = cv2.VideoCapture(0)
    
    # Optimize OpenCV camera properties to minimize ingestion latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce frame buffer delay for real-time responsiveness
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam index 0. Checking standard fallbacks...")
        # Check standard fallback camera indexes
        for idx in range(1, 4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[INFO] Successfully opened camera index {idx}")
                break
        else:
            print("[FATAL ERROR] No webcam devices found. Exiting.")
            return

    # Instruction Panel in Terminal
    print("\n" + "="*50)
    print("BLINK - REAL-TIME FACE DETECTION (PHASE 1)")
    print("="*50)
    print("Webcam successfully connected.")
    print("Controls:")
    print("  [s] - Capture and save cropped 112x112 face image to 'data/test/'")
    print("  [q] - Quit application")
    print("="*50 + "\n")

    latest_crop: Optional[np.ndarray] = None
    save_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame received from camera stream. Retrying...")
            continue
        
        # Mirror frame for intuitive real-time selfie view
        frame = cv2.flip(frame, 1)
        
        # Run detection pipeline
        detections = detector.detect_faces(frame)
        
        # Process and track the single closest/largest face for robust edge-authentication
        if detections:
            # Sort detections by size (width * height) so we focus on the most prominent person in the foreground
            detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
            primary_face = detections[0]
            bbox = primary_face['bbox']
            score = primary_face['score']
            
            # Crop the face and save to buffer for manual capture with rotation alignment
            latest_crop = detector.crop_and_align(frame, bbox, landmarks=primary_face.get('landmarks'), padding_ratio=0.15)
            
            # Draw primary beautiful modern box
            detector.draw_premium_bbox(frame, bbox, score)
            
            # Optionally draw secondary light boxes for other detected background faces
            for background_face in detections[1:]:
                bg_bbox = background_face['bbox']
                cv2.rectangle(frame, (bg_bbox[0], bg_bbox[1]), (bg_bbox[0]+bg_bbox[2], bg_bbox[1]+bg_bbox[3]), (0, 150, 0), 1, lineType=cv2.LINE_AA)

        # Update FPS
        fps = fps_calc.update()
        
        # Overlay premium stats card on the feed
        cv2.rectangle(frame, (10, 10), (190, 60), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (190, 60), (100, 100, 100), 1)
        cv2.putText(frame, f"System: BLINK AI Engine", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if fps > 20 else (0, 165, 255), 1, cv2.LINE_AA)

        # Pip Overlay: If a face is detected, overlay the 112x112 face crop in the top-right corner of the window
        if latest_crop is not None:
            ch, cw, _ = frame.shape
            pip_size = 112
            # Target region: top-right with 10px margin
            x_offset = cw - pip_size - 10
            y_offset = 10
            
            # Draw background & border for the PiP window
            cv2.rectangle(frame, (x_offset - 2, y_offset - 2), (x_offset + pip_size + 2, y_offset + pip_size + 2), (255, 255, 0), 2)
            frame[y_offset:y_offset+pip_size, x_offset:x_offset+pip_size] = latest_crop
            cv2.putText(frame, "CROP 112x112", (x_offset, y_offset + pip_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

        # Render output
        cv2.imshow("BLINK - Face Detection Stream", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("[INFO] Terminating stream...")
            break
            
        elif key == ord('s'):
            if latest_crop is not None:
                save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")
                timestamp = int(time.time())
                filename = f"crop_{timestamp}_{save_counter}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                cv2.imwrite(filepath, latest_crop)
                print(f"[SUCCESS] Cropped 112x112 face saved successfully to: {filepath}")
                save_counter += 1
                
                # Visual flash feedback in console
                print(f"--- Captured crop #{save_counter} ---")
            else:
                print("[WARNING] Cannot capture. No face detected currently.")

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System resources released successfully.")

if __name__ == "__main__":
    main()
