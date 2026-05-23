import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

# Safe environment-robust MediaPipe import
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except ImportError:
    import mediapipe.solutions.face_mesh as mp_face_mesh

class LivenessDetector:
    """
    [Phase 7 - Liveness Detection]
    Tracks facial landmarks in real time using MediaPipe Face Mesh.
    Calculates Eye Aspect Ratio (EAR) for blink detection and nose-yaw ratio for head turns.
    """
    def __init__(self, ear_threshold_close: float = 0.18, ear_threshold_open: float = 0.22):
        """
        Initializes Face Mesh and liveness state machine variables.
        """
        self.mp_face_mesh = mp_face_mesh
        # Refine landmarks enables high-precision iris/eye contour mesh points
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Blink state tracking parameters
        self.ear_threshold_close = ear_threshold_close
        self.ear_threshold_open = ear_threshold_open
        self.blink_state = "OPEN"  # Current state: OPEN or CLOSED
        self.blink_count = 0
        
        # Landmark index configuration (MediaPipe Face Mesh structure)
        self.left_eye_horizontal = (33, 133)
        self.left_eye_vertical1 = (160, 144)
        self.left_eye_vertical2 = (158, 153)
        
        self.right_eye_horizontal = (362, 263)
        self.right_eye_vertical1 = (385, 380)
        self.right_eye_vertical2 = (387, 373)
        
        # Nose and cheek markers for yaw estimation
        self.nose_tip = 1
        self.left_cheek = 234
        self.right_cheek = 454

    def extract_landmarks(self, frame: np.ndarray) -> Optional[Any]:
        """
        Grabs raw BGR frame and extracts Face Mesh landmarks.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

    def calculate_single_eye_ear(self, landmarks: Any, horizontal: Tuple[int, int], vert1: Tuple[int, int], vert2: Tuple[int, int]) -> float:
        """
        Computes Eye Aspect Ratio (EAR) for a single eye outline.
        Formula: EAR = (||v1_1 - v1_2|| + ||v2_1 - v2_2||) / (2 * ||h1 - h2||)
        """
        p_h1 = landmarks[horizontal[0]]
        p_h2 = landmarks[horizontal[1]]
        
        p_v1_1 = landmarks[vert1[0]]
        p_v1_2 = landmarks[vert1[1]]
        
        p_v2_1 = landmarks[vert2[0]]
        p_v2_2 = landmarks[vert2[1]]
        
        # Euclidean distance calculations
        d_h = np.sqrt((p_h1.x - p_h2.x)**2 + (p_h1.y - p_h2.y)**2)
        d_v1 = np.sqrt((p_v1_1.x - p_v1_2.x)**2 + (p_v1_1.y - p_v1_2.y)**2)
        d_v2 = np.sqrt((p_v2_1.x - p_v2_2.x)**2 + (p_v2_1.y - p_v2_2.y)**2)
        
        if d_h == 0:
            return 0.0
            
        return float((d_v1 + d_v2) / (2.0 * d_h))

    def calculate_ear(self, landmarks: Any) -> Tuple[float, float, float]:
        """
        Computes Left EAR, Right EAR, and average EAR.
        """
        left_ear = self.calculate_single_eye_ear(
            landmarks, self.left_eye_horizontal, self.left_eye_vertical1, self.left_eye_vertical2
        )
        right_ear = self.calculate_single_eye_ear(
            landmarks, self.right_eye_horizontal, self.right_eye_vertical1, self.right_eye_vertical2
        )
        avg_ear = (left_ear + right_ear) / 2.0
        return left_ear, right_ear, avg_ear

    def detect_blink(self, landmarks: Any) -> bool:
        """
        State Machine Blink Detection.
        State OPEN -> CLOSED (EAR < 0.18)
        State CLOSED -> OPEN (EAR > 0.22) => Increment count and return True.
        
        Returns:
            Boolean indicating a fresh blink event was registered.
        """
        _, _, avg_ear = self.calculate_ear(landmarks)
        blink_registered = False
        
        if self.blink_state == "OPEN":
            if avg_ear < self.ear_threshold_close:
                self.blink_state = "CLOSED"
        elif self.blink_state == "CLOSED":
            if avg_ear > self.ear_threshold_open:
                self.blink_state = "OPEN"
                self.blink_count += 1
                blink_registered = True
                
        return blink_registered

    def detect_head_turn(self, landmarks: Any) -> str:
        """
        Tracks horizontal cheek-to-nose width ratios to estimate pose yaw directions.
        Returns:
            String direction ('LEFT', 'RIGHT', or 'CENTER')
        """
        p_nose = landmarks[self.nose_tip]
        p_left = landmarks[self.left_cheek]
        p_right = landmarks[self.right_cheek]
        
        # Calculate horizontal distances (normalized x plane)
        d_left = abs(p_nose.x - p_left.x)
        d_right = abs(p_right.x - p_nose.x)
        
        if d_right == 0:
            return "CENTER"
            
        yaw_ratio = d_left / d_right
        
        # Threshold configurations for head yaw orientation
        if yaw_ratio < 0.42:
            return "LEFT"
        elif yaw_ratio > 2.38:
            return "RIGHT"
        else:
            return "CENTER"
