import os
import cv2
import numpy as np
import time
from collections import deque, Counter
from typing import Optional, Tuple

# Import pipeline components
from detect import FaceDetector
from preprocess import FacePreprocessor
from recognize import FaceRecognizer
from database import UserDatabase
from liveness import LivenessDetector
from utils import FPSCalculator, StageTimer, ensure_directories
from inference_worker import InferenceWorker

def draw_premium_bracket_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], label: str, is_identified: bool, score: Optional[float] = None):
    """
    Draws custom premium visual bounding brackets and status headers on the webcam frame.
    Color codes: Neon Green (Access Granted) and Neon Red (Access Denied).
    """
    xmin, ymin, width, height = bbox
    xmax, ymax = xmin + width, ymin + height
    
    # 1. Select Color Palette based on matching identity
    if is_identified:
        primary_color = (0, 255, 0)      # Neon Green (BGR: Blue=0, Green=255, Red=0)
        bg_card_color = (0, 120, 0)      # Deep Olive Green
        text_color = (255, 255, 255)     # White text
        border_thickness = 2
    else:
        primary_color = (0, 0, 255)      # Neon Crimson Red (BGR: Blue=0, Green=0, Red=255)
        bg_card_color = (0, 0, 150)      # Deep Maroon Red
        text_color = (255, 255, 255)     # White text
        border_thickness = 1

    corner_len = min(22, int(width * 0.2))
    corner_thickness = 4
    
    # 2. Draw standard background bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (80, 80, 80), 1, lineType=cv2.LINE_AA)
    
    # 3. Draw stylish dynamic corner brackets
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
    
    # 4. Draw premium visual status tag banner above/inside the brackets
    score_suffix = f" ({score*100:.1f}%)" if score is not None else ""
    status_text = f"{label}{score_suffix}"
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(status_text, font, font_scale, text_thickness)
    
    y_text = ymin - 10 if ymin - 10 > text_h else ymin + text_h + 10
    
    # Draw soft round tag background card
    cv2.rectangle(frame, (xmin, y_text - text_h - 5), (xmin + text_w + 10, y_text + baseline), bg_card_color, -1)
    cv2.rectangle(frame, (xmin, y_text - text_h - 5), (xmin + text_w + 10, y_text + baseline), primary_color, 1)
    cv2.putText(frame, status_text, (xmin + 5, y_text - 2), font, font_scale, text_color, text_thickness, lineType=cv2.LINE_AA)

def main():
    # Make sure core environment directories are instantiated
    ensure_directories()
    
    print("\n" + "="*60)
    print("BLINK - REAL-TIME FACE RECOGNITION ORCHESTRATOR (PHASE 8)")
    print("  Optimizations: INT8 Model + Background Inference + Frame-Skip")
    print("="*60)

    # ── Phase 8 Performance Config ────────────────────────────────────────────
    # Run ONNX inference every N frames in the background thread.
    # Reduces inference CPU load by N× while display stays smooth.
    # On mobile this maps to the TFLite delegate scheduling budget.
    INFERENCE_EVERY_N_FRAMES: int  = 2

    # Set True to print a per-stage latency table to the console every 120 frames.
    # Use this to benchmark the pipeline before porting to React Native.
    ENABLE_PROFILER: bool = False

    # ── Initialize all individual pipelines ──────────────────────────────────
    print("[INFO] Loading Face Detection Engine...")
    detector = FaceDetector(min_detection_confidence=0.6, model_selection=0)
    
    print("[INFO] Loading Face Preprocessor Pipeline...")
    preprocessor = FacePreprocessor(target_size=(112, 112), norm_mode="mobilefacenet")
    
    print("[INFO] Loading ONNX Face Recognition Model (INT8 preferred)...")
    recognizer = FaceRecognizer()
    
    print("[INFO] Connecting to User Database...")
    db = UserDatabase()
    # Trigger dynamic self-healing database migration check to sync embeddings automatically
    db.migrate_database_embeddings(preprocessor, recognizer)
    
    print("[INFO] Initializing Landmark-based Liveness Detector...")
    liveness = LivenessDetector()

    # ── Phase 8: Background Inference Worker ──────────────────────────────────
    print("[INFO] Starting Background Inference Worker thread...")
    worker = InferenceWorker(preprocessor, recognizer, db, threshold=0.60)
    worker.start()

    # ── Phase 8: Latency Profiler ─────────────────────────────────────────────
    stage_timer = StageTimer(window_size=60)

    fps_calc = FPSCalculator(window_size=20)
    frame_count: int = 0

    # ── Establish camera connection ───────────────────────────────────────────
    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Optimize OpenCV camera properties to minimize ingestion latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam index 0. Checking standard fallbacks...")
        for idx in range(1, 4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[INFO] Successfully opened camera index {idx}")
                break
        else:
            print("[FATAL ERROR] No webcam devices found. Exiting.")
            worker.stop()
            return

    # Instruction Panel in Terminal
    print("\n" + "="*50)
    print("BLINK CAMERA STREAM ACTIVE")
    print("="*50)
    print("Controls:")
    print("  [e] - ENROLL CURRENT TRACKED FACE (LIVE REGISTRATION)")
    print("  [p] - TOGGLE LATENCY PROFILER (prints stage ms table)")
    print("  [q] - Safe close and exit application")
    print("="*50 + "\n")

    # ── Liveness State Variables ──────────────────────────────────────────────
    auth_state = "LIVENESS_BLINK"   # LIVENESS_BLINK, LIVENESS_TURN, AUTHORIZED
    blink_target = 1
    head_turn_target = "RIGHT"
    last_face_time = time.time()
    
    latest_crop: Optional[np.ndarray] = None
    latest_embedding: Optional[np.ndarray] = None
    
    # --- Temporal Smoothing Buffer ---
    # Stores the last N raw predictions. Majority vote determines the stable displayed result.
    SMOOTHING_WINDOW = 12        # Number of frames to look back
    MAJORITY_THRESHOLD = 0.50   # Fraction of votes needed to confirm an identity switch
    prediction_buffer: deque = deque(maxlen=SMOOTHING_WINDOW)
    
    is_identified = False
    matched_name = "UNKNOWN"
    match_score = 0.0
    stable_name = "UNKNOWN"
    stable_score = 0.0
    stable_identified = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame received from camera stream. Retrying...")
            continue
        
        frame_count += 1
        
        # Mirror frame for intuitive real-time selfie view
        frame = cv2.flip(frame, 1)
        
        # ── Stage: Face Detection ─────────────────────────────────────
        with stage_timer.measure("detect_faces"):
            detections = detector.detect_faces(frame)
        
        # ── Stage: Liveness Landmark Extraction ───────────────────────
        with stage_timer.measure("liveness_mesh"):
            landmarks = liveness.extract_landmarks(frame)
        
        # Reset detection indicators
        current_crop = None
        current_emb = None
        
        if detections:
            # Sort detections by size (width * height) so we focus on the most prominent person in the foreground
            detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
            primary_face = detections[0]
            bbox = primary_face['bbox']
            
            # Cache visual update timestamp to track lost frame resets
            last_face_time = time.time()
            
            # ── Stage: Crop & Align ──────────────────────────────────────
            with stage_timer.measure("crop_align"):
                current_crop = detector.crop_and_align(frame, bbox, padding_ratio=0.15)
            
            if current_crop is not None:
                # Cache crop for display/enrollment
                latest_crop = current_crop.copy()

                # ── Phase 8: Submit crop to background inference worker ───
                # Only submit every N frames to reduce CPU load.
                # Worker always processes the LATEST crop; stale frames are dropped.
                if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                    worker.submit_crop(current_crop)

                # ── Read the latest stable result from the worker ─────────
                # This is always non-blocking — uses whatever the worker computed last.
                with stage_timer.measure("get_result"):
                    raw_name, raw_score, worker_emb = worker.get_result()

                if worker_emb is not None:
                    latest_embedding = worker_emb

                # Push raw result into the temporal smoothing buffer
                prediction_buffer.append((raw_name, raw_score))
                
                # --- Majority Vote Smoothing ---
                # Count how many times each name appears in the rolling window
                name_votes = Counter(label for label, _ in prediction_buffer)
                top_name, top_votes = name_votes.most_common(1)[0]
                vote_fraction = top_votes / len(prediction_buffer)
                
                if vote_fraction >= MAJORITY_THRESHOLD:
                    # Stable winner — update the displayed identity
                    stable_name = top_name
                    stable_identified = (top_name != "UNKNOWN")
                    # Use the average score of the winning name's frames for display
                    winning_scores = [s for lbl, s in prediction_buffer if lbl == top_name]
                    stable_score = sum(winning_scores) / len(winning_scores)
                # else: buffer is split — keep showing last stable result (no flicker)
                
                is_identified = stable_identified
                matched_name = stable_name
                match_score = stable_score
            
            # --- EVALUATE ACTIVE LIVENESS STATE MACHINE ---
            if landmarks is not None:
                # 1. Update blink tracking state
                liveness.detect_blink(landmarks)
                # 2. Get pose yaw estimation direction
                current_turn = liveness.detect_head_turn(landmarks)
                
                if auth_state == "LIVENESS_BLINK":
                    # Display Step 1 status tag on face brackets
                    display_tag = f"LIVENESS: PLEASE BLINK ({liveness.blink_count}/{blink_target})"
                    draw_premium_bracket_box(frame, bbox, display_tag, is_identified=False)
                    
                    if liveness.blink_count >= blink_target:
                        auth_state = "LIVENESS_TURN"
                        print("[INFO] Liveness: Eye blink successfully verified! Moving to pose direction check.")
                        
                elif auth_state == "LIVENESS_TURN":
                    # Display Step 2 status tag on face brackets
                    display_tag = f"LIVENESS: PLEASE TURN {head_turn_target}"
                    draw_premium_bracket_box(frame, bbox, display_tag, is_identified=False)
                    
                    if current_turn == head_turn_target:
                        auth_state = "AUTHORIZED"
                        print("[INFO] Liveness: Head turn successfully verified! Unlocking identity authentication.")
                        
                elif auth_state == "AUTHORIZED":
                    # Draw primary beautiful color-coded bounding brackets (Neon Green for Granted, Crimson for Stranger)
                    label_text = "ACCESS GRANTED" if is_identified else "ACCESS DENIED"
                    display_name = f"{label_text} | {matched_name.upper()}"
                    draw_premium_bracket_box(frame, bbox, display_name, is_identified, score=match_score if is_identified or match_score > 0.3 else None)
            else:
                # Fallback to standard tracking text if Face Mesh temporarily loses tracks
                label_text = "ACCESS GRANTED" if is_identified and auth_state == "AUTHORIZED" else "ACCESS DENIED"
                display_name = f"{label_text} | {matched_name.upper()}"
                draw_premium_bracket_box(frame, bbox, display_name, is_identified and auth_state == "AUTHORIZED", score=match_score if is_identified or match_score > 0.3 else None)
            
            # Optionally draw secondary faint boxes for other detected background faces
            for background_face in detections[1:]:
                bg_bbox = background_face['bbox']
                cv2.rectangle(frame, (bg_bbox[0], bg_bbox[1]), (bg_bbox[0]+bg_bbox[2], bg_bbox[1]+bg_bbox[3]), (80, 80, 80), 1, lineType=cv2.LINE_AA)
        else:
            # Face Track Lost Check: If no faces are tracked for > 2.5 seconds, reset liveness locks to secure the terminal
            if time.time() - last_face_time > 2.5:
                if auth_state != "LIVENESS_BLINK" or liveness.blink_count > 0:
                    print("[INFO] Face track lost for 2.5 seconds. Resetting active liveness security locks.")
                auth_state = "LIVENESS_BLINK"
                liveness.blink_count = 0
            
            is_identified = False
            matched_name = "UNKNOWN"
            match_score = 0.0
            stable_name = "UNKNOWN"
            stable_score = 0.0
            stable_identified = False
            prediction_buffer.clear()   # Flush buffer on face loss — clean slate for next person

        # Update FPS
        fps = fps_calc.update()

        # ── Phase 8: Profiler output (every 120 frames when enabled) ──────────
        if ENABLE_PROFILER and frame_count % 120 == 0 and frame_count > 0:
            stage_timer.report()
        
        # ── Stats card overlay (Top-Left Corner) ──────────────────────────────
        model_tag = recognizer.model_precision.upper() if not recognizer.simulated else "SIM"
        cv2.rectangle(frame, (10, 10), (235, 90), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 10), (235, 90), (100, 100, 100), 1)
        cv2.putText(frame, f"System : BLINK AI Engine",         (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Model  : {model_tag}",             (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS    : {fps:.1f}",               (15, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if fps > 18 else (0, 165, 255), 1, cv2.LINE_AA)
        liveness_color = (0, 255, 0) if auth_state == "AUTHORIZED" else (0, 165, 255)
        cv2.putText(frame, f"Security: {auth_state}",           (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, liveness_color, 1, cv2.LINE_AA)

        # Pip Overlay: If a face is tracked, overlay the preprocessed 112x112 crop in the top-right corner of the window
        if latest_crop is not None:
            ch, cw, _ = frame.shape
            pip_size = 112
            x_offset = cw - pip_size - 10
            y_offset = 10
            
            # Use color-coded border for Pip window matching access status
            pip_border_color = (0, 255, 0) if is_identified and auth_state == "AUTHORIZED" else (0, 0, 255)
            cv2.rectangle(frame, (x_offset - 2, y_offset - 2), (x_offset + pip_size + 2, y_offset + pip_size + 2), pip_border_color, 2)
            frame[y_offset:y_offset+pip_size, x_offset:x_offset+pip_size] = latest_crop
            cv2.putText(frame, "CROP 112x112", (x_offset, y_offset + pip_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pip_border_color, 1, cv2.LINE_AA)

        # Render output
        cv2.imshow("BLINK - Offline AI Authentication Stream", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("[INFO] Safely closing webcam and releasing resources...")
            break

        elif key == ord('p'):
            # Toggle latency profiler on/off
            ENABLE_PROFILER = not ENABLE_PROFILER
            status = "ON" if ENABLE_PROFILER else "OFF"
            print(f"[INFO] Stage latency profiler: {status}")
            if ENABLE_PROFILER:
                stage_timer.reset()  # Fresh window when enabling
            
        elif key == ord('e'):
            # Pauses OpenCV capture to prompt dynamic terminal enrollment
            if latest_crop is not None and latest_embedding is not None:
                print("\n" + "="*50)
                print("BLINK - LIVE USER ENROLLMENT")
                print("="*50)
                print("Camera stream paused. Check your terminal to enroll.")
                
                # Prompt user in terminal
                name_input = input(">>> Enter name to enroll currently tracked face: ").strip()
                
                if name_input:
                    # Enrolls user into the database JSON registry and saves crop image
                    success = db.enroll_user(name_input, latest_embedding, latest_crop, model_mode="simulated" if recognizer.simulated else "onnx")
                    if success:
                        print(f"[SUCCESS] '{name_input}' enrolled successfully! Resuming camera stream...")
                    else:
                        print("[ERROR] Enrollment failed. Resuming camera stream...")
                else:
                    print("[WARNING] Enrollment cancelled (empty name). Resuming camera stream...")
                print("="*50 + "\n")
            else:
                print("[WARNING] Cannot enroll. No face crop currently tracked in the frame.")

    # Cleanup resources
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System resources released successfully.")

if __name__ == "__main__":
    main()
