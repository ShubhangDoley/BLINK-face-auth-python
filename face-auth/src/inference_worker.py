"""
inference_worker.py — Phase 8: Background Inference Thread
===========================================================
Decouples ONNX inference from the camera display loop so the main thread
never blocks on a forward pass. The display loop always renders at the
camera's natural FPS using the most recently available inference result.

Architecture:
    Main Thread:                 Inference Thread:
    ─────────────────────        ──────────────────────
    cap.read()           ──────► preprocess()
    detect_faces()               generate_embedding()
    liveness.extract()           search_user()
    imshow() [last result]◄───── update shared result
    waitKey(1)

Mobile Note:
    This Python threading pattern is NOT ported to React Native.
    On mobile, ONNX Runtime Mobile / TFLite run asynchronously via their
    native APIs with hardware delegate scheduling. This worker exists solely
    to make the Python prototype demo run at full camera FPS.

Usage:
    worker = InferenceWorker(preprocessor, recognizer, db)
    worker.start()

    # In main loop:
    if face_crop is not None:
        worker.submit_crop(face_crop)          # Non-blocking
    name, score, emb = worker.get_result()     # Returns last stable result

    worker.stop()
"""

import threading
import numpy as np
from typing import Optional, Tuple
from preprocess import FacePreprocessor
from recognize import FaceRecognizer
from database import UserDatabase


class InferenceWorker:
    """
    [Phase 8 - Background Inference Thread]
    Encapsulates ONNX face recognition inference in a dedicated daemon thread.

    Thread-safety is maintained via:
        - threading.Event  : signals new crop availability to the worker
        - threading.Lock   : protects shared result variables from race conditions
        - threading.Event  : signals the worker to stop cleanly
    """

    def __init__(
        self,
        preprocessor: FacePreprocessor,
        recognizer:   FaceRecognizer,
        db:           UserDatabase,
        threshold:    float = 0.60,
    ):
        """
        Args:
            preprocessor: Initialized FacePreprocessor instance.
            recognizer:   Initialized FaceRecognizer instance (FP32 or INT8 ONNX).
            db:           Initialized UserDatabase instance.
            threshold:    Cosine similarity threshold for identity matching.
        """
        self.preprocessor = preprocessor
        self.recognizer   = recognizer
        self.db           = db
        self.threshold    = threshold

        # ── Shared state (protected by _result_lock) ─────────────────
        self._matched_name:  str             = "UNKNOWN"
        self._match_score:   float           = 0.0
        self._last_embedding: Optional[np.ndarray] = None
        self._result_lock = threading.Lock()

        # ── Pending crop slot (protected by _crop_lock) ──────────────
        # Only the LATEST submitted crop is processed; stale frames are dropped.
        self._pending_crop: Optional[np.ndarray] = None
        self._crop_lock = threading.Lock()

        # ── Thread control primitives ─────────────────────────────────
        self._new_crop_event = threading.Event()   # Wakes worker when crop arrives
        self._stop_event     = threading.Event()   # Signals graceful shutdown
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="BLINKInferenceWorker",
            daemon=True,   # Exits automatically when main thread exits
        )

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        """Starts the background inference thread."""
        self._stop_event.clear()
        self._thread.start()
        print("[INFO] InferenceWorker: Background inference thread started.")

    def stop(self):
        """Signals the worker to stop and waits for it to join cleanly."""
        self._stop_event.set()
        self._new_crop_event.set()  # Unblock the wait() in case it's sleeping
        self._thread.join(timeout=3.0)
        print("[INFO] InferenceWorker: Background thread stopped.")

    def submit_crop(self, crop: np.ndarray):
        """
        Submits a new face crop for inference. Non-blocking.
        If the worker hasn't finished the previous crop yet, the old one is
        silently replaced — we always want the freshest frame.

        Args:
            crop: 112×112 BGR face crop from detect.crop_and_align().
        """
        with self._crop_lock:
            self._pending_crop = crop.copy()
        self._new_crop_event.set()   # Wake up the worker

    def get_result(self) -> Tuple[str, float, Optional[np.ndarray]]:
        """
        Returns the most recent stable inference result. Non-blocking.
        The main thread calls this every frame to get the current identity.

        Returns:
            Tuple of (matched_name: str, score: float, embedding: Optional[np.ndarray])
            matched_name is 'UNKNOWN' if no match found or no inference run yet.
        """
        with self._result_lock:
            return self._matched_name, self._match_score, self._last_embedding

    @property
    def is_alive(self) -> bool:
        """Returns True if the background thread is still running."""
        return self._thread.is_alive()

    # ── Internal worker loop ──────────────────────────────────────────

    def _inference_loop(self):
        """
        The background thread's main loop.
        Waits for a new crop, processes it end-to-end, writes the result,
        then goes back to sleep. Exits cleanly when stop() is called.
        """
        while not self._stop_event.is_set():
            # Block until a crop arrives or stop is signalled
            self._new_crop_event.wait()
            self._new_crop_event.clear()

            if self._stop_event.is_set():
                break

            # Grab the latest crop (and clear the pending slot atomically)
            with self._crop_lock:
                crop = self._pending_crop
                self._pending_crop = None

            if crop is None:
                continue

            # ── Full inference pipeline ───────────────────────────────
            try:
                # Step 1: Preprocess → NCHW float32 tensor
                face_tensor = self.preprocessor.preprocess(crop, apply_clahe=True)

                # Step 2: ONNX inference → L2-normalized embedding
                embedding = self.recognizer.generate_embedding(face_tensor)

                # Step 3: Nearest-neighbor search in JSON registry
                matched_user, score = self.db.search_user(
                    embedding, self.recognizer, threshold=self.threshold
                )

                name = matched_user if matched_user is not None else "UNKNOWN"

                # Step 4: Write result (main thread reads this via get_result())
                with self._result_lock:
                    self._matched_name   = name
                    self._match_score    = score
                    self._last_embedding = embedding.copy()

            except Exception as e:
                # Never crash the background thread — log and continue
                print(f"[WARNING] InferenceWorker: Inference error: {e}")
