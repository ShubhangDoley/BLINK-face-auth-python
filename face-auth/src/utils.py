import os
import time
from collections import deque, defaultdict
from contextlib import contextmanager
from typing import Optional

class FPSCalculator:
    """
    Computes frames-per-second (FPS) dynamically using a moving average window.
    This prevents high-frequency fluctuations in the displayed FPS.
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.prev_time = time.time()

    def update(self) -> float:
        """Call this on every frame iteration to update time queue and get current FPS."""
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        self.prev_time = curr_time
        
        self.frame_times.append(time_diff)
        avg_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time


class StageTimer:
    """
    [Phase 8 - Latency Profiler]
    Context-manager based per-stage latency profiler.

    Collects rolling statistics (avg, max, total calls) for each named pipeline
    stage. Used to benchmark where ms are spent before porting to React Native.

    Usage:
        timer = StageTimer(window_size=60)

        with timer.measure("detect_faces"):
            detections = detector.detect_faces(frame)

        with timer.measure("onnx_inference"):
            embedding = recognizer.generate_embedding(face_tensor)

        # Print a formatted summary table every N frames
        if frame_count % 120 == 0:
            timer.report()
    """

    def __init__(self, window_size: int = 60):
        """
        Args:
            window_size: Rolling window size for average calculation (frames).
        """
        self._window_size = window_size
        # stage_name -> deque of recent durations in ms
        self._windows: dict = defaultdict(lambda: deque(maxlen=window_size))
        # stage_name -> all-time max ms
        self._max_ms: dict = defaultdict(float)
        # stage_name -> total call count
        self._calls: dict = defaultdict(int)
        # Tracks insertion order for display
        self._order: list = []

    @contextmanager
    def measure(self, stage: str):
        """
        Context manager that times the enclosed block and records it under `stage`.

        Args:
            stage: Human-readable pipeline stage name (e.g. 'onnx_inference').
        """
        if stage not in self._order:
            self._order.append(stage)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._windows[stage].append(elapsed_ms)
            self._calls[stage] += 1
            if elapsed_ms > self._max_ms[stage]:
                self._max_ms[stage] = elapsed_ms

    def avg_ms(self, stage: str) -> float:
        """Returns the rolling average latency in ms for a stage."""
        buf = self._windows.get(stage)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def report(self, print_output: bool = True) -> str:
        """
        Generates and optionally prints a formatted profiling summary table.

        Returns:
            The formatted report string.
        """
        if not self._order:
            return "(no stages recorded)"

        lines = [
            "",
            "┌─────────────────────────────────────────────────────┐",
            "│  BLINK Stage Latency Report (Phase 8 Profiler)      │",
            "├──────────────────────┬──────────┬──────────┬────────┤",
            "│ Stage                │  Avg ms  │  Max ms  │ Calls  │",
            "├──────────────────────┼──────────┼──────────┼────────┤",
        ]

        total_avg = 0.0
        for stage in self._order:
            avg = self.avg_ms(stage)
            mx  = self._max_ms.get(stage, 0.0)
            cnt = self._calls.get(stage, 0)
            total_avg += avg
            lines.append(f"│ {stage:<20} │ {avg:>8.2f} │ {mx:>8.2f} │ {cnt:>6} │")

        lines.append("├──────────────────────┼──────────┼──────────┼────────┤")
        lines.append(f"│ {'TOTAL':<20} │ {total_avg:>8.2f} │ {'':>8} │ {'':>6} │")
        lines.append("└──────────────────────┴──────────┴──────────┴────────┘")
        lines.append(f"  → Theoretical max FPS (pipeline): {1000/total_avg:.1f}" if total_avg > 0 else "")
        lines.append("")

        report_str = "\n".join(lines)
        if print_output:
            print(report_str)
        return report_str

    def reset(self):
        """Clears all recorded measurements."""
        self._windows.clear()
        self._max_ms.clear()
        self._calls.clear()
        self._order.clear()


def ensure_directories():
    """
    Ensures that all folders for the BLINK project are established.
    Directory structure:
      face-auth/
      ├── models/
      │   └── mobile/
      ├── data/
      │   ├── known_faces/
      │   └── test/
      └── src/
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directories = [
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "models", "mobile"),
        os.path.join(base_dir, "data", "known_faces"),
        os.path.join(base_dir, "data", "test"),
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"[INFO] Created directory: {directory}")
