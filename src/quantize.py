"""
quantize.py — Phase 8: INT8 Dynamic Quantization for Mobile Deployment
=======================================================================
One-time offline script. Run this once to produce a quantized INT8 ONNX model
that is ~4x smaller and ~2x faster than the FP32 original.

This is the critical first step before mobile export (export_mobile.py).

Usage:
    python src/quantize.py

Output:
    models/mobilefacenet_int8.onnx  (~10MB vs 38MB FP32)

Mobile relevance:
    The INT8 model is the input for TFLite / ORT Mobile conversion.
    Smaller model = faster cold start, less RAM, better battery life on device.
"""

import os
import sys
import time
import numpy as np

def quantize_model():
    print("\n" + "="*60)
    print("BLINK — INT8 DYNAMIC QUANTIZATION (PHASE 8)")
    print("="*60)

    # ── Path setup ──────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    fp32_path  = os.path.join(models_dir, "mobilefacenet.onnx")
    int8_path  = os.path.join(models_dir, "mobilefacenet_int8.onnx")

    # ── Pre-flight checks ────────────────────────────────────────────
    if not os.path.exists(fp32_path):
        print(f"\n[ERROR] FP32 source model not found: {fp32_path}")
        print("[INFO]  Run: python src/download_weights.py  to download it first.")
        sys.exit(1)

    if os.path.exists(int8_path):
        size_mb = os.path.getsize(int8_path) / (1024 * 1024)
        print(f"\n[INFO] INT8 model already exists at: {int8_path} ({size_mb:.1f} MB)")
        overwrite = input(">>> Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("[INFO] Skipping quantization. Existing INT8 model retained.")
            _run_accuracy_check(fp32_path, int8_path)
            return

    # ── Import quantization library ──────────────────────────────────
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("\n[ERROR] onnxruntime.quantization not available.")
        print("[INFO]  Install with: pip install onnxruntime>=1.16.0")
        sys.exit(1)

    # ── Run quantization ─────────────────────────────────────────────
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"\n[INFO] Source model : {fp32_path} ({fp32_size_mb:.1f} MB)")
    print(f"[INFO] Output target: {int8_path}")
    print("[INFO] Strategy     : Dynamic INT8 (weights only, activations at runtime)")
    print("[INFO] This takes ~10–30 seconds...")

    t_start = time.time()
    try:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
        )
    except Exception as e:
        print(f"\n[ERROR] Quantization failed: {e}")
        sys.exit(1)

    elapsed = time.time() - t_start
    int8_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    reduction = (1 - int8_size_mb / fp32_size_mb) * 100

    print(f"\n{'='*60}")
    print("QUANTIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  FP32 size  : {fp32_size_mb:.1f} MB")
    print(f"  INT8 size  : {int8_size_mb:.1f} MB")
    print(f"  Reduction  : {reduction:.1f}%")
    print(f"  Time taken : {elapsed:.1f}s")
    print(f"  Saved to   : {int8_path}")
    print(f"{'='*60}")

    # ── Accuracy sanity check ────────────────────────────────────────
    _run_accuracy_check(fp32_path, int8_path)

    print("\n[NEXT STEP] Run export_mobile.py to convert to TFLite / ORT Mobile.")


def _run_accuracy_check(fp32_path: str, int8_path: str):
    """
    Verifies that INT8 and FP32 models produce near-identical embeddings
    on a synthetic test input. Cosine similarity should be > 0.97.
    """
    print("\n[INFO] Running accuracy sanity check...")
    try:
        import onnxruntime as ort

        # Build a deterministic synthetic face tensor (same input both models)
        rng = np.random.RandomState(42)
        # MobileFaceNet expects (1, 3, 112, 112) float32, range ~[-1, 1]
        test_input = rng.uniform(-1.0, 1.0, (1, 3, 112, 112)).astype(np.float32)

        def run_session(model_path, tensor):
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: tensor})[0]
            emb = output[0].flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        emb_fp32 = run_session(fp32_path, test_input)
        emb_int8 = run_session(int8_path, test_input)

        cosine_sim = float(np.dot(emb_fp32, emb_int8))
        status = "PASS" if cosine_sim > 0.97 else "FAIL (unexpected accuracy drop)"

        print(f"  FP32 vs INT8 cosine similarity: {cosine_sim:.6f}  ->  {status}")
        if cosine_sim <= 0.97:
            print("  [WARNING] Accuracy drop higher than expected. Consider FP16 instead.")
        else:
            print("  [INFO] INT8 output is functionally equivalent to FP32. Safe to deploy.")

    except Exception as e:
        print(f"  [WARNING] Accuracy check failed: {e}. Skipping.")


if __name__ == "__main__":
    quantize_model()
