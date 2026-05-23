"""
export_mobile.py — Phase 8: Mobile Model Export Pipeline
=========================================================
Converts the quantized ONNX model to formats suitable for React Native deployment.

Supported export targets:
  Path A: TFLite (Recommended for mobile)
    mobilefacenet_int8.onnx → TF SavedModel → mobilefacenet.tflite
    Output: models/mobile/mobilefacenet.tflite

  Path B: ONNX Runtime Mobile (Alternative)
    mobilefacenet_int8.onnx → ORT optimized format
    Output: models/mobile/mobilefacenet.with_runtime_opt.onnx

Run quantize.py first to produce mobilefacenet_int8.onnx before running this.

Usage:
    python src/export_mobile.py [--target tflite|ort|both]
"""

import os
import sys
import argparse
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MOBILE_DIR = os.path.join(MODELS_DIR, "mobile")
INT8_PATH  = os.path.join(MODELS_DIR, "mobilefacenet_int8.onnx")
FP32_PATH  = os.path.join(MODELS_DIR, "mobilefacenet.onnx")


def _get_source_model() -> str:
    """Returns the best available ONNX model path (INT8 preferred)."""
    if os.path.exists(INT8_PATH):
        print(f"[INFO] Source: INT8 quantized model → {INT8_PATH}")
        return INT8_PATH
    elif os.path.exists(FP32_PATH):
        print(f"[WARNING] INT8 model not found. Using FP32 fallback → {FP32_PATH}")
        print("[TIP] Run quantize.py first for a smaller, faster mobile model.")
        return FP32_PATH
    else:
        print("[ERROR] No ONNX model found in models/")
        print("  1. Run: python src/download_weights.py")
        print("  2. Run: python src/quantize.py")
        sys.exit(1)


# ── Path A: TFLite Export ────────────────────────────────────────────────────

def export_tflite(source_onnx: str) -> str:
    """
    Converts ONNX model to TFLite format.

    Pipeline:
        ONNX → TensorFlow SavedModel (via onnx-tf) → TFLite FlatBuffer

    Required packages:
        pip install onnx-tf tensorflow

    Returns:
        Path to the generated .tflite file.
    """
    output_path = os.path.join(MOBILE_DIR, "mobilefacenet.tflite")

    print("\n" + "─"*55)
    print("PATH A: TFLite Export")
    print("─"*55)

    # ── Step 1: ONNX → TF SavedModel ──────────────────────────────────
    try:
        import onnx
        import onnx_tf
        from onnx_tf.backend import prepare
    except ImportError:
        print("[ERROR] onnx-tf not installed. Run: pip install onnx-tf tensorflow")
        print("[SKIP] TFLite export skipped.")
        return ""

    savedmodel_dir = os.path.join(MOBILE_DIR, "_savedmodel_tmp")
    print(f"[STEP 1/3] Loading ONNX model: {source_onnx}")
    onnx_model = onnx.load(source_onnx)

    print(f"[STEP 2/3] Converting ONNX → TF SavedModel: {savedmodel_dir}")
    try:
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(savedmodel_dir)
    except Exception as e:
        print(f"[ERROR] ONNX → SavedModel conversion failed: {e}")
        print("[TIP] Some ONNX ops may not be supported by onnx-tf.")
        print("      Consider trying the ORT Mobile path instead (--target ort).")
        return ""

    # ── Step 2: TF SavedModel → TFLite ────────────────────────────────
    print(f"[STEP 3/3] Converting SavedModel → TFLite: {output_path}")
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
        # Keep float32 activations (dynamic range quantization for weights only)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        os.makedirs(MOBILE_DIR, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[SUCCESS] TFLite model saved: {output_path} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"[ERROR] SavedModel → TFLite conversion failed: {e}")
        return ""

    # ── Cleanup temp SavedModel dir ────────────────────────────────────
    import shutil
    if os.path.exists(savedmodel_dir):
        shutil.rmtree(savedmodel_dir)

    return output_path


# ── Path B: ONNX Runtime Mobile Export ──────────────────────────────────────

def export_ort_mobile(source_onnx: str) -> str:
    """
    Optimizes the ONNX model for ONNX Runtime Mobile (onnxruntime-react-native).

    Applies ORT graph optimizations and produces a .with_runtime_opt.onnx file
    that loads faster and uses less memory on mobile.

    Required packages:
        pip install onnxruntime (already in requirements.txt)

    Returns:
        Path to the optimized .onnx file.
    """
    output_path = os.path.join(MOBILE_DIR, "mobilefacenet.with_runtime_opt.onnx")

    print("\n" + "─"*55)
    print("PATH B: ONNX Runtime Mobile Export")
    print("─"*55)

    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer as ort_optimizer
    except ImportError:
        print("[WARNING] onnxruntime.transformers not available for full graph opt.")

    try:
        # Basic ORT session optimization — serializes the optimized graph
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_path

        os.makedirs(MOBILE_DIR, exist_ok=True)
        print(f"[INFO] Applying ORT_ENABLE_ALL graph optimizations...")
        # Loading a session with optimized_model_filepath causes ORT to write
        # the optimized graph to disk automatically
        _ = ort.InferenceSession(
            source_onnx,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[SUCCESS] ORT Mobile model saved: {output_path} ({size_mb:.1f} MB)")
        else:
            print(f"[WARNING] Optimized model not written to {output_path}.")
            print("          Some ORT versions handle this differently.")

    except Exception as e:
        print(f"[ERROR] ORT Mobile export failed: {e}")
        return ""

    return output_path


# ── Accuracy Validation ──────────────────────────────────────────────────────

def validate_tflite(tflite_path: str, source_onnx: str):
    """Compares TFLite vs ONNX embedding outputs for accuracy verification."""
    if not tflite_path or not os.path.exists(tflite_path):
        return

    print("\n[INFO] Validating TFLite accuracy vs ONNX source...")
    try:
        import tensorflow as tf
        import onnxruntime as ort

        rng = np.random.RandomState(42)
        test_input = rng.uniform(-1.0, 1.0, (1, 3, 112, 112)).astype(np.float32)

        # ONNX inference
        sess = ort.InferenceSession(source_onnx, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        onnx_emb = sess.run(None, {in_name: test_input})[0][0].flatten()
        onnx_emb /= (np.linalg.norm(onnx_emb) + 1e-8)

        # TFLite inference
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        inp_detail  = interpreter.get_input_details()[0]
        out_detail  = interpreter.get_output_details()[0]

        # TFLite uses NHWC by default from onnx-tf conversion
        # Transpose NCHW → NHWC for TFLite input
        tflite_input = np.transpose(test_input, (0, 2, 3, 1))
        interpreter.set_tensor(inp_detail["index"], tflite_input)
        interpreter.invoke()
        tflite_emb = interpreter.get_tensor(out_detail["index"])[0].flatten()
        tflite_emb /= (np.linalg.norm(tflite_emb) + 1e-8)

        cosine_sim = float(np.dot(onnx_emb, tflite_emb))
        status = "✓ PASS" if cosine_sim > 0.95 else "✗ FAIL"
        print(f"  ONNX vs TFLite cosine similarity: {cosine_sim:.6f}  → {status}")

    except Exception as e:
        print(f"  [WARNING] TFLite validation failed: {e}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLINK Mobile Model Exporter")
    parser.add_argument(
        "--target",
        choices=["tflite", "ort", "both"],
        default="both",
        help="Export target: tflite (TFLite), ort (ORT Mobile), both (default: both)",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("BLINK — MOBILE MODEL EXPORT PIPELINE (PHASE 8)")
    print("="*60)

    os.makedirs(MOBILE_DIR, exist_ok=True)
    source = _get_source_model()

    tflite_out = ""
    ort_out    = ""

    if args.target in ("tflite", "both"):
        tflite_out = export_tflite(source)
        if tflite_out:
            validate_tflite(tflite_out, source)

    if args.target in ("ort", "both"):
        ort_out = export_ort_mobile(source)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)

    if tflite_out and os.path.exists(tflite_out):
        size = os.path.getsize(tflite_out) / (1024*1024)
        print(f"  ✓ TFLite  : {tflite_out} ({size:.1f} MB)")
        print(f"    → Use with: react-native-tflite")
        print(f"    → RN package: https://github.com/mrousavy/react-native-tflite")
    else:
        print(f"  ✗ TFLite  : Not exported (missing onnx-tf / tensorflow)")
        print(f"    → Install: pip install onnx-tf tensorflow")

    if ort_out and os.path.exists(ort_out):
        size = os.path.getsize(ort_out) / (1024*1024)
        print(f"  ✓ ORT Mobile: {ort_out} ({size:.1f} MB)")
        print(f"    → Use with: onnxruntime-react-native")
        print(f"    → npm package: onnxruntime-react-native")
    else:
        print(f"  ✗ ORT Mobile: Not exported")

    print(f"\n  Mobile models directory: {MOBILE_DIR}")
    print(f"\n[NEXT STEP] See MOBILE_INTEGRATION.md for RN integration guide.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
