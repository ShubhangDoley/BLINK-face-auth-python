# BLINK — Mobile Integration Guide (React Native)

This document is the **source of truth** for porting the BLINK face recognition engine to a React Native app. It precisely documents every preprocessing and inference step so the RN developer can reimplement them in JS/native without guessing.

---

## Architecture Overview

```
React Native App
│
├── react-native-vision-camera      ← Camera frame capture
│   └── Frame Processor Plugin
│       ├── MediaPipe Tasks Vision  ← Face detection + crop
│       └── BLINK Inference Plugin  ← Preprocessing + TFLite/ONNX
│
├── models/mobile/
│   ├── mobilefacenet.tflite        ← Preferred (CoreML/NNAPI delegate)
│   └── mobilefacenet.with_runtime_opt.onnx  ← Alternative (ORT Mobile)
│
└── data/users.json                 ← Enrolled user registry (sync from Python)
```

---

## Step 1: Camera Setup

Use **react-native-vision-camera v4+** for zero-copy frame access:

```bash
npm install react-native-vision-camera
```

Request camera permission and set format:
```ts
const format = useCameraFormat(device, [
  { videoResolution: { width: 640, height: 480 } },
  { fps: 30 },
]);
```

---

## Step 2: Face Detection & Crop

### Option A — MediaPipe Tasks Vision (Recommended)
```bash
npm install @mediapipe/tasks-vision
```

```ts
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks(
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
);
const faceDetector = await FaceDetector.createFromOptions(vision, {
  baseOptions: { modelAssetPath: './blaze_face_short_range.tflite' },
  runningMode: 'VIDEO',
  minDetectionConfidence: 0.6,
});
```

### Option B — Vision Camera + TFLite plugin
Use the `react-native-vision-camera` frame processor with a custom TFLite face detector.

### Crop & Resize (must match Python)
After getting the bounding box, crop and resize to **exactly 112×112**:

```ts
// Padding ratio must match Python: 15%
const padX = bbox.width  * 0.15;
const padY = bbox.height * 0.15;
const x1 = Math.max(0, bbox.x - padX);
const y1 = Math.max(0, bbox.y - padY);
const x2 = Math.min(frameWidth,  bbox.x + bbox.width  + padX);
const y2 = Math.min(frameHeight, bbox.y + bbox.height + padY);

// Resize to 112×112 — use INTER_AREA equivalent (bilinear for downscaling)
const crop = resizeTo112x112(frame, x1, y1, x2, y2);
```

---

## Step 3: Preprocessing Pipeline

> [!IMPORTANT]
> **These steps must be executed in this exact order.** Each step is critical for producing embeddings that match the Python-enrolled ones in users.json.

### 3.1 — Color Format
The camera frame is typically in **BGR (Android)** or **YUV/BGRA (iOS)**.
Convert to **RGB** before any other operation.

```ts
// On Android (JNI/native): use cv::cvtColor(frame, rgb, COLOR_BGR2RGB)
// On iOS (Metal): swap R and B channels in a shader
// In JS (react-native-fast-image or skia): most libraries output RGB natively
```

### 3.2 — CLAHE (Contrast Enhancement) ⚠️ Required

CLAHE must be applied **before normalization**, on the **LAB color space L-channel only**.

Parameters (must match Python exactly):
- **Color space**: `BGR → LAB`
- **Channel**: L only (luminance), leave A and B untouched
- **Clip Limit**: `2.0`
- **Tile Grid Size**: `8 × 8`

**JavaScript (WebAssembly OpenCV):**
```ts
import cv from 'opencv.js'; // or @techstark/opencv-js

const lab = new cv.Mat();
cv.cvtColor(cropMat, lab, cv.COLOR_BGR2Lab);

const channels = new cv.MatVector();
cv.split(lab, channels);

const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
const lEnhanced = new cv.Mat();
clahe.apply(channels.get(0), lEnhanced);

channels.set(0, lEnhanced);
cv.merge(channels, lab);
cv.cvtColor(lab, cropMat, cv.COLOR_Lab2BGR);
```

**Native (Android C++/JNI):**
```cpp
cv::Mat lab;
cv::cvtColor(crop, lab, cv::COLOR_BGR2Lab);
std::vector<cv::Mat> channels;
cv::split(lab, channels);
auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
clahe->apply(channels[0], channels[0]);
cv::merge(channels, lab);
cv::cvtColor(lab, crop, cv::COLOR_Lab2BGR);
```

**Native (iOS Swift/ObjC via OpenCV):**
Same as Android C++, call from an ObjC++ bridge.

### 3.3 — Normalization (MobileFaceNet Standard)

After CLAHE, normalize every pixel value:

```
pixel_normalized = (pixel_float32 - 127.5) / 128.0
```

This maps the `[0, 255]` uint8 range to `[-0.9961, 0.9961]` float32.

```ts
// In typed arrays (Float32Array):
for (let i = 0; i < pixels.length; i++) {
  normalized[i] = (pixels[i] - 127.5) / 128.0;
}
```

### 3.4 — Tensor Layout (NCHW)

MobileFaceNet expects input in **NCHW** format: `[batch=1, channels=3, height=112, width=112]`.

Most camera APIs give you **HWC** (height, width, channels). You must transpose:

```ts
// HWC → CHW → NCHW
const tensor = new Float32Array(1 * 3 * 112 * 112);
for (let h = 0; h < 112; h++) {
  for (let w = 0; w < 112; w++) {
    for (let c = 0; c < 3; c++) {
      // HWC index: (h * 112 + w) * 3 + c
      // CHW index: c * 112 * 112 + h * 112 + w
      tensor[c * 112 * 112 + h * 112 + w] = normalizedHWC[(h * 112 + w) * 3 + c];
    }
  }
}
// tensor shape: [1, 3, 112, 112] — ready for inference
```

---

## Step 4: Model Inference

### Option A — TFLite (Recommended)

```bash
npm install react-native-tflite
# or: npx expo install react-native-tflite
```

```ts
import { loadTensorflowModel } from 'react-native-tflite';

const model = await loadTensorflowModel(
  require('./models/mobile/mobilefacenet.tflite'),
  'default'  // use 'gpu' for Android GPU delegate, 'core-ml' for iOS
);

// Input: Float32Array of shape [1, 3, 112, 112]
const output = await model.run([tensor]);
const embedding128d: Float32Array = output[0]; // shape: [128]
```

**Hardware Delegates:**
| Platform | Delegate | Speed gain |
|----------|----------|-----------|
| Android  | GPU (`'gpu'`) | 3–5× vs CPU |
| Android  | NNAPI (`'nnapi'`) | Varies by chip |
| iOS      | CoreML (`'core-ml'`) | 3–8× vs CPU |

### Option B — ONNX Runtime Mobile

```bash
npm install onnxruntime-react-native
```

```ts
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

const session = await InferenceSession.create(
  './models/mobile/mobilefacenet.with_runtime_opt.onnx'
);

const inputTensor = new Tensor('float32', tensor, [1, 3, 112, 112]);
const feeds = { [session.inputNames[0]]: inputTensor };
const results = await session.run(feeds);
const embedding128d = results[session.outputNames[0]].data as Float32Array;
```

---

## Step 5: Post-Processing

### L2 Normalize the Embedding

Always normalize the raw model output before comparison or storage:

```ts
function l2Normalize(embedding: Float32Array): Float32Array {
  let norm = 0;
  for (const v of embedding) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return embedding;
  return embedding.map(v => v / norm);
}

const normalizedEmbedding = l2Normalize(embedding128d);
```

### Cosine Similarity (Identity Matching)

Since embeddings are L2-normalized (unit vectors), cosine similarity = dot product:

```ts
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // range: [-1, 1]; 1.0 = identical
}

// Threshold for match:
const MATCH_THRESHOLD = 0.60;

function searchUser(
  queryEmb: Float32Array,
  users: Record<string, { embedding: number[] }>
): { name: string | null; score: number } {
  let bestName: string | null = null;
  let bestScore = -1;

  for (const [name, record] of Object.entries(users)) {
    const refEmb = new Float32Array(record.embedding);
    const score = cosineSimilarity(queryEmb, refEmb);
    if (score > bestScore) {
      bestScore = score;
      bestName = name;
    }
  }

  return bestScore >= MATCH_THRESHOLD
    ? { name: bestName, score: bestScore }
    : { name: null,     score: bestScore };
}
```

---

## Step 6: Temporal Smoothing (Match Python Behavior)

The Python engine uses a **12-frame majority vote** to prevent flickering. Implement the same on mobile:

```ts
const SMOOTHING_WINDOW = 12;
const MAJORITY_THRESHOLD = 0.50;
const predictionBuffer: Array<[string, number]> = [];

function updateStableIdentity(rawName: string, rawScore: number) {
  predictionBuffer.push([rawName, rawScore]);
  if (predictionBuffer.length > SMOOTHING_WINDOW) predictionBuffer.shift();

  const votes: Record<string, number> = {};
  for (const [name] of predictionBuffer) votes[name] = (votes[name] ?? 0) + 1;

  const [topName, topVotes] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  if (topVotes / predictionBuffer.length >= MAJORITY_THRESHOLD) {
    return topName;
  }
  return stableName; // keep last stable result — no flicker
}
```

---

## User Database (users.json) Schema

The `users.json` file enrolled from the Python engine can be used directly in the RN app. Structure:

```json
{
  "Alice": {
    "embedding": [0.023, -0.142, ..., 0.087],  // 128 float values, L2-normalized
    "enrolled_at": "2025-05-23T12:00:00Z",
    "model_mode": "int8"                         // "int8" | "fp32" | "simulated"
  }
}
```

> [!WARNING]
> **Embeddings are model-mode specific.** If you switch from FP32 to INT8 (or FP32→TFLite), you must re-enroll all users. The Python engine's `migrate_database_embeddings()` handles this automatically on startup.

> [!IMPORTANT]
> **For mobile enrollment**: Either re-enroll via the Python tool and sync `users.json`, or implement an enrollment flow in the RN app that generates embeddings with the same pipeline and writes to `users.json`.

---

## Performance Targets (Mobile)

Based on INT8 Python benchmarks (Phase 8 StageTimer output), estimated mobile times:

| Stage | Python CPU (ms) | Android GPU (ms) | iOS CoreML (ms) |
|-------|----------------|-----------------|----------------|
| Face Detection (BlazeFace) | 8–15 | 2–5 | 2–4 |
| CLAHE preprocessing | 2–4 | 1–2 | 1–2 |
| ONNX/TFLite Inference (INT8) | 8–18 | **2–5** | **2–6** |
| DB search (cosine, 10 users) | 0.1 | 0.1 | 0.1 |
| **Total per frame** | **~20–40ms** | **~8–15ms** | **~6–12ms** |
| **Target FPS** | 25–30 | **60+** | **60+** |

---

## Recommended RN Packages

| Purpose | Package | Notes |
|---------|---------|-------|
| Camera | `react-native-vision-camera` | v4+, frame processor support |
| TFLite inference | `react-native-tflite` | GPU/CoreML delegate |
| ORT Mobile | `onnxruntime-react-native` | Alternative to TFLite |
| Face detection | `@mediapipe/tasks-vision` | WASM or native |
| OpenCV (CLAHE) | `opencv-react-native` or native bridge | For CLAHE preprocessing |

---

## Quick Start Checklist

- [ ] Run `python src/quantize.py` → `models/mobilefacenet_int8.onnx`
- [ ] Run `python src/export_mobile.py --target tflite` → `models/mobile/mobilefacenet.tflite`
- [ ] Copy `models/mobile/mobilefacenet.tflite` into your RN app's assets
- [ ] Copy `data/users.json` into your RN app (or implement sync)
- [ ] Implement preprocessing: resize → CLAHE (LAB, L-channel) → normalize → NCHW
- [ ] Load TFLite model with GPU/CoreML delegate
- [ ] Implement L2 normalization and cosine similarity matching
- [ ] Implement 12-frame majority vote buffer for temporal smoothing
- [ ] Test: enroll via Python, verify match on mobile
