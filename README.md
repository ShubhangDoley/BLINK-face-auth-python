<div align="center">

<br/>

```
██████╗ ██╗     ██╗███╗   ██╗██╗  ██╗
██╔══██╗██║     ██║████╗  ██║██║ ██╔╝
██████╔╝██║     ██║██╔██╗ ██║█████╔╝ 
██╔══██╗██║     ██║██║╚██╗██║██╔═██╗ 
██████╔╝███████╗██║██║ ╚████║██║  ██╗
╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
```

# BLINK — Biometric Face Authentication System

**Enterprise-grade, offline-first face recognition and attendance management.**  
Dual-component: a **React Native mobile app** + a **Python FastAPI backend** with end-to-end encrypted (E2EE) cloud synchronization to AWS.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React Native](https://img.shields.io/badge/React%20Native-0.73+-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactnative.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![AWS](https://img.shields.io/badge/AWS-App%20Runner%20%2B%20RDS-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Python Backend (AI Engine)](#-python-backend-ai-engine)
  - [Face Detection](#face-detection)
  - [Face Preprocessing](#face-preprocessing)
  - [Face Recognition](#face-recognition)
  - [Liveness Detection](#liveness-detection)
  - [FastAPI Server](#fastapi-server)
- [React Native Mobile App](#-react-native-mobile-app)
  - [Screens](#screens)
  - [Local AI Inference](#local-ai-inference-on-device)
  - [End-to-End Encryption](#end-to-end-encryption-e2ee)
  - [Offline-First Database](#offline-first-sqlite-database)
  - [Cloud Sync Engine](#cloud-sync-engine)
- [E2EE Cloud Sync Architecture](#-e2ee-cloud-sync-architecture)
- [PostgreSQL Setup (Local)](#-postgresql-setup-local)
- [AWS Deployment Guide](#-aws-deployment-guide)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [Security Design](#-security-design)

---

## 🔍 Overview

BLINK is a production-quality biometric attendance and authentication system built for real-world deployment. It consists of two tightly integrated components:

| Component | Technology | Role |
|-----------|-----------|------|
| **Python Backend** | FastAPI + MediaPipe + ONNX Runtime | Face detection, preprocessing, embedding generation |
| **Mobile App** | React Native + ONNX Runtime Mobile | On-device inference, attendance UI, offline sync |

### Key Capabilities

- ⚡ **On-Device AI** — MobileFaceNet ONNX model runs entirely on the Android/iOS device. No camera photos are ever uploaded to any server.
- 🔒 **End-to-End Encryption** — Biometric embedding vectors are encrypted with **AES-256-GCM** before leaving the device. The cloud stores only ciphertext — completely blind to your actual biometrics.
- 📱 **Offline-First** — The system works without an internet connection. Attendance logs and user registrations queue locally in SQLite and sync automatically when connectivity is restored.
- 🌐 **Hybrid Cloud Sync** — PostgreSQL on AWS RDS acts as an encrypted relay store, enabling multiple supervisor devices to share the same enrolled biometric registry.
- 🧬 **Liveness Detection** — Eye blink detection (Eye Aspect Ratio) and head yaw tracking via MediaPipe Face Mesh prevents photo/screen spoofing attacks.
- 📊 **Rich Dashboard** — Real-time attendance metrics, daily stats, late arrival tracking, and trend analytics.

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       MOBILE APP (Android/iOS)               │
│                                                              │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐  │
│  │  ML Kit Face │   │  ONNX Runtime   │   │ CryptoManager│  │
│  │  Detection   │──►│  MobileFaceNet  │──►│  AES-256-GCM │  │
│  │  (On-Device) │   │  (On-Device)    │   │  E2EE Encrypt│  │
│  └──────────────┘   └─────────────────┘   └──────┬───────┘  │
│                                                   │          │
│  ┌─────────────────────────────────────────────── ▼ ──────┐  │
│  │  SQLite (op-sqlite)         sync_queue  │  users table │  │
│  │  Offline-first local store  (PENDING)   │  attendance  │  │
│  └────────────────────────────────────────┬─────────────┘  │
│                                           │ SyncManager     │
└───────────────────────────────────────────┼─────────────────┘
                                            │ HTTPS (E2EE JSON)
                              ┌─────────────▼──────────────────┐
                              │   FastAPI Backend (Python)      │
                              │   POST /api/v1/sync/users       │
                              │   POST /api/v1/sync/attendance  │
                              │   POST /api/v1/embeddings       │
                              └─────────────┬──────────────────┘
                                            │ SQLAlchemy ORM
                              ┌─────────────▼──────────────────┐
                              │ PostgreSQL (Local or AWS RDS)   │
                              │  users table: {id, name,        │
                              │   encrypted_envelope_json,..}   │
                              │  attendance table: {...}        │
                              └────────────────────────────────┘
```

### Biometric Privacy Guarantee

```
📱 Mobile Device                         ☁️ Cloud Database (PostgreSQL)
┌─────────────────────┐                 ┌──────────────────────────────┐
│ Raw Embedding:       │                 │ Stored Record:               │
│ [0.123, -0.456, ...]│  AES-256-GCM   │ {                            │
│ (128-dimensional     │ ─────────────► │   ciphertext: "Xk2pQ9...",  │
│  float32 vector)     │  Encrypt       │   iv: "Ab3mK1...",           │
│                      │                │   tag: "ZxT9pL..."           │
│ 🔑 KEY NEVER LEAVES  │                │ }                            │
│    THE DEVICE        │                │ ← Completely unreadable      │
└─────────────────────┘                 │   even if DB is breached     │
                                        └──────────────────────────────┘
```

---

## 📁 Repository Structure

```
BLINK/                          ← Python Backend (this repo)
├── src/
│   ├── server.py               ← FastAPI server (main entry point)
│   ├── detect.py               ← MediaPipe face detection engine
│   ├── preprocess.py           ← CLAHE + tensor normalization pipeline
│   ├── recognize.py            ← MobileFaceNet ONNX inference + cosine similarity
│   ├── liveness.py             ← Eye blink (EAR) + head yaw liveness detection
│   ├── main.py                 ← Desktop real-time webcam demo
│   ├── models.py               ← SQLAlchemy ORM models (PostgreSQL)
│   ├── cloud_database.py       ← DB connection manager (PG ↔ SQLite fallback)
│   ├── test_db_conn.py         ← Diagnostic utility: test DB connection
│   ├── quantize.py             ← INT8 model quantization script
│   ├── export_mobile.py        ← Export model to .ort format for mobile
│   └── .env                    ← Database credentials (never commit!)
├── models/
│   ├── mobilefacenet.onnx      ← FP32 model weights
│   └── mobilefacenet_int8.onnx ← INT8 quantized model (~4x smaller)
├── data/
│   └── test/                   ← Saved face crops for testing
├── requirements.txt
└── Dockerfile                  ← Production container image

BLINK-RN/blink_app/            ← React Native Mobile App
├── src/
│   ├── screens/
│   │   ├── Dashboard/          ← Attendance dashboard + metrics
│   │   ├── Register/           ← New user face enrollment
│   │   ├── Authentication/     ← Face scan + verification
│   │   ├── Logs/               ← Attendance audit log
│   │   ├── Settings/           ← App configuration
│   │   └── Splash/             ← Startup splash screen
│   ├── services/
│   │   ├── ai/
│   │   │   ├── LocalInference.ts   ← On-device ONNX inference engine
│   │   │   ├── CryptoManager.ts    ← AES-256-GCM E2EE encryption
│   │   │   └── MLKitFaceDetector.ts← ML Kit face detection wrapper
│   │   ├── api/
│   │   │   └── ApiClient.ts        ← Axios HTTP client for backend
│   │   ├── sync/
│   │   │   └── SyncManager.ts      ← Offline-first sync reconciliation
│   │   └── network/
│   │       └── NetworkMonitor.ts   ← Network connectivity monitoring
│   ├── database/
│   │   ├── sqlite/
│   │   │   ├── DatabaseManager.ts  ← op-sqlite connection manager
│   │   │   └── migrations.ts       ← Versioned schema migrations (v1→v4)
│   │   └── repositories/
│   │       ├── UserRepository.ts
│   │       ├── AttendanceRepository.ts
│   │       ├── AuthLogRepository.ts
│   │       ├── SettingsRepository.ts
│   │       └── SyncQueueRepository.ts
│   ├── components/             ← Reusable UI components
│   ├── navigation/             ← React Navigation stack
│   ├── theme/                  ← Design tokens (colors, spacing, typography)
│   └── types/                  ← TypeScript interfaces
└── android/                    ← Android-specific native config
```

---

## 🐍 Python Backend (AI Engine)

The Python backend is a **FastAPI** server that provides the biometric computation engine. It runs on your server or laptop and exposes REST endpoints used by both the mobile app and desktop testing tools.

### Face Detection

**File:** [`src/detect.py`](src/detect.py)

Uses **MediaPipe Face Detection** to locate faces in real-time camera frames. Key features:

- `FaceDetector.detect_faces(frame)` — Returns list of `{bbox, score, landmarks}` for all detected faces
- `FaceDetector.crop_and_align(frame, bbox, landmarks)` — Extracts a padded, rotation-aligned 112×112 face crop using eye landmark vectors
- `FaceDetector.draw_premium_bbox(frame, bbox, score)` — Renders a neon corner-bracket overlay for the desktop UI
- Automatically selects the **largest face** (closest to camera) as the primary subject

**Model selection:**
- `model_selection=0`: Short-range (≤ 2 meters) — used for close-range selfies
- `model_selection=1`: Long-range (≤ 5 meters) — for surveillance angles

### Face Preprocessing

**File:** [`src/preprocess.py`](src/preprocess.py)

Standardizes raw face crops into float32 tensors consumable by MobileFaceNet.

**Full Pipeline:** `BGR Crop → Resize (112×112) → CLAHE → RGB → Normalize → NCHW Tensor`

| Step | Description |
|------|-------------|
| **Resize** | Enforces 112×112 using `INTER_AREA` (optimal for downsampling) |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization — corrects uneven lighting in the LAB L-channel |
| **Normalization** | `(pixel - 127.5) / 128.0` maps `[0, 255]` → `[-0.996, 0.996]` (MobileFaceNet standard) |
| **Tensor Format** | Transposes HWC → CHW, adds batch dimension → shape `(1, 3, 112, 112)` |

### Face Recognition

**File:** [`src/recognize.py`](src/recognize.py)

Generates high-dimensional face embedding vectors using **MobileFaceNet** ONNX.

**Model Priority Chain (auto-detection):**
```
mobilefacenet_int8.onnx  →  mobilefacenet.onnx  →  Simulated fallback
      (4x smaller)              (FP32 full)         (for testing only)
```

Key methods:

- `generate_embedding(face_tensor)` — Runs ONNX inference, returns **L2-normalized** 128D float32 vector
- `compare_embeddings(emb1, emb2)` — Computes cosine similarity (simplified to dot product since vectors are unit-length)
- `verify_identity(emb1, emb2, threshold=0.60)` — Returns `(is_match: bool, similarity_score: float)`

**Threshold Guidance:**
| Score Range | Interpretation |
|-------------|---------------|
| `≥ 0.60` | ✅ Same person |
| `0.40 – 0.59` | ⚠️ Uncertain — adjust angle/lighting |
| `< 0.40` | ❌ Different person |

**Simulated Fallback** (when no ONNX model is available): Projects the image through a fixed linear projection matrix seeded at `42`, producing deterministic 128D embeddings that accurately preserve image similarity properties — perfect for development without model files.

### Liveness Detection

**File:** [`src/liveness.py`](src/liveness.py)

Prevents photo and screen spoofing using **MediaPipe Face Mesh** (468 landmarks).

**Two independent liveness signals:**

**1. Eye Blink (EAR — Eye Aspect Ratio)**
```
EAR = (||v1_top - v1_bot|| + ||v2_top - v2_bot||) / (2 × ||h_left - h_right||)

EAR < 0.18  →  Eye CLOSED state
EAR > 0.22  →  Eye OPEN state → Blink registered ✓
```
A state machine transitions `OPEN → CLOSED → OPEN` to count genuine blinks.

**2. Head Turn (Yaw Estimation)**
```
yaw_ratio = nose_to_left_cheek / nose_to_right_cheek

yaw_ratio < 0.42  →  Head turned LEFT
yaw_ratio > 2.38  →  Head turned RIGHT
Otherwise         →  CENTER
```

### FastAPI Server

**File:** [`src/server.py`](src/server.py)

The main HTTP API server. Starts with `python src/server.py` and listens on port `8000`.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/embeddings` | Upload a face image, receive 128D embedding vector |
| `GET` | `/health` | Server status + model mode check |
| `POST` | `/api/v1/sync/users` | Batch upsert E2EE user envelopes from mobile |
| `GET` | `/api/v1/sync/users` | Fetch all enrolled E2EE envelopes |
| `POST` | `/api/v1/sync/attendance` | Batch upsert attendance logs from mobile |

**Database fallback:** If no `DATABASE_URL` environment variable is set, the server automatically falls back to a local `cloud_blink.db` SQLite file.

---

## 📱 React Native Mobile App

Built with React Native (TypeScript), targeting Android (primary) and iOS.

### Screens

| Screen | Description |
|--------|-------------|
| **SplashScreen** | Animated boot screen while SQLite migrations and models load |
| **DashboardScreen** | Main hub — daily attendance stats, animated metrics, pull-to-refresh sync |
| **RegisterScreen** | Guided face enrollment with camera capture, real-time preview, and E2EE encryption |
| **AuthenticationScreen** | Live face scan, 1:N matching against local registry, liveness challenge |
| **LogsScreen** | Full attendance audit log with user name, timestamp, and sync status |
| **SettingsScreen** | Server URL config, organization password entry, model mode selection |

### Local AI Inference (On-Device)

**File:** `src/services/ai/LocalInference.ts`

Runs the full face embedding pipeline 100% on the Android/iOS device. **No photos are ever sent to the server.**

**Inference Pipeline:**
```
Camera Photo (JPEG)
  │
  ├─→ ML Kit Face Detection (on-device)
  │     → Bounding box + head roll angle
  │
  ├─→ Crop + Rotation Alignment (±roll degrees)
  │     → 30% padding for safety
  │
  ├─→ Resize to 112×112
  │
  ├─→ Pixel Normalization: (x - 127.5) / 128.0
  │     → Float32 NCHW tensor [1, 3, 112, 112]
  │
  ├─→ ONNX Runtime Mobile (mobilefacenet.ort)
  │     → Raw 128D embedding
  │
  └─→ L2 Normalization
        → Unit-length 128D embedding vector ✓
```

**Key implementation details:**
- Model file (`mobilefacenet.ort`) is bundled as an Android asset and copied to `DocumentDirectoryPath` on first launch
- Supports all pixel formats: `RGBA`, `BGRA`, `ARGB`, `ABGR`, `RGB`, `BGR` — automatically detected per device
- Memory-managed: all image objects are explicitly `.dispose()`d after use to prevent leaks

### End-to-End Encryption (E2EE)

**File:** `src/services/ai/CryptoManager.ts`

Pure JavaScript implementation using `node-forge` — no native C++ compilation required, ensuring zero build errors across all Android/iOS configurations.

**Key Derivation (PBKDF2):**
```typescript
// Derives a 256-bit AES key from a shared organization password
const keyHex = CryptoManager.deriveKey(organizationPassword);
// PBKDF2: 10,000 iterations, HMAC-SHA256, fixed org salt
// → Same password on any device = same key = same encrypted data readable everywhere
```

**Encryption:**
```typescript
// Encrypts a 128D Float32Array into a portable JSON envelope
const envelope = CryptoManager.encryptEmbedding(embedding, keyHex);
// Returns: { ciphertext: "Base64...", iv: "Base64...", tag: "Base64..." }
// IV: 12 random bytes (per-encryption, NEVER reused)
// Tag: 128-bit authentication tag (detects tampering)
```

**Decryption:**
```typescript
// Decrypts and validates integrity — throws on tampered data
const embedding = CryptoManager.decryptEmbedding(envelope, keyHex);
```

**Why `node-forge` instead of `react-native-quick-crypto`?**  
`react-native-quick-crypto` requires JSI C++ bridge compilation, which often breaks Gradle builds on Windows. `node-forge` is 100% pure JavaScript, compiles in any environment, and runs in React Native's JS engine without any native glue.

### Offline-First SQLite Database

**File:** `src/database/sqlite/migrations.ts`

Uses `op-sqlite` for high-performance synchronous SQLite access. The schema uses versioned forward-only migrations:

| Version | Description |
|---------|-------------|
| v1 | Initial: `users`, `authentication_logs`, `sync_queue` tables |
| v2 | Added `attendance` table with `CHECK_IN`/`CHECK_OUT` and `punctuality` fields |
| v3 | Added `status` column to `users` (`ACTIVE`/`DEACTIVATED`) |
| v4 | Added `encrypted_envelope_json` to `users` for E2EE biometric storage |

**Repositories (Data Access Objects):**

| Repository | Purpose |
|-----------|---------|
| `UserRepository` | CRUD for enrolled users, sync status updates |
| `AttendanceRepository` | Insert/query check-in/check-out records |
| `AuthLogRepository` | Immutable audit log of every scan attempt |
| `SyncQueueRepository` | Enqueue, dequeue, mark synced/failed sync jobs |
| `SettingsRepository` | Persistent key-value app settings store |

### Cloud Sync Engine

**File:** `src/services/sync/SyncManager.ts`

Manages two-way reconciliation between the device's SQLite store and the cloud PostgreSQL database.

**Sync Cycle (triggered on network restore or pull-to-refresh):**

```
Phase 1: PUSH — Local → Cloud
  1. Read all PENDING jobs from sync_queue table
  2. Group into user batches and attendance log batches
  3. POST batches to /api/v1/sync/users and /api/v1/sync/attendance
  4. On success: mark SQLite records as SYNCED, clear queue entries
  5. On failure: mark queue entries as FAILED (will retry next cycle)

Phase 2: PULL — Cloud → Local
  1. GET /api/v1/sync/users → all enrolled remote profiles
  2. For each profile NOT in local SQLite:
     → Decrypt encrypted_envelope_json using org master key (on-device!)
     → Insert decrypted user with embedding into local SQLite
     → Mark as SYNCED
  3. Local matching now works for users enrolled on OTHER devices ✓
```

---

## 🔐 E2EE Cloud Sync Architecture

```
Mobile Device A (Registration)          Cloud (PostgreSQL)
┌─────────────────────────────┐         ┌────────────────────────────┐
│ 1. Capture face photo       │         │ users table:               │
│ 2. On-device ONNX inference │         │  id: "uuid-123"            │
│ 3. Get 128D embedding       │         │  name: "Alice"             │
│ 4. Derive AES key (PBKDF2)  │  POST   │  encrypted_envelope_json:  │
│    from org password        │ ──────► │   {                        │
│ 5. AES-256-GCM encrypt emb  │         │    ciphertext: "Xk2p...",  │
│ 6. Upload {name + envelope} │         │    iv: "Ab3m...",          │
│    (NO RAW EMBEDDING!)      │         │    tag: "Zx9T..."          │
└─────────────────────────────┘         │   }                        │
                                        └──────────────┬─────────────┘
Mobile Device B (Verification)                         │ GET
┌─────────────────────────────┐                        │
│ 7. Sync pull from cloud     │◄──────────────────────┘
│ 8. Download envelopes       │
│ 9. Derive same AES key      │
│    (same org password)      │
│10. Decrypt embedding        │
│    → 128D float32 vector    │
│11. Store in local SQLite    │
│12. Match face offline ✓     │
└─────────────────────────────┘
```

**Security Properties:**
- ✅ Raw biometric vectors **never leave the device** in plaintext
- ✅ Cloud database is fully **blind** — cannot decrypt or identify any person
- ✅ Each enrollment uses a **unique 12-byte random IV** — identical faces produce different ciphertext
- ✅ **AES-GCM authentication tag** detects any tampering or data corruption before decryption
- ✅ Key derivation uses **PBKDF2 with 10,000 iterations** — resistant to brute force
- ✅ **Same password on any device** derives the same key, enabling seamless cross-device sharing

---

## 🐘 PostgreSQL Setup (Local)

### Prerequisites
- PostgreSQL 13+ installed and running on `localhost:5432`

### 1. Create the Database

Connect to PostgreSQL as the admin user and create the database:
```bash
psql -U postgres
```
```sql
CREATE DATABASE blink_db;
\q
```

Or use our automated script (from the backend directory):
```bash
conda activate faceauth
python -c "
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
conn = psycopg2.connect('postgresql://postgres:YOUR_PASSWORD@localhost:5432/postgres')
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
conn.cursor().execute('CREATE DATABASE blink_db')
conn.close()
print('Database created!')
"
```

### 2. Configure Environment

Edit [`src/.env`](src/.env):
```env
# Format: postgresql://[user]:[password]@[host]:[port]/[database]
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/blink_db
```

### 3. Test Connection & Create Tables

```bash
conda activate faceauth
python src/test_db_conn.py
```

Expected output:
```
[DIAG] Attempting connection to: localhost:5432/blink_db (credentials hidden)
[SUCCESS] Connection established! PostgreSQL database is reachable.
[DIAG] Synchronizing database tables...
[SUCCESS] Tables synchronized/created successfully.
```

### 4. Verify Tables Exist

In `psql`:
```sql
\c blink_db
\dt
```
You should see:
```
 Schema |    Name    | Type  |  Owner
--------+------------+-------+----------
 public | attendance | table | postgres
 public | users      | table | postgres
```

---

## ☁️ AWS Deployment Guide

The backend ships with a production-grade `Dockerfile`. The recommended cloud topology uses **AWS App Runner** (for the FastAPI server) and **Amazon RDS** (for PostgreSQL).

### Architecture

```
                        AWS Cloud
         ┌──────────────────────────────────┐
         │  AWS App Runner                  │
         │  ┌────────────────────────────┐  │
         │  │  FastAPI Container          │  │
         │  │  Built from: Dockerfile     │  │
         │  │  Port: 8000 (HTTPS auto)    │  │
         │  └──────────┬─────────────────┘  │
         │             │ DATABASE_URL env    │
         │  ┌──────────▼─────────────────┐  │
         │  │  Amazon RDS PostgreSQL      │  │
         │  │  blink_db                   │  │
         │  │  (private VPC subnet)       │  │
         │  └────────────────────────────┘  │
         └──────────────────────────────────┘
                        ▲
                        │ HTTPS POST /api/v1/sync/...
               📱 Mobile App (React Native)
```

### Step 1: Provision Amazon RDS (PostgreSQL)

1. Go to **AWS Console → RDS → Create database**
2. Select **PostgreSQL**, template: **Free Tier**
3. Settings:
   - **DB identifier**: `blink-db-instance`
   - **Master username**: `postgres`
   - **Master password**: `<your-secure-password>`
4. Connectivity: **Public access: Yes** (for initial setup)
5. Click **Create database** (takes ~5 minutes)
6. Once active, copy the **Endpoint** (e.g., `blink-db-instance.xxxxx.us-east-1.rds.amazonaws.com`)

### Step 2: Security Group — Allow Port 5432

In the RDS security group:
- Add inbound rule: **PostgreSQL (TCP 5432)** from your App Runner security group

### Step 3: Deploy Backend via AWS App Runner

1. Go to **AWS Console → App Runner → Create service**
2. Source: **Source code repository** → Connect GitHub
3. Select your `BLINK` repository, branch `main`
4. Build settings: **Runtime: Custom** (uses your `Dockerfile`)
5. Service settings:
   - **Port**: `8000`
   - **Environment variables** → Add:
     ```
     Key:   DATABASE_URL
     Value: postgresql://postgres:<password>@blink-db-instance.xxxxx.us-east-1.rds.amazonaws.com:5432/blink_db
     ```
6. Click **Create & Deploy**

App Runner will build the Docker image in the cloud (installing OpenCV, MediaPipe, ONNX Runtime), start the server, and give you a **free HTTPS URL** like:
```
https://xxxxxxxxxx.us-east-1.awsapprunner.com
```

### Step 4: Update Mobile App

In `BLINK-RN/blink_app/src/services/api/ApiClient.ts`, update the base URL:

```typescript
// Before (local development)
export const apiClient = new ApiClient('http://192.168.x.x:8000/api/v1');

// After (AWS production)
export const apiClient = new ApiClient('https://xxxxxxxxxx.us-east-1.awsapprunner.com/api/v1');
```

### Step 5: Verify Deployment

Test the live health endpoint:
```bash
curl https://xxxxxxxxxx.us-east-1.awsapprunner.com/health
# → {"status": "healthy", "engine": "BLINK Core", "simulated_fallback": false}
```

---

## 🚀 Quick Start

### Python Backend

```bash
# 1. Clone and enter directory
cd BLINK/

# 2. Create / activate conda environment
conda create -n faceauth python=3.10 -y
conda activate faceauth

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download model weights (if not present)
python src/download_weights.py

# 5. Set up environment
cp src/.env.example src/.env
# Edit src/.env with your PostgreSQL credentials

# 6. Start the server
python src/server.py
# → Server running at http://0.0.0.0:8000
# → API docs at http://localhost:8000/docs
```

### React Native Mobile App

```bash
cd BLINK-RN/blink_app/

# 1. Install JS dependencies
npm install

# 2. Android build
npx react-native run-android

# 3. Or iOS build
cd ios && pod install && cd ..
npx react-native run-ios
```

> **Note:** The mobile app requires the ONNX model `mobilefacenet.ort` bundled in `android/app/src/main/assets/`. Run `python src/export_mobile.py` from the backend to generate it.

### Desktop Real-time Demo (Webcam)

```bash
conda activate faceauth
python src/main.py
# Opens webcam window with real-time face detection, liveness, and attendance flow
```

---

## 🔑 Environment Variables

| Variable | Location | Required | Description |
|----------|----------|----------|-------------|
| `DATABASE_URL` | `src/.env` | Yes (for cloud sync) | PostgreSQL connection string |

**Format:**
```env
DATABASE_URL=postgresql://username:password@host:port/database
```

**Examples:**
```env
# Local development
DATABASE_URL=postgresql://postgres:mypassword@localhost:5432/blink_db

# AWS RDS
DATABASE_URL=postgresql://postgres:mypassword@blink-db.xxxxx.us-east-1.rds.amazonaws.com:5432/blink_db
```

**Fallback behavior:** If `DATABASE_URL` is not set, the server uses a local `cloud_blink.db` SQLite file automatically.

---

## 📡 API Reference

### `POST /api/v1/embeddings`

Upload a face image and receive its 128D embedding vector.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | JPEG/PNG face photo |

**Response:**
```json
{
  "success": true,
  "embedding": [0.123, -0.456, 0.789, ...],  // 128 float values
  "dimension": 128,
  "model_precision": "int8"
}
```

---

### `POST /api/v1/sync/users`

Batch upsert E2EE encrypted user profiles.

**Request body:**
```json
[
  {
    "id": "uuid-string",
    "name": "Alice Johnson",
    "encrypted_envelope_json": {
      "ciphertext": "Base64...",
      "iv": "Base64...",
      "tag": "Base64..."
    },
    "model_mode": "ort",
    "enrolled_at": "2024-01-15T10:30:00Z",
    "status": "ACTIVE"
  }
]
```

**Response:**
```json
{
  "success": true,
  "synced_count": 1,
  "synced_ids": ["uuid-string"]
}
```

---

### `GET /api/v1/sync/users`

Fetch all enrolled E2EE profiles from the cloud.

**Response:**
```json
{
  "success": true,
  "users": [
    {
      "id": "uuid-string",
      "name": "Alice Johnson",
      "encrypted_envelope_json": { "ciphertext": "...", "iv": "...", "tag": "..." },
      "model_mode": "ort",
      "enrolled_at": "2024-01-15T10:30:00Z",
      "status": "ACTIVE"
    }
  ]
}
```

---

### `POST /api/v1/sync/attendance`

Batch upsert offline attendance records.

**Request body:**
```json
[
  {
    "id": "att-uuid",
    "person_id": "user-uuid",
    "timestamp": "2024-01-15T09:02:33Z",
    "status": "CHECK_IN",
    "punctuality": "ON_TIME"
  }
]
```

---

## 🛡️ Security Design

| Attack Vector | Defense |
|--------------|---------|
| **Database breach** | AES-256-GCM encrypted envelopes — ciphertext is meaningless without the device key |
| **Man-in-the-middle** | HTTPS enforced on all cloud communication (App Runner provides TLS automatically) |
| **Photo spoofing** | Liveness detection: eye blink (EAR) + head yaw challenge |
| **Replay attacks** | Unique 12-byte random IV per encryption — same face produces different ciphertext |
| **Weak keys** | PBKDF2 with 10,000 iterations and SHA-256 — computationally expensive to brute-force |
| **Data tampering** | AES-GCM 128-bit authentication tag — decryption fails if envelope is modified |
| **Unauthorized sync** | All sync endpoints require the encrypted envelope — plaintext biometrics are never accepted |

---

## 📦 Model Files

| File | Size | Precision | Speed | Description |
|------|------|-----------|-------|-------------|
| `models/mobilefacenet.onnx` | ~4MB | FP32 | Baseline | Full precision model |
| `models/mobilefacenet_int8.onnx` | ~1MB | INT8 | ~2× faster | Quantized for edge deployment |
| `android/assets/mobilefacenet.ort` | ~1MB | INT8/FP32 | Mobile | ORT-format for React Native |

**Download weights:**
```bash
python src/download_weights.py
```

**Quantize to INT8:**
```bash
python src/quantize.py
```

**Export to mobile ORT format:**
```bash
python src/export_mobile.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and test thoroughly
4. Commit with a descriptive message: `git commit -m "feat: add multi-face tracking"`
5. Push and open a Pull Request

---

## 📄 License

This project is developed for hackathon/research purposes. All biometric data handling follows privacy-by-design principles — raw biometrics never leave the device unencrypted.

---

<div align="center">

**Built with ❤️ — Powered by MobileFaceNet, MediaPipe, ONNX Runtime, FastAPI, and React Native**

</div>
