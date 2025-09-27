# Hack4DK
Code from hack4DK 2025 - Sorry i didn't clean the code, i just focused on getting something that worked. 


# 🔍 Image Matching Pipeline

This project finds the **best matching image** in a large library (e.g. 40,000+ pictures) for every query image in another folder.  
It uses a **multi-step funnel approach**: fast methods first, deeper checks later, to balance **speed and accuracy**.

---

## Features
- Handles **exact duplicates** and **near-duplicates** (resized, cropped, compressed).  
- Multi-stage pipeline:  
  1. **SHA-256 fingerprints** – instant exact-match detection.  
  2. **Perceptual hashes (pHash, dHash, aHash)** – quick look-alike filtering.  
  3. **AI embeddings (OpenCLIP)** – deeper semantic similarity (cosine).  
  4. **ORB + RANSAC** – detailed feature matching (corners, edges).  
  5. **Final decision** – best candidate chosen, with progress logging.  
- Configurable thresholds and top-k narrowing: **10 → 7 → 4 → 1**.  
- Progress logs every 100 images.  
- Writes results to a clean **TSV file** for analysis.

---

## Project Structure
```text
.
├── pipeline.py       # main script
├── result.tsv        # output results (after running)
└── README.md         # this file

## Dependencies 
pip install pillow imagehash opencv-python-headless numpy torch open-clip-torch

## How It Works

### Step 1 – SHA-256 Fingerprints
- Hash each file using `SHA-256`.
- If two images share the same fingerprint, they are **bit-for-bit identical**.
- **Method:** `SHA-256`

### Step 2 – Perceptual Hashing
- Create three visual hashes for each image:
  - **pHash** (perceptual hash; DCT-based)
  - **dHash** (difference hash; pixel gradients)
  - **aHash** (average hash; brightness)
- Compare with **Hamming distance** (bit differences) and keep the **Top-10** closest.
- **Methods:** `pHash`, `dHash`, `aHash`, `Hamming distance`

### Step 3 – AI Vision (Embeddings)
- Convert images to vector embeddings using **OpenCLIP (ViT-B/32)**.
- Compare embeddings via **cosine similarity** and keep the **Top-7**.
- **Methods:** `OpenCLIP`, `cosine similarity`

### Step 4 – ORB + RANSAC
- Detect and match local features with **ORB**.
- Filter with **Lowe’s ratio test** and validate geometry with **RANSAC homography**.
- Keep the **Top-4** with the most inliers.
- **Methods:** `ORB`, `Lowe ratio test`, `RANSAC`

### Step 5 – Final Pick
- If an exact `SHA-256` match is found at any point, it wins.
- Otherwise, select the best-scoring candidate from Step 4.
- **Output:** 1 best match per query.

---

## Output

Results are saved to `result.tsv`:

| Query File | Stage1 Top10 | Stage2 Top7 | Stage3 Top4 | FinalPick | Reason |
|-----------:|:-------------|:------------|:------------|:----------|:-------|
| `img1.jpg` | `a.jpg, …`   | `a.jpg, …`  | `a.jpg, …`  | `a.jpg`   | Exact |
| `img2.jpg` | `b.jpg, …`   | `b.jpg, …`  | `b.jpg, …`  | `b.jpg`   | BestCandidate |

---

## Configuration

Adjust constants at the top of `pipeline.py`:

- `HASH_SIZE` – perceptual hash size (default `16` → 256 bits)
- `HASH_TOP` – candidates kept after Step 2 (default `10`)
- `EMB_TOP` – candidates kept after embeddings (default `7`)
- `ORB_TOP` – candidates kept after ORB (default `4`)
- `DEBUG` – set to `True` to print detailed progress

---

## ⚡ Performance Notes
- `SHA-256` and perceptual hashing are **very fast** (good for 40k+ images).
- OpenCLIP embeddings are **slower**; use a **GPU** if available.
- ORB is the **slowest** but runs only on a tiny shortlist (Top-7 → Top-4).
- Progress logs appear every 100 images.

---
