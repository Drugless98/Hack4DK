# Image matching pipeline (rotation-aware, threaded, per-image flush)
# Stages: SHA-256 exact → Hash Top-10 → Embedding Top-7 → ORB Top-4 → VLM final pick
# Prereqs:
#   pip install pillow imagehash opencv-python-headless numpy torch open-clip-torch
#   (optional) ollama + model: ollama pull moondream
# Usage: adjust the paths at the bottom and run: python this_file.py

import os
import csv
import math
import asyncio
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps, UnidentifiedImageError
import imagehash
import numpy as np

# OpenCV (ORB)
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# OpenCLIP (embeddings)
try:
    import torch
    import open_clip
    OPENCLIP_AVAILABLE = True
except Exception:
    OPENCLIP_AVAILABLE = False

# Ollama (final pick with VLM)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# ------------------ Config ------------------
THREADS = 4  # up to 4 threads

# Shortlist sizes: 10 → 7 → 4 → 1
HASH_TOP = 10
EMB_TOP  = 7
ORB_TOP  = 4

HASH_SIZE = 16
TOTAL_BITS = HASH_SIZE * HASH_SIZE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
OUTPUT_TSV = "result_rot_aware_async.tsv"
DEBUG = True
LOG_EVERY = 100  # progress log frequency

# Rotation handling (applied to the QUERY only for speed)
ROTATIONS = [0, 90, 180, 270]  # degrees

# ORB params
ORB_NFEATURES = 4000
RATIO_TEST = 0.75
RANSAC_REPROJ = 5.0
# -------------------------------------------


# ----------------- Utilities ----------------
def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def safe_open_rgb(path: Path):
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception:
        return None

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def phash_triplet(im: Image.Image):
    try:
        ph = imagehash.phash(im, hash_size=HASH_SIZE)
        dh = imagehash.dhash(im, hash_size=HASH_SIZE)
        ah = imagehash.average_hash(im, hash_size=HASH_SIZE)
        return int(str(ph),16), int(str(dh),16), int(str(ah),16)
    except (UnidentifiedImageError, OSError, ValueError):
        return None

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def best_hash_distance(hq, hc) -> int:
    # lower is better
    return min(hamming(hq[i], hc[i]) for i in range(3))


# -------------- Precompute (threaded) --------------
def precompute_sha_map(paths):
    from collections import defaultdict
    sha_map = defaultdict(list)

    def work(p: Path):
        try:
            return (p, sha256_file(p))
        except Exception:
            return (p, None)

    if DEBUG:
        print(f"Precomputing SHA-256 for Library 2 ({len(paths)}) with {THREADS} threads…")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(work, p) for p in paths]
        done = 0
        for fut in as_completed(futures):
            p, sh = fut.result()
            if sh:
                sha_map[sh].append(p)
            done += 1
            if DEBUG and done % 1000 == 0:
                print(f"  … {done}/{len(paths)}")
    return sha_map

def precompute_hash_triplets(paths):
    cache = {}

    def work(p: Path):
        im = safe_open_rgb(p)
        if im is None:
            return (p, None)
        return (p, phash_triplet(im))

    if DEBUG:
        print(f"Precomputing p/d/a-hash triplets for Library 2 ({len(paths)}) with {THREADS} threads…")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(work, p) for p in paths]
        done = 0
        for fut in as_completed(futures):
            p, hs = fut.result()
            cache[p] = hs
            done += 1
            if DEBUG and done % 1000 == 0:
                print(f"  … {done}/{len(paths)}")
    return cache


# -------------- Rotation-aware hashing --------------
def query_rotated_hashes(qpath: Path):
    im = safe_open_rgb(qpath)
    if im is None:
        return []
    hashes = []
    for ang in ROTATIONS:
        rim = im if ang == 0 else im.rotate(ang, expand=True)
        hs = phash_triplet(rim)
        if hs:
            hashes.append(hs)
    return hashes

def rank_by_hash_topk(qpath: Path, lib2_hashes, k=HASH_TOP):
    q_hashes = query_rotated_hashes(qpath)
    if not q_hashes:
        return list(lib2_hashes.keys())[:k]

    items = list(lib2_hashes.items())

    def score(item):
        p2, h2 = item
        if not h2:
            return math.inf
        # best distance over all query rotations
        best = min(best_hash_distance(hq, h2) for hq in q_hashes)
        return best

    scored = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(score, it) for it in items]
        for fut, it in zip(as_completed(futures), items):
            s = fut.result()
            scored.append((it[0], s))

    scored.sort(key=lambda x: x[1])  # lower is better
    return [p for p, _ in scored[:k]]


# ---------------- Embeddings (OpenCLIP) ---------------
_clip_model = None
_clip_preprocess = None
_clip_device = None
_emb_cache = {}  # (path, rotation) -> vec (np.ndarray)

def _load_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device
    if not OPENCLIP_AVAILABLE:
        return None, None, None
    name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=_clip_device
    )
    _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_device

def embed_image(path: Path, rotation: int = 0):
    key = (path, rotation)
    if key in _emb_cache:
        return _emb_cache[key]
    m, preprocess, device = _load_clip()
    if m is None:
        return None
    im = safe_open_rgb(path)
    if im is None:
        return None
    if rotation:
        im = im.rotate(rotation, expand=True)
    with torch.no_grad():
        t = preprocess(im).unsqueeze(0).to(device)
        vec = m.encode_image(t)
        vec = torch.nn.functional.normalize(vec, dim=-1)
        arr = vec[0].detach().cpu().numpy()
    _emb_cache[key] = arr
    return arr

def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0

def rank_by_embeddings_topk(qpath: Path, candidates, k=EMB_TOP):
    if not OPENCLIP_AVAILABLE:
        return candidates[:k]

    # Precompute all 4 rotations for the query
    q_vecs = {ang: embed_image(qpath, ang) for ang in ROTATIONS}

    def score(p2: Path) -> float:
        v2 = embed_image(p2, 0)  # candidate unrotated (consistency/speed)
        # take the best cosine across query rotations
        return max(cosine(q_vecs[ang], v2) for ang in ROTATIONS)

    scored = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(score, p): p for p in candidates}
        for fut in as_completed(futures):
            p = futures[fut]
            s = fut.result()
            scored.append((p, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:k]]


# ---------------- ORB (rotation-robust) ----------------
class ORBIndex:
    def __init__(self):
        if OPENCV_AVAILABLE:
            self.orb = cv2.ORB_create(ORB_NFEATURES)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.cache = {}

    def _compute(self, path: Path):
        if not OPENCV_AVAILABLE:
            return None, None
        if path in self.cache:
            return self.cache[path]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.cache[path] = (None, None)
            return None, None
        k, d = self.orb.detectAndCompute(img, None)
        self.cache[path] = (k, d)
        return k, d

    def inliers(self, p1: Path, p2: Path) -> int:
        if not OPENCV_AVAILABLE:
            return -1
        k1, d1 = self._compute(p1)
        k2, d2 = self._compute(p2)
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            return 0
        matches = self.bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance] if matches else []
        if len(good) < 4:
            return 0
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
        return int(mask.sum()) if mask is not None else 0

def rank_by_orb_topk(qpath: Path, candidates, orb: ORBIndex, k=ORB_TOP):
    if not OPENCV_AVAILABLE:
        return candidates[:k]

    def score(p2: Path) -> int:
        try:
            return orb.inliers(qpath, p2)
        except Exception:
            return 0

    scored = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = {ex.submit(score, p): p for p in candidates}
        for fut in as_completed(futures):
            p = futures[fut]
            s = fut.result()
            scored.append((p, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:k]]


# ------------- Final pick via Ollama (VLM) -------------
async def final_pick_vlm(qpath: Path, last_four):
    """
    Ask a light VLM (moondream) to pick the best match among the final 4.
    Falls back to the first candidate if Ollama is unavailable or parsing fails.
    """
    if not (OLLAMA_AVAILABLE and last_four):
        return last_four[0] if last_four else None

    try:
        prompt = (
            "You are an image comparison assistant. You will be given ONE query image and FOUR candidate images. "
            "Choose exactly ONE candidate that is the same photo or the closest match to the query (consider rotation, crop, compression). "
            "Reply ONLY with the number 1, 2, 3, or 4.\n"
            "Query is first; then candidates are in order."
        )
        msg = {
            "role": "user",
            "content": prompt,
            "images": [str(qpath)] + [str(p) for p in last_four]
        }
        client = ollama.AsyncClient()
        resp = await client.chat(model="moondream", messages=[msg])
        text = resp["message"]["content"].strip()
        choice = None
        for ch in text:
            if ch in "1234":
                choice = int(ch) - 1
                break
        if choice is None:
            return last_four[0]
        return last_four[choice]
    except Exception:
        return last_four[0]


# ----------------------- Main ------------------------
def map_libraries_pipeline(lib1_dir: str, lib2_dir: str, out_tsv: str = OUTPUT_TSV):
    lib1 = Path(lib1_dir)
    lib2 = Path(lib2_dir)

    qlist = list_images(lib1)
    lib2_items = list_images(lib2)
    if not qlist or not lib2_items:
        print("One of the libraries is empty.")
        return

    # Precompute for lib2 (threaded)
    sha_map2 = precompute_sha_map(lib2_items)
    lib2_hashes = precompute_hash_triplets(lib2_items)

    orb = ORBIndex()
    total = len(qlist)
    print(f"\nStarting matching: {total} query images…")

    # open output file once, write header; then append per image
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Lib1_File", "Stage2_Hash_Top10", "Stage3_Embed_Top7", "Stage4_ORB_Top4", "FinalPick", "FinalReason"])

        for idx, q in enumerate(qlist, 1):
            row = None
            try:
                print(f"\n[{idx}/{total}] {q.name}")

                # Fast exact check via SHA map
                try:
                    q_sha = sha256_file(q)
                    exacts = sha_map2.get(q_sha, [])
                    if exacts:
                        print(f"  SHA-256 exact: {exacts[0].name}")
                        row = [q.name, "-", "-", "-", exacts[0].name, "Exact"]
                        w.writerow(row); f.flush()
                        if idx % LOG_EVERY == 0:
                            print(f"--- Progress: {idx}/{total} ---")
                        continue
                except Exception:
                    pass

                # Step 2: Rotation-aware perceptual hashing → Top-10
                top10 = rank_by_hash_topk(q, lib2_hashes, k=HASH_TOP)
                if DEBUG:
                    print("  Hash Top-10:", ", ".join(p.name for p in top10))

                # Step 3: Rotation-aware embeddings → Top-7
                top7 = rank_by_embeddings_topk(q, top10, k=EMB_TOP)
                if DEBUG:
                    print("  Embed Top-7:", ", ".join(p.name for p in top7))

                # Step 4: ORB (rotation robust) → Top-4
                top4 = rank_by_orb_topk(q, top7, orb, k=ORB_TOP)
                if DEBUG:
                    print("  ORB Top-4:", ", ".join(p.name for p in top4))

                # Step 5: Final pick with light VLM (moondream)
                if top4:
                    final = asyncio.run(final_pick_vlm(q, top4))
                else:
                    final = None

                row = [
                    q.name,
                    ", ".join(p.name for p in top10),
                    ", ".join(p.name for p in top7),
                    ", ".join(p.name for p in top4),
                    (final.name if final else ""),
                    "VLM" if (final and OLLAMA_AVAILABLE) else "BestCandidate"
                ]

            except Exception as e:
                print(f"  [ERROR] Failed on {q.name}: {e}")
                row = [q.name, "", "", "", "ERROR", "Failed"]

            # write each row immediately
            w.writerow(row)
            f.flush()

            if idx % LOG_EVERY == 0:
                print(f"--- Progress: {idx}/{total} images processed ---")

    print(f"\nWrote {out_tsv}. Processed {len(qlist)} images.")


# ------------------- Run -------------------
if __name__ == "__main__":
    map_libraries_pipeline(
        r"C:\SHARES\Development\HACK4DK\DAMGAARD\Damgaard\jpegDamgaardResized",
        r"C:\SHARES\Development\HACK4DK\holger\previews\combi",
        OUTPUT_TSV
    )
