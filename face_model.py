from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import List
from helpers import l2_normalize

def init_face_app(det_size=(640, 640)) -> FaceAnalysis:
    """
    Initialize a FaceAnalysis app (ArcFace embeddings + RetinaFace detector).
    Uses GPU if onnxruntime-gpu is installed and CUDA available.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def build_ref_embeddings(app: FaceAnalysis, refs_dir: str) -> List[np.ndarray]:
    if not os.path.isdir(refs_dir):
        raise FileNotFoundError(f"refs_dir not found: {refs_dir}")
    ref_paths = [os.path.join(refs_dir, f) for f in os.listdir(refs_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    if not ref_paths:
        raise RuntimeError(f"No reference images found in {refs_dir}")
    embs = []
    print(f"[Refs] Loading {len(ref_paths)} reference images...")
    for p in tqdm(ref_paths):
        img = cv2.imread(p)
        if img is None:
            print(f"  [WARN] Could not read {p}")
            continue
        faces = app.get(img)
        if not faces:
            print(f"  [WARN] No face detected in {p}")
            continue
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = l2_normalize(faces[0].normed_embedding.astype(np.float32))
        embs.append(emb)
    if not embs:
        raise RuntimeError("No usable embeddings from reference images.")
    print(f"[Refs] Ready: {len(embs)} embeddings.")
    return embs
