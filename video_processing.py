import cv2
import os
import json
from typing import List, Dict
from tqdm import tqdm
from helpers import resize_keep, cut_clip_ffmpeg
from face_model import build_ref_embeddings, init_face_app
import numpy as np

def frame_has_match(app, frame_bgr: np.ndarray, refs: List[np.ndarray], sim_thresh: float) -> bool:
    faces = app.get(frame_bgr)
    if not faces:
        return False
    from helpers import l2_normalize, cosine_sim
    for f in faces:
        emb = l2_normalize(f.normed_embedding.astype(np.float32))
        sims = [cosine_sim(emb, r) for r in refs]
        if max(sims) >= sim_thresh:
            return True
    return False

def process_video(video_path: str, cfg: Dict) -> Dict:
    app = init_face_app(det_size=cfg["det_size"])
    refs = build_ref_embeddings(app, cfg["refs_dir"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0
    frame_skip = max(1, int(cfg["frame_skip"]))
    max_side = int(cfg["max_side"])
    timestamps = []
    idx = 0
    processed = 0
    expect_iters = total_frames // frame_skip if total_frames else 0
    print(f"[Video] {os.path.basename(video_path)} | fps={fps:.2f}, frames={total_frames}, duration={duration/3600:.2f}h")
    pbar = tqdm(total=expect_iters, desc=f"Scanning {os.path.basename(video_path)}", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            if max_side > 0:
                frame2, _ = resize_keep(frame, max_side)
            else:
                frame2 = frame
            if frame_has_match(app, frame2, refs, cfg["cosine_sim_threshold"]):
                t = idx / fps
                timestamps.append(t)
            pbar.update(1)
            processed += 1
        idx += 1
    cap.release()
    pbar.close()
    from helpers import group_timestamps, pad_filter_segments
    raw = group_timestamps(timestamps, cfg["gap_seconds"])
    segs = pad_filter_segments(raw, duration, cfg["pre_pad"], cfg["post_pad"], cfg["min_duration"])
    base = os.path.splitext(os.path.basename(video_path))[0]
    clip_dir = os.path.join(cfg["output_dir"], base)
    os.makedirs(clip_dir, exist_ok=True)
    clips = []
    for i, (s, e) in enumerate(segs, 1):
        out_path = os.path.join(clip_dir, f"{base}_clip_{i:03d}.mp4")
        cut_clip_ffmpeg(video_path, s, e, out_path, cfg["reencode"], cfg["vcodec"], cfg["acodec"])
        clips.append({"index": i, "start": round(s, 3), "end": round(e, 3), "path": out_path})
    log = {
        "video": video_path,
        "fps": fps,
        "duration_sec": duration,
        "frame_skip": frame_skip,
        "threshold": cfg["cosine_sim_threshold"],
        "segments": clips,
    }
    with open(os.path.join(cfg["output_dir"], f"{base}_segments.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[Done] {os.path.basename(video_path)}  {len(clips)} clips")
    return log
