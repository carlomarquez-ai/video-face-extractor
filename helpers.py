import os
import cv2
import numpy as np
import subprocess
from typing import List, Tuple

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def resize_keep(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    if max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame, 1.0
    scale = max_side / float(m)
    out = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale

def is_video_file(name: str) -> bool:
    return name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mpg", ".mpeg"))

def cut_clip_ffmpeg(src: str, start: float, end: float, out_path: str, reencode: bool, vcodec: str, acodec: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", src,
            "-c:v", vcodec,
            "-c:a", acodec,
            "-preset", "veryfast",
            "-movflags", "+faststart",
            out_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", src,
            "-c", "copy",
            out_path
        ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)

def group_timestamps(timestamps: List[float], gap: float) -> List[Tuple[float, float]]:
    if not timestamps:
        return []
    timestamps.sort()
    segs = []
    s = timestamps[0]
    prev = s
    for t in timestamps[1:]:
        if t - prev > gap:
            segs.append((s, prev))
            s = t
        prev = t
    segs.append((s, prev))
    return segs

def pad_filter_segments(segs: List[Tuple[float, float]], duration: float,
                        pre: float, post: float, min_dur: float) -> List[Tuple[float, float]]:
    out = []
    for s, e in segs:
        s2 = max(0.0, s - pre)
        e2 = min(duration, e + post)
        if e2 - s2 >= min_dur:
            out.append((s2, e2))
    return out
