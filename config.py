CONFIG = {
    # IO
    "refs_dir": "./refs",
    "videos_dir": "./videos",
    "output_dir": "./out",

    # Processing speed/quality
    "frame_skip": 25,            # analyze every Nth frame (1 = every frame). 2~5 is good for speed
    "max_side": 720,            # resize long side to this for speed (0 = disable)
    "det_size": (640, 640),     # detector input size for RetinaFace (speed/accuracy tradeoff)

    # Matching
    "cosine_sim_threshold": 0.38,  # lower = stricter; increase if missing true matches (0.35~0.45 typical)
    "gap_seconds": 2.0,            # merge detections within this gap into one segment
    "pre_pad": 1.0,                # seconds before segment
    "post_pad": 1.0,               # seconds after segment
    "min_duration": 1.0,           # drop segments shorter than this after padding

    # Cutting
    "reencode": True,              # True = precise cuts; False = stream copy (fast but keyframe-limited)
    "vcodec": "libx264",
    "acodec": "aac",

    # Parallelism
    "WORKERS": 2,                  # number of parallel videos to process (1 GPU: 1-2; 2 GPUs: raise & set CUDA_VISIBLE_DEVICES per process if needed)
}
